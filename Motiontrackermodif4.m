function [ext_Z,frameRR,markazRR,labelRR]= Motiontrackermodif4(dx,dy,dst,fdst)

warning off
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();
obj.reader= load('mod_cll_track_out2.mat');
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
% Detect moving objects, and track them across video frames.
linek=[];
for ii=1:83
    frame = readFrame(ii);
    [centroidss, bboxes, mask] = detectObjects(frame);
    centroidss=obj.center.centroids{ii};
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();

    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    
    ext_Z{ii}=createNewTracks();
    
    
    displayTrackingResults();
end
  
    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create a video file reader.
            obj.reader= load('mod_cll_track_out2.mat');
            obj.center= load('mod_cll_track_out2.mat');
            
        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.maskPlayer = vision.VideoPlayer('Position', [20, 20, 1077, 900]);
        obj.videoPlayer = vision.VideoPlayer('Position', [50, 50, 1077, 600]);

        % Create System objects for foreground detection and blob analysis

        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 10, 'MinimumBackgroundRatio', 0.7);

        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 5);
        
    end

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'markaz', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
% %                    
    end

    function frame= readFrame(ii)

        aa=sprintf('obj.reader.im%d',ii);
        vv = evalin('base', aa);
        VV=vv(:,:,1);
        frame = vv;
        
    end

    function [centroidss, bboxes, mask] = detectObjects(frame)
% %         % Detect foreground.
% %         % Apply morphological operations to remove noise and fill in holes.
% %         % Perform blob analysis to find connected components.
    framemdf = frame;
    
    framemdf=im2single(framemdf);
    V=framemdf;
    V=rgb2gray(V);
    I=V;
    I=imadjust(I,[.4 .7],[0 1]);
    
    [~, threshold] = edge(I, 'sobel');
    fudgeFactor = 0.7;
    BWs = edge(I,'sobel', threshold * fudgeFactor);
    
    se90 = strel('line', 3, 90);
    se0 = strel('line', 3, 0);

    BWsdil = imdilate(BWs, [se90 se0]);
    BWdfill = imfill(BWsdil, 'holes');
    BWnobord = imclearborder(BWdfill, 4);
    
    seD = strel('diamond',1);
    BWfinal = imerode(BWnobord,seD);
    BWfinal = imerode(BWfinal,seD);
    Z= BWfinal;
    Z=im2double(Z);
    Z = imbinarize(Z);
    Z = bwareaopen(Z, 10);
    Z = imfill(Z ,'holes');
    s = regionprops(Z ,{'Centroid'});
    mask=Z;
    [~, centroidss, bboxes] = obj.blobAnalyser.step(mask);

    end

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            markaz =tracks(i).markaz;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid); 
                tracks(i).markaz = predictedCentroid;
        end
    end

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroidss, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroidss);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroidss(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).markaz = centroid;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

    
%%    Delete lost tracks
    function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 3;
        ageThreshold = 3;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.1) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end
%%
    function extract_er = createNewTracks()
        centroidss = centroidss(unassignedDetections, :);

        for i = 1:size(centroidss, 1)
                centroid = centroidss(i,:);

            % Create a Kalman filter object.
if ii<3
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 10], [100, 5], 100);
else
    dL1=20*(max(dst(:,ii))-min(dst(:,ii)));
    dL2=dL1/2;
    fns=0.5*max(fdst);
    kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [dL1, 10], [dL2, 5], fns);
end
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'markaz', centroid,...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
        extract_er=tracks;
    end
    function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 2;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                centroidss = cat(1, reliableTracks.markaz);
                
                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                      circle1 = [centroidss , 5*ones(size(centroidss,1),1)];
                      circle2 = [centroidss , 2*ones(size(centroidss,1),1)];
                  
                  nColor = length(ids);
                  col=jet(nColor);
%                    
                  frame = insertObjectAnnotation(frame,'circle',...
                  circle1,labels,'Color',[255.*col(:,1),255.*col(:,2),255.*col(:,3)]);
              
              
                  c=obj.reader.c;
                  nColor = length(c{ii});
                  col=jet(nColor);
                               
                % Draw the objects on the mask.

                mask = insertObjectAnnotation(mask,'circle',...
                  circle2,labels,'Color','white','linewidth',7);
                     %***********
                     markazRR{ii}=centroidss;
                     labelRR{ii}=labels;

            end
        end
    
        % Display the mask and the frame.
        obj.videoPlayer.step(frame);
        frameRR{ii}=frame;
       
    end

pause(1)
end