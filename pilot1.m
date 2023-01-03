clear
clc

load('output1.mat')

temp_lbl(1:200,1)=nan; %current frame labels
temp_cntx(1:200,1)=nan;
    temp_cntx=double(temp_cntx);
temp_cnty(1:200,1)=nan;
    temp_cnty=double(temp_cnty);
    
for ii=3:83
    clear lbel cntr
    
frame = frameRR{ii};
centroidss=markazRR{ii};
    centroidss=double(centroidss);
labels=labelRR{ii};

circle1 = [centroidss , 5*ones(size(centroidss,1),1)];
    circle1=double(circle1);
circle2 = [centroidss , 2*ones(size(centroidss,1),1)];
    circle2=double(circle2);
nColor = length(centroidss);
col=jet(nColor);
             
  frame = insertObjectAnnotation(frame,'circle',...
  circle1,labels,'Color',[255.*col(:,1),255.*col(:,2),255.*col(:,3)]);
  mask = insertObjectAnnotation(frame,'circle',...
  circle2,labels,'Color','white','linewidth',7);
%%
for k=1:length(labels)
    add=str2double(labels{k});
    lbel(add,1)=add;
    cntr(add,:)=centroidss(k,:);
end
    cntr=double(cntr);

for j=1:length(lbel)
    if cntr(j,1)==0
        lbel(j,1)=j;
        cntr(j,:)=nan;
    end
end
    cntr=double(cntr);
    
cntr(end+1:200,:)=nan;
    cntr=double(cntr);
    
temp_cntx=horzcat(temp_cntx,cntr(:,1));
temp_cnty=horzcat(temp_cnty,cntr(:,2));

%%
  imshow(frame,'InitialMagnification','fit') 
  hold on
 
  r=1;
for j=1:200
    nColor = length(lbel);
    rgb=jet(nColor);
    r1=rgb(:,1);
    r2=rgb(:,2);
    r3=rgb(:,3);
    hold on
    
     %%%%%%%%%%%%%%%%%%%%%%% craeting flasher for lost cells
     if sum(isnan(temp_cntx(j,:))) < size(temp_cntx(j,:),2) 
         plot(temp_cntx(j,:),temp_cnty(j,:),'Color',[r1(r) r2(r) r3(r)],'linewidth',1);
         r=r+1;
     else
         plot(temp_cntx(j,:),temp_cnty(j,:));
     end
end

drawnow
end
















