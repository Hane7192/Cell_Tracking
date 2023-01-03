clear
clc
obj.reader= load('imAge_sg_data.mat');
[ext_Z0,frameRR0,markazRR0,labelRR0]= Motiontrackermodif2() ;
load('mod_cll_track_out2.mat');
[ext_Z,frameRR,markazRR,labelRR]= Motiontrackermodif4(dx,dy,dst,fdst);
