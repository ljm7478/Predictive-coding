%% run visual_reconstruction
lambda = [0.1:0.2:0.9]; % Example lambda values from 0.1 to 10
nfold = 9; 

%% Load training fmri responses
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject1_fmri/';
load([dataroot,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
fmri_avg = (fmri.data1+fmri.data2) / 2; % average across repeats
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));


%%
clc;
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/';
layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 
saveroot='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/visual_reconstruction/'
X = fmri_avg';

for lay = 1 : length(layername)
    load([dataroot,'layers5_3_32_64_96_192channel', '/', 'Ahat', '/',  'PredNet_feature_maps_processed_layer', num2str(lay),'_Ahat', '_concatenated.mat'],'lay_feat_concatenated');

    dim = size(lay_feat_concatenated);
    Nu = prod(dim(1:end-1));% number of units
    Nf = dim(end); % number of time points
    Y = reshape(lay_feat_concatenated, Nu, Nf)';
    
    opts = struct(); % Initialize opts as an empty struct

    [W, Rmat, Errmat, Lambda] = visual_reconstruction(Y, X, lambda, nfold, opts);
    
    file_name = 'W.mat';
    save([saveroot, file_name], 'W', '-v7.3');

    disp(['finish layer', num2str(lay)])
end

