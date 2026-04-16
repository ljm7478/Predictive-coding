clc;clear

%% Load single test fmri responses
subject_id = 'subject1';
dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/', subject_id, '_fmri/'];

load([dataroot,'testing_fmri.mat'],'fmritest'); % from movie_fmri_processing.m

fmritest1_avg = mean(fmritest.test1, 3); % Averaging across the 10 repeats
% fmritest2_avg = mean(fmritest.test2, 3); % Averaging across the 10 repeats
% fmritest3_avg = mean(fmritest.test3, 3); % Averaging across the 10 repeats
% fmritest4_avg = mean(fmritest.test4, 3); % Averaging across the 10 repeats
% fmritest5_avg = mean(fmritest.test5, 3); % Averaging across the 10 repeats

%% Y_test and correaltion _all test
clc;
layername = {'/E1', '/E2', '/E3', '/E4'};  

subject_id = 'sub1';

%modified-32c
% dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/E/';
% saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/test1/test1_E_WH_time/', subject_id, '/'];

%modified-96c
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/E/layer1_4/';
saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/test1/test1_E_WH_time/', subject_id, '/'];

num_layers = length(layername);

all_corr = cell(num_layers, 1);
for lay = 1:num_layers
    secpath = [dataroot, 'PredNet_feature_maps_pcareduced_test1.h5'];
    if exist(secpath, 'file')
        Y_test = h5read(secpath, [layername{lay}, '/data']);
    else
        disp(['File does not exist: ', secpath]);
    end


    % Load weights matrix
    W_filename = [dataroot, subject_id, '/',  'W_lambda5_', layername{lay}(2:end), '.mat'];
    load(W_filename, 'W');
    
    % Compute predicted BOLD using the averaged feature maps and the weights matrix
    pred_fmri = Y_test * W;
    
    
    % Check dimensions match
    if size(pred_fmri, 1) ~= size(fmritest1_avg, 1)
        fmritest1_avg=fmritest1_avg';
    end


    correlation_scores = zeros(1, size(pred_fmri, 2));  % Preallocate a vector for correlations
    for v = 1:size(pred_fmri, 2)
        correlation_scores(v) = corr(pred_fmri(:, v), fmritest1_avg(:, v));  % Pearson correlation for each voxel
    end
    all_corr{lay} = correlation_scores;

    file_name = [subject_id,'_corr_lambda5_', layername{lay}(2:end) ,'.mat'];
    save([saveroot, file_name], 'correlation_scores', '-v7.3');
end

concat_corr = cat(1, all_corr{1,1}, all_corr{2,1}, all_corr{3,1}, all_corr{4,1});


%% dtseries 
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

%load cii structure 
subject_id='subject1';
fmripath =['/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');

% concat_corr = fmri_avg;
%save back into dtseries for visualization - layidx
cii.cdata = concat_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(saveroot, [subject_id,'_lambda5_E.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');
