clc;clear
%% Load training fmri responses
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject1_fmri/';
load([dataroot,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
fmri_avg = (fmri.data1+fmri.data2) / 2; % average across repeats
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));

%% voxelwise_encoding - training with total training movies (18seg)- per layer
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/Ahat/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/Ahat_noPCA/';

layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 

lambda = [0.1:0.2:0.9];
nfold = 9; %from paper

for lay = 1 : length(layername)
    disp(['Layer: ', num2str(lay)]);
    load([dataroot,'PredNet_feature_maps_processed_layer',num2str(lay),'_Ahat', '_concatenated.mat'],'lay_feat_concatenated');
    
    dim = size(lay_feat_concatenated);
    Nu = prod(dim(1:end-1));% number of units
    Nf = dim(end); % number of time points
    Y_train = reshape(lay_feat_concatenated, Nu, Nf)';

    dim = size(Y_train) % #time-by-#components

    [W, Rmat, Lambda] = voxelwise_encoding(Y_train, fmri_avg', lambda, nfold);

    % save W (feature x voxel)
    file_name = ['W_', layername{lay}(2:end) , '.mat']
    save([saveroot, file_name], 'W', '-v7.3');

    disp(['save W for layer ', num2str(lay)]);
end

%% Load test fmri responses
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject1_fmri/';
load([dataroot,'testing_fmri.mat'],'fmritest'); % from movie_fmri_processing.m

% fmritest1_rep1 = fmritest.test1(:,:,1);
% fmritest2_rep1 = fmritest.test2(:,:,1);
% fmritest3_rep1 = fmritest.test3(:,:,1);
% fmritest4_rep1 = fmritest.test4(:,:,1);
% fmritest5_rep1 = fmritest.test5(:,:,1);
% fmri_cat = cat(2, fmritest1_avg, fmritest2_avg, fmritest3_avg, fmritest4_avg, fmritest5_avg);

% Averaging across repetitions for each segment
fmri_avg = zeros(size(fmritest.test1, 1), 240, 5);
for i = 1:5
    segment = fmritest.(sprintf('test%d', i));
    fmri_avg(:, :, i) = mean(segment, 3);  % Average across third dimension (repetitions)
end
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));

%% Y_test 
clc;
layername = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};  % Example layer names
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/bivariate_analysis/Ahat/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/Ahat_noPCA/';

num_layers = length(layername);
num_segs = 5;  % Number of segments

all_corr = cell(num_layers, 1); 
for lay = 1:num_layers
    all_Y_test = [];
    for seg = 1:num_segs
        secpath = [dataroot, 'PredNet_feature_maps_processed_test', num2str(seg), '_Ahat', '.h5']
        if exist(secpath, 'file')
            Y_test = h5read(secpath, [layername{lay}, '/data']);
            
            dim=size(Y_test)
            Nu = prod(dim(1:end-1));% number of units
            Nf = dim(end); % number of time points
            Y_test = reshape(Y_test, Nu, Nf)';
            dim=size(Y_test)

            all_Y_test = [all_Y_test; Y_test];  % Concatenate along the first dimension (time)
        else
            disp(['File does not exist: ', secpath]);
        
        end
    end
    % Load weights matrix
    W_filename = [saveroot, 'W_', layername{lay}(2:end), '.mat'];
    load(W_filename, 'W');

    % Compute predicted BOLD using the averaged feature maps and the weights matrix
    pred_fmri = all_Y_test * W;
    
    % Check dimensions match
    if size(pred_fmri, 1) ~= size(fmri_avg, 1)
        fmri_avg=fmri_avg';
    end

    correlation_scores = zeros(1, size(pred_fmri, 2));  % Preallocate a vector for correlations
    for v = 1:size(pred_fmri, 2)
        correlation_scores(v) = corr(pred_fmri(:, v), fmri_avg(:, v));  % Pearson correlation for each voxel
    end
    all_corr{lay} = correlation_scores; 

    % % Iterate only over specific voxels
    % correlation_scores = -1 * ones(1, size(fmri_avg, 2));
    % for voxel = 1:numel(specific_voxels)
    %     voxel_index = specific_voxels(voxel);
    %     % Calculate correlation only for the specified voxels
    %     correlation_scores(voxel_index) = corr(pred_fmri(:, voxel_index), fmri_avg(:, voxel_index));
    % end
    % all_corr{lay} = correlation_scores; 
end

concat_corr = cat(1, all_corr{1,1}, all_corr{2,1}, all_corr{3,1}, all_corr{4,1});

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

%load cii structure 
subject_id='subject1';
fmripath = '/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/subject1/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');

% concat_corr = fmri_avg;
%save back into dtseries for visualization - layidx
cii.cdata = concat_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(saveroot, [subject_id,'_Ahat.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');
%% make predicted BOLD
%test pcareduced feature
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 

for lay = 1 : length(layername)
    disp(['Layer: ', num2str(lay)]);
    secpath = [saveroot,'PredNet_feature_maps_pcareduced_concatenated_test.h5'];  
    Y_test = h5read(secpath,[layername{lay},'/data']);% #time-by-#components

    filename=['W_', layername{lay}(2:end) , '.mat'];
    load([saveroot, filename], 'W');

    disp(['Y_test size: ',num2str(size(Y_test)),'; W size: ', num2str(size(W))]);

    pred_fmri = Y_test * W; 

    % save pred_fmri (feature x voxel)
    file_name = ['predfMRI_', layername{lay}(2:end) , '.mat']
    save([saveroot, file_name], 'pred_fmri', '-v7.3');
end



%% Correlation between predicted BOLD and measured BOLD
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'};
fmri_cat = fmri_cat';  % Assuming fmri_cat is initially Nv-by-Nt, transpose to Nt-by-Nv

for lay = 1:length(layername)
    disp(['Layer: ', num2str(lay)]);

    filename = ['predfMRI_', layername{lay}(2:end), '.mat'];
    load([saveroot, filename], 'pred_fmri');
    disp(['pred fMRI size: ', num2str(size(pred_fmri))]);

    % Check dimensions match
    if size(pred_fmri, 1) ~= size(fmri_cat, 1)
        error('Mismatch in number of time points between predicted and actual fMRI data');
    end

    R = zeros(1, size(pred_fmri, 2));  % Preallocate a vector for correlations
    for v = 1:size(pred_fmri, 2)
        R(v) = corr(pred_fmri(:, v), fmri_cat(:, v));  % Pearson correlation for each voxel
    end

    file_name = ['corr_avg_rep_', layername{lay}(2:end), '.mat'];
    save([saveroot, file_name], 'R', '-v7.3');
    disp(['Saved correlation data for ', layername{lay}(2:end)]);
end

%% Visual area correaltion between predicted BOLD and measured BOLD
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'};
fmri_cat = fmri_cat';  % Assuming fmri_cat is initially Nv-by-Nt, transpose to Nt-by-Nv

label_32k= ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
vertex=label_32k.cdata;

specific_voxels = find(vertex == 1 | vertex == 181 | ...
    vertex == 3 | vertex == 183 | ...
    vertex == 4 | vertex == 184 | vertex == 16 | vertex == 196 | ...
    vertex == 7 | vertex == 187 | vertex == 19 | vertex == 199 | ...
    vertex == 23 | vertex == 203 | vertex == 2 | vertex == 182 | ...
    vertex == 17 | vertex == 197 | vertex == 20 | vertex == 200 | ...
    vertex == 21 | vertex == 201 | vertex == 46 | vertex == 226 | ...
    vertex == 142 | vertex == 322 | vertex == 145 | vertex == 325 | ...
    vertex == 146 | vertex == 326 | vertex == 152 | vertex == 332 | ...
    vertex == 155 | vertex == 335 | vertex == 31 | vertex == 211 | ...
    vertex == 95 | vertex == 275 | vertex == 122 | vertex == 302 | vertex == 121 | vertex == 301 | ...
    vertex == 119 | vertex == 299 | vertex == 126 | vertex == 306 | vertex == 127 | vertex == 307 | ...
    vertex == 132 | vertex == 133 | vertex ==134 | vertex == 135 | vertex == 136 | ...
    vertex == 312 | vertex == 313 | vertex == 314| vertex == 315 | vertex == 316| ...
    vertex == 142 | vertex == 143 | vertex == 322 | vertex == 323 | ...
    vertex == 139 | vertex == 140 | vertex == 141 | ...
    vertex == 319 | vertex == 320 | vertex == 321 | ...
    vertex == 150 | vertex == 151 | vertex == 330 | vertex == 331 | ...
    vertex == 152 | vertex == 332 | vertex == 153 | vertex == 154 | ...
    vertex == 156 | vertex == 333 | vertex == 334 | vertex == 336 | ...
    vertex == 160 | vertex == 340 | vertex == 163 | vertex == 343| vertex == 230 | vertex == 50 |...
    vertex == 229 | vertex == 49 | vertex == 339 | vertex == 159 | ...
    vertex == 202 | vertex == 22 | vertex == 337 | vertex == 157| ...
    vertex == 352 | vertex == 172 | vertex == 357 | vertex == 177| ...
    vertex == 131 | vertex == 311 | vertex == 15 | vertex == 195 | ...
    vertex == 300 | vertex == 120| vertex == 202 | vertex == 317 | vertex == 318 | vertex == 137 | vertex == 138| ...
    vertex == 13 | vertex == 19 | vertex == 158 | vertex == 5 | vertex == 193 | vertex == 199 | vertex == 338 | vertex == 185 |...
    vertex == 6 | vertex == 156 | vertex == 186 | vertex == 336 | ...
    vertex==18 | vertex==198| vertex == 298 | vertex == 118);


for lay = 1:length(layername)
    disp(['Layer: ', num2str(lay)]);

    filename = ['predfMRI_', layername{lay}(2:end), '.mat'];
    load([saveroot, filename], 'pred_fmri');
    disp(['pred fMRI size: ', num2str(size(pred_fmri))]);

    % Check dimensions match
    if size(pred_fmri, 1) ~= size(fmri_cat, 1)
        error('Mismatch in number of time points between predicted and actual fMRI data');
    end

    R = -1 * ones(1, size(fmri_cat, 2));

    % Iterate only over specific voxels
    for idx = 1:numel(specific_voxels)
        voxel_index = specific_voxels(idx);
        % Calculate correlation only for the specified voxels
        R(voxel_index) = corr(pred_fmri(:, voxel_index), fmri_cat(:, voxel_index));
    end

    file_name = ['corr_avg_visual_', layername{lay}(2:end), '.mat'];
    save([saveroot, file_name], 'R', '-v7.3');
    disp(['Saved correlation data for ', layername{lay}(2:end)]);
end
 %% jmlee : wb_view visualize 
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

%load cii structure 
subject_id='subject1';
fmripath = '/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/subject1/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');

% lay1=load([saveroot, 'corrAhat1.mat']).R;
% lay2=load([saveroot, 'corrAhat2.mat']).R;
% lay3=load([saveroot, 'corrAhat3.mat']).R;
% lay4=load([saveroot, 'corrAhat4.mat']).R;

lay1=load([saveroot, 'corr_avg_visual_Ahat1.mat']).R;
lay2=load([saveroot, 'corr_avg_visual_Ahat1.mat']).R;
lay3=load([saveroot, 'corr_avg_visual_Ahat3.mat']).R;
lay4=load([saveroot, 'corr_avg_visual_Ahat4.mat']).R;

test_corr = cat(1, lay1, lay2, lay3, lay4);

%save back into dtseries for visualization - layidx
cii.cdata = test_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(saveroot, [subject_id,'_visual_avg_test_corr.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');