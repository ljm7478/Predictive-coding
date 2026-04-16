
%% Concatenate CNN activation time series in the 1st layer across movie segments
% CNN layer labels
% layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 
% layername = {'/A1';'/A2';'/A3';'/A4';}; 
layername = {'/E1';'/E2';'/E3';'/E4';}; 

dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/bivariate_analysis/sub1/E/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

%% Training movies
for lay = 1 : length(layername)
    for seg = 1 : 18
        disp(['Seg: ', num2str(seg)]);
        secpath = [dataroot,'PredNet_feature_maps_processed_seg', num2str(seg),'_E', '.h5'];      
        % info = h5info(secpath);
        lay_feat = h5read(secpath,[layername{lay}, '/data']);
        dim = size(lay_feat);
        Nf = dim(end); % number of frames
        if seg == 1
           lay_feat_concatenated = zeros([dim(1:end-1),Nf*18],'single'); 
        end
        if lay <= 5
            lay_feat_concatenated(:,:,:,(seg-1)*Nf+1:seg*Nf) = lay_feat;
        else
            lay_feat_concatenated(:,(seg-1)*Nf+1:seg*Nf) = lay_feat;
        end
    end

    % check time series
    % figure;plot(squeeze(lay_feat_concatenated(25,25,56,:)));

    save([saveroot,'PredNet_feature_maps_processed_layer',num2str(lay),'_E', '_concatenated.mat'],...
        'lay_feat_concatenated','-v7.3');
end


%% Load fmri responses
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject1_fmri/';

load([dataroot,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
fmri_avg = (fmri.data1+fmri.data2) / 2; % average across repeats
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));
 
%% Map hierarchical CNN features to brain
% Correlating all the voxels to all the CNN units is time-comsuming.
% Here is the analysis given some example voxels in the visual cortex.
% select voxels
voxels = [21892, 21357, 21885, 51456, 22778, 53919 43797, 54301];

for lay = 1 : length(layername)
    load([dataroot,'PredNet_feature_maps_processed_layer',num2str(lay),'_E', '_concatenated.mat'],'lay_feat_concatenated');
    dim = size(lay_feat_concatenated);
    Nu = prod(dim(1:end-1));% number of units
    Nf = dim(end); % number of time points
    lay_feat_concatenated = reshape(lay_feat_concatenated, Nu, Nf)';

    Rmat = zeros([Nu,length(voxels)]);
    k1 = 1;
    while k1 <= length(voxels)
       disp(['Layer: ',num2str(lay),'; Voxel: ', num2str(k1)]);
       k2 = min(length(voxels), k1+100);
       R = amri_sig_corr(lay_feat_concatenated, fmri_avg(voxels(k1:k2),:)');
       Rmat(:,k1:k2) = R;
       k1 = k2+1;
    end
    save([saveroot, 'cross_corr_fmri_cnn_layer',num2str(lay),'_E', '.mat'], 'Rmat', '-v7.3');
end

lay_corr = zeros(length(layername),length(voxels));
for lay = 1 : length(layername)
    disp(['Layer: ',num2str(lay)]);
    load([saveroot, 'cross_corr_fmri_cnn_layer',num2str(lay),'_E', '.mat'],'Rmat');
    lay_corr(lay,:) = max(Rmat,[],1);
end

save([saveroot, 'cross_corr_fmri_cnn_E.mat'], 'lay_corr');

% Assign layer index to each voxel
[~,layidx] = max(lay_corr,[],1);


% % Display correlation profile for example voxels
% figure(100);
% for v = 1 : length(voxels)
%     plot(1:length(layername),lay_corr(:,v)','--o');
%     title(v);
%     pause;
% end


%% jmlee: Map hierarchical CNN features to brain - whole voxels (59412)
% CNN layer labels
% layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 
% layername = {'/A1';'/A2';'/A3';'/A4';}; 
layername = {'/E1';'/E2';'/E3';'/E4';}; 

dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/bivariate_analysis/E/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

voxels = 1:59412;

for lay = 1 : length(layername)
    load([saveroot,'PredNet_feature_maps_processed_layer',num2str(lay),'_E', '_concatenated.mat'],'lay_feat_concatenated');
    dim = size(lay_feat_concatenated);
    Nu = prod(dim(1:end-1));% number of units
    Nf = dim(end); % number of time points
    lay_feat_concatenated = reshape(lay_feat_concatenated, Nu, Nf)';

    Rmat = zeros([Nu,length(voxels)]);

    Rmat = amri_sig_corr(lay_feat_concatenated, fmri_avg(voxels,:)');

    save([saveroot, 'jmlee_sub1_cross_corr_fmri_cnn_layer',num2str(lay),'_E', '.mat'], 'Rmat', '-v7.3');
end

% Initialize a matrix to hold the maximum correlation for each voxel across layers
lay_corr = zeros(length(layername),length(voxels));

for lay = 1 : length(layername)
    disp(['Layer: ',num2str(lay)]);
    load([saveroot, 'jmlee_sub1_cross_corr_fmri_cnn_layer',num2str(lay),'_E', '.mat'],'Rmat');
    lay_corr(lay,:) = max(Rmat,[],1);

    % Assign layer index to each voxel
    [~,layidx] = max(lay_corr,[],1);
end

save([saveroot, 'jmlee_sub2_cross_max_corr_fmri_cnn_E.mat'], 'lay_corr');
save([saveroot, 'jmlee_sub2_cross_max_corr_layidx_fmri_cnn_E.mat'], 'layidx');

% 
% % Assign layer index to each voxel
% [~,layidx] = max(lay_corr,[],1);


% Display correlation profile for example voxels
% figure(100);
% for v = 1 : length(voxels)
%     plot(1:length(layername),lay_corr(:,v)','--o');
%     title(v);
%     pause;
% end
clc;
%% jmlee : load subject1 cii for wb_view visualize 
clc;clear;
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/';

%load cii structure 
subject_id='subject1';
fmripath = '/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/subject1/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');
%% make max_lay_corr_ for Whole brain 59k dtseries for visualization
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

%load max_lay_corr
load([dataroot, 'jmlee_sub1_cross_max_corr_fmri_cnn_E.mat'],'lay_corr');
size(lay_corr)

%save back into dtseries for visualization - laycorr
cii.cdata = lay_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(dataroot, [subject_id,'_WH_max_laycorr_E.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');

%% make max_lay_idx_ for Whole brain 59k dtseries for visualization
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

%load max_lay_idx
load([dataroot, 'jmlee_sub1_cross_max_corr_layidx_fmri_cnn_E.mat'],'layidx');

%save back into dtseries for visualization - layidx
cii.cdata = layidx';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(dataroot, [subject_id,'_WH_max_layidx_Ahat.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');

%% only visual region mapping 
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

label_32k= ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
vertex=label_32k.cdata;
 
total_area = find(vertex == 1 | vertex == 181 | ...
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
% find key for vertex 
%  if istable(label_32k.diminfo{1, 2}.maps.table)
%     % If it's a table and 'name' is a column of type string or char
%     tableData = label_32k.diminfo{1, 2}.maps.table;
%     filteredData = tableData(strcmp(tableData.name, 'L_EC_ROI'), :);
% elseif isstruct(label_32k.diminfo{1, 2}.maps.table)
%     % If it's a structure array
%     structData = label_32k.diminfo{1, 2}.maps.table;
%     indices = arrayfun(@(x) strcmp(x.name, 'L_EC_ROI'), structData);
%     filteredStructData = structData(indices);
% end

%% make visual only max_lay_corr_ for Whole brain 59k dtseries for visualization
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

%load max_lay_corr
load([dataroot, 'jmlee_sub1_cross_max_corr_fmri_cnn_E.mat'],'lay_corr');
size(lay_corr)

%corr
visual_corr_idx = zeros(size(lay_corr()));
for i = 1:4
    visual_idx = zeros(size(lay_corr(i,:)));
    lay_corr_tmp = lay_corr(i, :);
    visual_idx(total_area) = lay_corr_tmp(total_area);
    tmp_idx = find(visual_idx == 0);
    visual_idx(tmp_idx) = -1;
    visual_corr_idx(i,:) = visual_idx;
end

%dtseries save -corr
cii.cdata = visual_corr_idx';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(dataroot, [subject_id,'_visual_max_laycorr_E.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');

% % Display correlation profile for example voxels
% figure(100);
% for v = 1 : length(voxels)
%     plot(1:length(layername),lay_corr(:,v)','--o');
%     title(v);
%     % pause;
% end
%% make  visual only max_lay_idx_ for Whole brain 59k dtseries for visualization
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/E/';

%load max_lay_idx
load([dataroot, 'jmlee_sub1_cross_max_corr_layidx_fmri_cnn_E.mat'],'layidx');

visual_idx = zeros(size(layidx));
visual_idx(total_area) = layidx(total_area);
tmp_idx = find(visual_idx == 0);
visual_idx(tmp_idx) = -1;

%dtseries save -idx
cii.cdata = visual_idx';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(dataroot, [subject_id,'_visual_max_layidx_E.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');

% % Display correlation profile for example voxels
% figure(100);
% for v = 1 : length(voxels)
%     plot(1:length(layername),layidx(:,v)','--o');
%     title(v);
%     % pause;
% end


%% compare max_lay_idx in visual only between prediction vs error  - tmp need some modification
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/';

%load max_lay_idx
prediction=load([dataroot, 'Ahat/','jmlee_cross_max_corr_layidx_fmri_cnn_Ahat.mat']).layidx;
error=load([dataroot, 'E/','jmlee_sub1_cross_max_corr_layidx_fmri_cnn_E.mat']).layidx;

concat_pred_error = cat(1, prediction, error);  % first row-prediction ; second row-error;

for lay = 1 : 2
    disp(['Layer: ',num2str(lay)]);
    % Assign mas layer ; 1-prediction, 2-error
    [~,layidx] = max(concat_pred_error,[],1);
end

visual_idx = zeros(size(layidx));
visual_idx(total_area) = layidx(total_area);
tmp_idx = find(visual_idx == 0);
visual_idx(tmp_idx) = -1;

%dtseries save -idx
cii.cdata = visual_idx';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(dataroot, [subject_id,'_visual_max_Ahat_vs_Error.dtseries.nii']);
ciftisavereset(cii, (save_file_name), 'wb_command');
