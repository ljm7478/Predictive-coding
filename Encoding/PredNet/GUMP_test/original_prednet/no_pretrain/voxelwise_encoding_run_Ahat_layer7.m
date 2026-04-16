clc;clear
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for sub = 1:length(num_subjects)
    % Load training fmri responses
    subject_id = num_subjects{sub};
    disp(['current sub is ', subject_id])
    
    fmriroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/', subject_id];
    load([fmriroot, '/', subject_id, '_per_run_fmri.mat'],'fmri_data'); 
    
    train_fmri = cat(2,fmri_data.run1, fmri_data.run2, fmri_data.run3, fmri_data.run4);
    clc;
    % voxelwise_encoding - training with total training movies (18seg)- per layer
    clc;    
    % layername decide
    % layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'}; 
    layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6'}; 
    if length(layername) == 4 
        layer_name = 'layer4';
    elseif length(layername) == 7 
        layer_name = 'layer7';
    end

    dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result/',layer_name, '/Ahat/'];
    
    lambda = [0.1:0.2:0.9];
    nfold = 3; 
    
    for lay = 1 : length(layername)
        disp(['Layer: ', num2str(lay)]);
        secpath = [dataroot,'PredNet_feature_maps_pcareduced_concatenated.h5'];  
        Y_train = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
        dim = size(Y_train)
    
        [W, Rmat, Lambda] = voxelwise_encoding(Y_train, train_fmri', lambda, nfold);
    
        % save W (feature x voxel)
        file_name = ['W_lambda5_', layername{lay}(2:end) , '.mat']
        saveroot = [dataroot, subject_id,'/']; 
        if ~exist(saveroot, 'dir')
            mkdir(saveroot)
        end
        save([saveroot, file_name], 'W', '-v7.3');
    end
    
    %% Load and process test fmri responses
    clc;
    dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result/',layer_name, '/Ahat/'];

    all_corr = cell(length(layername), 1); 
    
    for run = 5:8
        for lay = 1:length(layername)
            secpath = [dataroot, 'PredNet_feature_maps_pcareduced_run', num2str(run), '.h5'];
    
            if exist(secpath, 'file')
                Y_test = h5read(secpath, [layername{lay}, '/data']);
            else
                disp(['File does not exist: ', secpath]);
                continue;
            end
    
            % Load fMRI test data
            test_fmri = fmri_data.(['run', num2str(run)]);
    
            % Load trained weights
            W_filename = [saveroot, 'W_lambda5_', layername{lay}(2:end), '.mat'];
            load(W_filename, 'W');
    
            % Compute predicted BOLD
            pred_fmri = Y_test * W;
    
            % Ensure dimension alignment
            if size(pred_fmri, 1) ~= size(test_fmri, 1)
                test_fmri = test_fmri';
            end
    
            correlation_scores = zeros(1, size(pred_fmri, 2));  % Preallocate a vector for correlations
            for v = 1:size(pred_fmri, 2)
                correlation_scores(v) = corr(pred_fmri(:, v), test_fmri(:, v));  % Pearson correlation for each voxel
            end
            all_corr{lay} = correlation_scores;
        end    
        %save concat_corr
        concat_corr = cat(1, all_corr{:});
        file_name = [subject_id,'_corr_lambda5_run_', num2str(run), '.mat'];
        save([saveroot, file_name], 'concat_corr', '-v7.3');
    end 
    
    %% Compute average correlation per test (combined across layers)
    clc;
    
    dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result/',layer_name, '/Ahat/'];

    per_layer = cell(length(layername), 4); % Store for each layer across runs
    all_corr = cell(length(layername), 1); 
    all_avg_corr = cell(length(layername), 1); 
    
    % Load all correlation data for each layer and run
    for lay = 1:length(layername)
        for run = 5:8
            % Load the correlation file for this run and layer
            test_predicted_fmri_filename = fullfile(dataroot, subject_id,...
                [subject_id,'_corr_lambda5_run_', num2str(run), '.mat']);
            
            disp([subject_id,'_corr_lambda5_run_', num2str(run), '.mat'])
    
            if exist(test_predicted_fmri_filename, 'file')
                load(test_predicted_fmri_filename, 'concat_corr');
                per_layer{lay, run-4} = concat_corr(lay,:); % Store by layer and run index
            else
                warning(['File not found: ', test_predicted_fmri_filename]);
            end
        end
    end
    
    % Compute average correlation per layer
    for lay = 1:length(layername)
        valid_runs = ~cellfun(@isempty, per_layer(lay, :)); % Check which runs exist
        if any(valid_runs)
            concat_corr = cat(3, per_layer{lay, valid_runs}); % Concatenate available runs
            avg_corr = mean(concat_corr, 3); % Average over runs
            
            all_avg_corr{lay} = avg_corr; % Assign to correct layer column
        else
            warning(['No valid correlation data found for layer: ', num2str(lay)]);
        end
    end
    
    concat_corr = cat(1, all_avg_corr{:});
    
    % Save the single concatenated correlation matrix (voxel, 4)
    file_name = [subject_id, '_avgcorr_lambda5.mat'];
    save(fullfile(dataroot, subject_id, file_name), 'concat_corr', '-v7.3');
    
    %% make cdata for brain mapping 
    addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
    addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
    addpath(genpath('/local_raid1/01_software/spm12'));
    addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
    clc;
    
    % load cii structure 
    fmripath ='/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/'; % path to the cifti files
    filename = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
    cii = ciftiopen([fmripath, filename],'wb_command');
    
    for run = 5:8 
        % load concat_corr per test run 
        file_name = [subject_id,'_corr_lambda5_run_', num2str(run), '.mat'];
        load(fullfile(dataroot, subject_id, file_name), 'concat_corr');
        
        % save into dtseries 
        run_cdata = cii; 
        run_cdata.cdata = concat_corr';
        run_cdata.diminfo{1, 1}.length = size(run_cdata.cdata, 1);
        run_cdata.diminfo{1, 2}.length = size(run_cdata.cdata, 2);
        run_cdata.diminfo{1, 1}.models(3:end) = [];
        save_file_name = fullfile(dataroot, subject_id, [subject_id,'_lambda5_avg_Ahat_run_', num2str(run), '.dtseries.nii']);
        ciftisavereset(run_cdata, (save_file_name), 'wb_command');
    end 
    
    % load avg concat_corr
    file_name = [subject_id, '_avgcorr_lambda5.mat'];
    load(fullfile(dataroot,subject_id, file_name), 'concat_corr');
    
    % save into dtseries
    avg_cdata = cii;
    avg_cdata.cdata = concat_corr';
    avg_cdata.diminfo{1, 1}.length = size(avg_cdata.cdata, 1);
    avg_cdata.diminfo{1, 2}.length = size(avg_cdata.cdata, 2);
    avg_cdata.diminfo{1, 1}.models(3:end) = [];
    save_file_name = fullfile(dataroot, subject_id, [subject_id,'_lambda5_avg_Ahat.dtseries.nii']);
    ciftisavereset(avg_cdata, (save_file_name), 'wb_command');
end



%% visual area correaltion 
% layername = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};  
% 
% subject_id = 'sub3';
% 
% dataroot='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';
% saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/', subject_id, '/'];
% 
% num_layers = length(layername);
% num_segs = 5;  % Number of segments
% 
% label_32k= ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
% vertex=label_32k.cdata;
% 
% specific_voxels = find(vertex == 1 | vertex == 181 | ...
%     vertex == 3 | vertex == 183 | ...
%     vertex == 4 | vertex == 184 | vertex == 16 | vertex == 196 | ...
%     vertex == 7 | vertex == 187 | vertex == 19 | vertex == 199 | ...
%     vertex == 23 | vertex == 203 | vertex == 2 | vertex == 182 | ...
%     vertex == 17 | vertex == 197 | vertex == 20 | vertex == 200 | ...
%     vertex == 21 | vertex == 201 | vertex == 46 | vertex == 226 | ...
%     vertex == 142 | vertex == 322 | vertex == 145 | vertex == 325 | ...
%     vertex == 146 | vertex == 326 | vertex == 152 | vertex == 332 | ...
%     vertex == 155 | vertex == 335 | vertex == 31 | vertex == 211 | ...
%     vertex == 95 | vertex == 275 | vertex == 122 | vertex == 302 | vertex == 121 | vertex == 301 | ...
%     vertex == 119 | vertex == 299 | vertex == 126 | vertex == 306 | vertex == 127 | vertex == 307 | ...
%     vertex == 132 | vertex == 133 | vertex ==134 | vertex == 135 | vertex == 136 | ...
%     vertex == 312 | vertex == 313 | vertex == 314| vertex == 315 | vertex == 316| ...
%     vertex == 142 | vertex == 143 | vertex == 322 | vertex == 323 | ...
%     vertex == 139 | vertex == 140 | vertex == 141 | ...
%     vertex == 319 | vertex == 320 | vertex == 321 | ...
%     vertex == 150 | vertex == 151 | vertex == 330 | vertex == 331 | ...
%     vertex == 152 | vertex == 332 | vertex == 153 | vertex == 154 | ...
%     vertex == 156 | vertex == 333 | vertex == 334 | vertex == 336 | ...
%     vertex == 160 | vertex == 340 | vertex == 163 | vertex == 343| vertex == 230 | vertex == 50 |...
%     vertex == 229 | vertex == 49 | vertex == 339 | vertex == 159 | ...
%     vertex == 202 | vertex == 22 | vertex == 337 | vertex == 157| ...
%     vertex == 352 | vertex == 172 | vertex == 357 | vertex == 177| ...
%     vertex == 131 | vertex == 311 | vertex == 15 | vertex == 195 | ...
%     vertex == 300 | vertex == 120| vertex == 202 | vertex == 317 | vertex == 318 | vertex == 137 | vertex == 138| ...
%     vertex == 13 | vertex == 19 | vertex == 158 | vertex == 5 | vertex == 193 | vertex == 199 | vertex == 338 | vertex == 185 |...
%     vertex == 6 | vertex == 156 | vertex == 186 | vertex == 336 | ...
%     vertex==18 | vertex==198| vertex == 298 | vertex == 118);
% 
% all_corr = cell(num_layers, 1); 
% for lay = 1:num_layers
%     all_Y_test = [];
%     for seg = 1:num_segs
%         secpath = [dataroot, 'PredNet_feature_maps_pcareduced_test', num2str(seg), '.h5'];
%         if exist(secpath, 'file')
%             Y_test = h5read(secpath, [layername{lay}, '/data']);
%             all_Y_test = [all_Y_test; Y_test];  % Concatenate along the first dimension (time)
%         else
%             disp(['File does not exist: ', secpath]);
% 
%         end
%     end
%     % Load weights matrix
%     W_filename = [saveroot, 'W_lambda5_', layername{lay}(2:end), '.mat'];
%     load(W_filename, 'W');
% 
%     % Compute predicted BOLD using the averaged feature maps and the weights matrix
%     pred_fmri = all_Y_test * W;
% 
%     % Check dimensions match
%     if size(pred_fmri, 1) ~= size(fmri_avg, 1)
%         fmri_avg=fmri_avg';
%     end
% 
%     % Iterate only over specific voxels
%     correlation_scores = -1 * ones(1, size(fmri_avg, 2));
%     for voxel = 1:numel(specific_voxels)
%         voxel_index = specific_voxels(voxel);
%         % Calculate correlation only for the specified voxels
%         correlation_scores(voxel_index) = corr(pred_fmri(:, voxel_index), fmri_avg(:, voxel_index));
%     end
%     all_corr{lay} = correlation_scores;
% 
%     file_name = [subject_id,'_visual_corr_lambda5_', layername{lay}(2:end) ,'.mat'];
%     save([saveroot, file_name], 'correlation_scores', '-v7.3');
% end
% 
% concat_corr = cat(1, all_corr{1,1}, all_corr{2,1}, all_corr{3,1}, all_corr{4,1});
% 
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
% addpath(genpath('/local_raid1/01_software/spm12'));
% addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
% clc;
% 
% %load cii structure 
% subject_id='subject1';
% fmripath =['/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
% seg = 1;
% filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
% cii = ciftiopen([fmripath, filename],'wb_command');
% 
% % concat_corr = fmri_avg;
% %save back into dtseries for visualization - layidx
% cii.cdata = concat_corr';
% cii.diminfo{1, 1}.length = size(cii.cdata, 1);
% cii.diminfo{1, 2}.length = size(cii.cdata, 2);
% cii.diminfo{1, 1}.models(3:end) = [];
% save_file_name = fullfile(saveroot, [subject_id,'_visual_lambda5_Ahat.dtseries.nii']);
% ciftisavereset(cii, (save_file_name), 'wb_command');
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Total_sub processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %% total sub encoding 
% clc;clear
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';
% 
% layername = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};
% subjects = {'sub1';'sub2';'sub3';};
% 
% num_layers = length(layername);
% num_subjects = length(subjects);
% 
% corr_avg = cell(num_layers, 1);
% 
% for lay = 1:num_layers
%     sum_values = 0; % Initialize sum for each layer
% 
%     for s = 1:num_subjects
%         subject_id = subjects{s};
%         disp(subject_id);
% 
%         % Define file path and load correlation
%         dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/', subject_id, '/'];
%         file_name = [subject_id, '_corr_lambda5_', layername{lay}(2:end), '.mat'];
%         data = load([dataroot, file_name]);
%         correlations = data.correlation_scores;
% 
%         sum_values = sum_values + correlations';
%     end
% 
%     % Calculate average correlations
%     corr_avg{lay} = sum_values / num_subjects;
% end
% % Save the average correlations
% save([saveroot, sprintf('%s_avg_corr.mat', layername{lay}(2:5))], 'corr_avg', '-v7.3');
% 
% % Reset corr_avg
% corr_avg = {};
% 
% %% total sub _dtseries -WHbrain
% clc;clear
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';
% 
% corr_avg = load([saveroot, 'Ahat_avg_corr.mat']).corr_avg;
% corr_avg = cat(2, corr_avg{1,1}, corr_avg{2,1}, corr_avg{3,1}, corr_avg{4,1});
% 
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
% addpath(genpath('/local_raid1/01_software/spm12'));
% addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
% clc;
% 
% %load cii structure 
% subject_id='subject1';
% fmripath =['/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
% seg = 1;
% filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
% cii = ciftiopen([fmripath, filename],'wb_command');
% 
% %save back into dtseries for visualization - layidx
% cii.cdata = corr_avg;
% cii.diminfo{1, 1}.length = size(cii.cdata, 1);
% cii.diminfo{1, 2}.length = size(cii.cdata, 2);
% cii.diminfo{1, 1}.models(3:end) = [];
% save_file_name = fullfile(saveroot, 'total_sub_lambda5_Ahat.dtseries.nii');
% ciftisavereset(cii, (save_file_name), 'wb_command');
% 
% 
% %% total sub encoding 
% clc;clear
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';
% 
% layername = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};
% subjects = {'sub1';'sub2';'sub3';};
% 
% num_layers = length(layername);
% num_subjects = length(subjects);
% 
% corr_avg = cell(num_layers, 1);
% 
% for lay = 1:num_layers
%     sum_values = 0; % Initialize sum for each layer
% 
%     for s = 1:num_subjects
%         subject_id = subjects{s};
%         disp(subject_id);
% 
%         % Define file path and load correlation
%         dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/', subject_id, '/'];
%         file_name = [subject_id, '_visual_corr_lambda5_', layername{lay}(2:end), '.mat'];
%         data = load([dataroot, file_name]);
%         correlations = data.correlation_scores;
% 
%         sum_values = sum_values + correlations';
%     end
% 
%     % Calculate average correlations
%     corr_avg{lay} = sum_values / num_subjects;
% end
% % Save the average correlations
% save([saveroot, sprintf('%s_visual_avg_corr.mat', layername{lay}(2:5))], 'corr_avg', '-v7.3');
% 
% % Reset corr_avg
% corr_avg = {};
% 
% 
% %% total sub _dtseries -visual area
% clc;clear
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';
% 
% corr_avg = load([saveroot, 'Ahat_visual_avg_corr.mat']).corr_avg;
% corr_avg = cat(2, corr_avg{1,1}, corr_avg{2,1}, corr_avg{3,1}, corr_avg{4,1});
% 
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
% addpath(genpath('/local_raid1/01_software/spm12'));
% addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
% clc;
% 
% %load cii structure 
% subject_id='subject1';
% fmripath =['/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
% seg = 1;
% filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
% cii = ciftiopen([fmripath, filename],'wb_command');
% 
% %save back into dtseries for visualization - layidx
% cii.cdata = corr_avg;
% cii.diminfo{1, 1}.length = size(cii.cdata, 1);
% cii.diminfo{1, 2}.length = size(cii.cdata, 2);
% cii.diminfo{1, 1}.models(3:end) = [];
% save_file_name = fullfile(saveroot, 'visual_sub_lambda5_Ahat.dtseries.nii');
% ciftisavereset(cii, (save_file_name), 'wb_command');
% 
% %% make predicted BOLD - test concatenated segmentation 
% %test pcareduced feature
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/Ahat/';
% layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 
% 
% for lay = 1 : length(layername)
%     disp(['Layer: ', num2str(lay)]);
%     secpath = [saveroot,'PredNet_feature_maps_pcareduced_concatenated_test.h5'];  
%     Y_test = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
% 
%     filename=['W_lambda5_', layername{lay}(2:end) , '.mat'];
%     load([saveroot, filename], 'W');
% 
%     disp(['Y_test size: ',num2str(size(Y_test)),'; W size: ', num2str(size(W))]);
% 
%     pred_fmri = Y_test * W; 
% 
%     % save pred_fmri (feature x voxel)
%     file_name = ['predfMRI_', layername{lay}(2:end) , '.mat']
%     save([saveroot, file_name], 'pred_fmri', '-v7.3');
% end
% 
% %% Correlation between predicted BOLD and measured BOLD
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
% layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'};
% fmri_cat = fmri_cat';  % Assuming fmri_cat is initially Nv-by-Nt, transpose to Nt-by-Nv
% 
% for lay = 1:length(layername)
%     disp(['Layer: ', num2str(lay)]);
% 
%     filename = ['predfMRI_', layername{lay}(2:end), '.mat'];
%     load([saveroot, filename], 'pred_fmri');
%     disp(['pred fMRI size: ', num2str(size(pred_fmri))]);
% 
%     % Check dimensions match
%     if size(pred_fmri, 1) ~= size(fmri_cat, 1)
%         error('Mismatch in number of time points between predicted and actual fMRI data');
%     end
% 
%     R = zeros(1, size(pred_fmri, 2));  % Preallocate a vector for correlations
%     for v = 1:size(pred_fmri, 2)
%         R(v) = corr(pred_fmri(:, v), fmri_cat(:, v));  % Pearson correlation for each voxel
%     end
% 
%     file_name = ['corr_avg_rep_', layername{lay}(2:end), '.mat'];
%     save([saveroot, file_name], 'R', '-v7.3');
%     disp(['Saved correlation data for ', layername{lay}(2:end)]);
% end
% 
% %% Visual area correaltion between predicted BOLD and measured BOLD
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
% layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'};
% fmri_cat = fmri_cat';  % Assuming fmri_cat is initially Nv-by-Nt, transpose to Nt-by-Nv
% 
% label_32k= ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
% vertex=label_32k.cdata;
% 
% specific_voxels = find(vertex == 1 | vertex == 181 | ...
%     vertex == 3 | vertex == 183 | ...
%     vertex == 4 | vertex == 184 | vertex == 16 | vertex == 196 | ...
%     vertex == 7 | vertex == 187 | vertex == 19 | vertex == 199 | ...
%     vertex == 23 | vertex == 203 | vertex == 2 | vertex == 182 | ...
%     vertex == 17 | vertex == 197 | vertex == 20 | vertex == 200 | ...
%     vertex == 21 | vertex == 201 | vertex == 46 | vertex == 226 | ...
%     vertex == 142 | vertex == 322 | vertex == 145 | vertex == 325 | ...
%     vertex == 146 | vertex == 326 | vertex == 152 | vertex == 332 | ...
%     vertex == 155 | vertex == 335 | vertex == 31 | vertex == 211 | ...
%     vertex == 95 | vertex == 275 | vertex == 122 | vertex == 302 | vertex == 121 | vertex == 301 | ...
%     vertex == 119 | vertex == 299 | vertex == 126 | vertex == 306 | vertex == 127 | vertex == 307 | ...
%     vertex == 132 | vertex == 133 | vertex ==134 | vertex == 135 | vertex == 136 | ...
%     vertex == 312 | vertex == 313 | vertex == 314| vertex == 315 | vertex == 316| ...
%     vertex == 142 | vertex == 143 | vertex == 322 | vertex == 323 | ...
%     vertex == 139 | vertex == 140 | vertex == 141 | ...
%     vertex == 319 | vertex == 320 | vertex == 321 | ...
%     vertex == 150 | vertex == 151 | vertex == 330 | vertex == 331 | ...
%     vertex == 152 | vertex == 332 | vertex == 153 | vertex == 154 | ...
%     vertex == 156 | vertex == 333 | vertex == 334 | vertex == 336 | ...
%     vertex == 160 | vertex == 340 | vertex == 163 | vertex == 343| vertex == 230 | vertex == 50 |...
%     vertex == 229 | vertex == 49 | vertex == 339 | vertex == 159 | ...
%     vertex == 202 | vertex == 22 | vertex == 337 | vertex == 157| ...
%     vertex == 352 | vertex == 172 | vertex == 357 | vertex == 177| ...
%     vertex == 131 | vertex == 311 | vertex == 15 | vertex == 195 | ...
%     vertex == 300 | vertex == 120| vertex == 202 | vertex == 317 | vertex == 318 | vertex == 137 | vertex == 138| ...
%     vertex == 13 | vertex == 19 | vertex == 158 | vertex == 5 | vertex == 193 | vertex == 199 | vertex == 338 | vertex == 185 |...
%     vertex == 6 | vertex == 156 | vertex == 186 | vertex == 336 | ...
%     vertex==18 | vertex==198| vertex == 298 | vertex == 118);
% %%
% 
% for lay = 1:length(layername)
%     disp(['Layer: ', num2str(lay)]);
% 
%     filename = ['predfMRI_', layername{lay}(2:end), '.mat'];
%     load([saveroot, filename], 'pred_fmri');
%     disp(['pred fMRI size: ', num2str(size(pred_fmri))]);
% 
%     % Check dimensions match
%     if size(pred_fmri, 1) ~= size(fmri_cat, 1)
%         error('Mismatch in number of time points between predicted and actual fMRI data');
%     end
% 
%     R = -1 * ones(1, size(fmri_cat, 2));
% 
%     % Iterate only over specific voxels
%     for idx = 1:numel(specific_voxels)
%         voxel_index = specific_voxels(idx);
%         % Calculate correlation only for the specified voxels
%         R(voxel_index) = corr(pred_fmri(:, voxel_index), fmri_cat(:, voxel_index));
%     end
% 
%     file_name = ['corr_avg_visual_', layername{lay}(2:end), '.mat'];
%     save([saveroot, file_name], 'R', '-v7.3');
%     disp(['Saved correlation data for ', layername{lay}(2:end)]);
% end
%  %% jmlee : wb_view visualize 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
% 
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
% addpath(genpath('/local_raid1/01_software/spm12'));
% addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
% clc;
% 
% %load cii structure 
% subject_id='subject1';
% fmripath = '/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/subject1/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
% seg = 1;
% filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
% cii = ciftiopen([fmripath, filename],'wb_command');
% 
% % lay1=load([saveroot, 'corrAhat1.mat']).R;
% % lay2=load([saveroot, 'corrAhat2.mat']).R;
% % lay3=load([saveroot, 'corrAhat3.mat']).R;
% % lay4=load([saveroot, 'corrAhat4.mat']).R;
% 
% lay1=load([saveroot, 'corr_avg_visual_Ahat1.mat']).R;
% lay2=load([saveroot, 'corr_avg_visual_Ahat1.mat']).R;
% lay3=load([saveroot, 'corr_avg_visual_Ahat3.mat']).R;
% lay4=load([saveroot, 'corr_avg_visual_Ahat4.mat']).R;
% 
% test_corr = cat(1, lay1, lay2, lay3, lay4);
% 
% %save back into dtseries for visualization - layidx
% cii.cdata = test_corr';
% cii.diminfo{1, 1}.length = size(cii.cdata, 1);
% cii.diminfo{1, 2}.length = size(cii.cdata, 2);
% cii.diminfo{1, 1}.models(3:end) = [];
% save_file_name = fullfile(saveroot, [subject_id,'_visual_avg_test_corr.dtseries.nii']);
% ciftisavereset(cii, (save_file_name), 'wb_command');