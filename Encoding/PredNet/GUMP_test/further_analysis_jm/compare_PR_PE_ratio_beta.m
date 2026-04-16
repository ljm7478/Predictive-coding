%%
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

clc;
%% === Compute Ahat and E group beta ===
types = {'Ahat', 'E'};
num_subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
% layer_names = {'0','1','2','3','4','5','6'};
layer_names = {'0','1','2','3'};
layer_name = 'layer4';  % Folder name
model_type = 'original_prednet'; %original_prednet

dataroot_base = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/', model_type, '/multi_time_result/1second_25frame/'];

template_path = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
template_file = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
cii = ciftiopen([template_path, template_file],'wb_command');

group_beta_all = struct();

for t = 1:length(types) 
    type = types{t};
    dataroot = fullfile(dataroot_base, layer_name, type);
    layer_wise_group_beta = [];

    for l = 1:length(layer_names)
        layer_id = layer_names{l};
        current_layer = [type layer_id];  % ex) Ahat3 or E5
        disp(['Processing layer: ', current_layer]);

        for s = 1:length(num_subjects)
            sid = num_subjects{s};
            fname = fullfile(dataroot, sid, ['W_lambda5_' current_layer '.mat']);

            if exist(fname, 'file')
                load(fname, 'W');  % W: [feature x voxel]
                beta_mean = mean(W, 1);  % mean(W, 1) vs mean(abs(W), 1)

                if s == 1
                    all_beta = zeros(length(num_subjects), length(beta_mean));
                end
                all_beta(s, :) = beta_mean;
            else
                error('Missing file: %s', fname);
            end
        end

        % Group average beta per voxel
        group_beta_map = mean(all_beta, 1);  % [1 x voxel]
        layer_wise_group_beta(l, :) = group_beta_map;
    end

    % Store result: [num(layer) x voxel]
    group_beta_all.(type) = layer_wise_group_beta;
end

%% === Subtraction: Ahat - E ===
diff_beta = group_beta_all.Ahat - group_beta_all.E;

new_cii = cii;
new_cii.cdata = diff_beta';
new_cii.diminfo{1,1}.length = size(new_cii.cdata, 1);
new_cii.diminfo{1,2}.length = size(new_cii.cdata, 2);
new_cii.diminfo{1,1}.models(3:end) = [];

out_diff = fullfile(dataroot_base, layer_name, 'beta_diff_Ahat_minus_E_all_layers.dtseries.nii');
ciftisavereset(new_cii, out_diff, 'wb_command');
fprintf('[Saved Subtraction Map]: %s\n', out_diff);
