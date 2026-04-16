%%
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

clc;
%% === Compute Ahat and E group corr ===
types = {'Ahat', 'E'};
num_subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};

model_type = 'modified_prednet'; %original_prednet
layer_type = 'layer7';

dataroot_base = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/', model_type, '/multi_time_result/2second_50frame/'];

template_path = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
template_file = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
cii = ciftiopen([template_path, template_file],'wb_command');


% -------- Main loop --------
all_diffs = [];   % [nSubjects x num_layers x num_voxels]
n_loaded  = 0;

for s = 1:length(num_subjects)
    sid = num_subjects{s};

    % Prediction (Ahat)
    fname_A = fullfile(dataroot_base, layer_type, 'Ahat', sid, [sid, '_avgcorr_lambda5.mat']);
    if ~exist(fname_A, 'file'), error('Missing: %s', fname_A); end
    S_A = load(fname_A, 'concat_corr');
    Ahat_corr = S_A.concat_corr;   % [num_layers x num_voxels]

    % Prediction error (E)
    fname_E = fullfile(dataroot_base, layer_type, 'E', sid, [sid, '_avgcorr_lambda5.mat']);
    if ~exist(fname_E, 'file'), error('Missing: %s', fname_E); end
    S_E = load(fname_E, 'concat_corr');
    E_corr = S_E.concat_corr;      % [num_layers x num_voxels]

    % Shape check
    if ~isequal(size(Ahat_corr), size(E_corr))
        error('Shape mismatch for %s: Ahat %s vs E %s', ...
              sid, mat2str(size(Ahat_corr)), mat2str(size(E_corr)));
    end

    % Difference: Ahat - E
    diff_corr = Ahat_corr - E_corr;  % abs(Ahat_corr - E_corr) or Ahat_corr - E_corr

    % Accumulate
    if isempty(all_diffs)
        all_diffs = zeros(length(num_subjects), size(diff_corr,1), size(diff_corr,2));
    end
    all_diffs(s,:,:) = diff_corr;
    n_loaded = n_loaded + 1;
end

% -------- Group-level maps --------
% Average across subjects [num_layers x num_voxels]
group_diff = squeeze(mean(all_diffs,1));

%% === Subtraction: Ahat - E ===
new_cii = cii;
new_cii.cdata = group_diff';
new_cii.diminfo{1,1}.length = size(new_cii.cdata, 1);
new_cii.diminfo{1,2}.length = size(new_cii.cdata, 2);
new_cii.diminfo{1,1}.models(3:end) = [];

out_diff = fullfile(dataroot_base, layer_type, 'corr_diff_Ahat_minus_E_all_layers.dtseries.nii');
ciftisavereset(new_cii, out_diff, 'wb_command');
fprintf('[Saved Subtraction Map]: %s\n', out_diff);
