%% === Load per-layer ROI lists and apply masks to group beta maps ===
clc;

subjects  = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
% Layers to include (order matters for PCA)
% layername = {'/C0'; '/C1'; '/C2'; '/C3'};
% layername = {'/C0'; '/C1'; '/C2'; '/C3'; '/C4'; '/C5'; '/C6'};

layername = {'/E0'; '/E1'; '/E2'; '/E3'};  
% layername = {'/E0'; '/E1'; '/E2'; '/E3'; '/E4'; '/E5'; '/E6'};

% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6'}; 

n_lyr = numel(layername);

% Resolve layer set (for paths)
if n_lyr == 4
    layer_tag = 'layer4';
elseif n_lyr == 7  
    layer_tag = 'layer7';
else
    error('layername must have 4 or 7 entries.');
end

prednet_ver = 'modified_prednet'; % or 'modified_prednet'

% Derive "type" (e.g., 'Ahat' from '/Ahat0', 'E' from '/E3')
tok = regexp(layername{1}, '^/([A-Za-z]+)\d+$', 'tokens','once');
assert(~isempty(tok), 'layername entries must look like ''/Ahat0'', ''/E3'', etc.');
type = tok{1};  % e.g., 'Ahat' or 'E'

% Base paths
base_dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/' ...
                 'Titanic_trained/GUMP_test/encoding_analysis/', prednet_ver, '/result/' ...
                 layer_tag, '/', type, '/'];
out_dir = fullfile(base_dataroot, 'group_stats');
assert(exist(out_dir,'dir')==7, 'Missing out_dir: %s', out_dir);

%% 1) Load ROI lists
roi_file = fullfile(out_dir,'Group_ANY_FDR_keepROIs.mat');
assert(exist(roi_file,'file')==2, 'Missing ROI list: %s', roi_file);
Sroi = load(roi_file);
assert(isfield(Sroi,'all_ROIs'), 'Expected variable ''all_ROIs'' in %s', roi_file);
rois_per_layer = Sroi.all_ROIs;
assert(iscell(rois_per_layer) && numel(rois_per_layer)==n_lyr, ...
    'ROI list must be a cell array of length n_lyr');

% %% 2) Load group beta maps (MATs saved earlier)
% % Try variable names robustly
% mat_maxabs = fullfile(base_dataroot, sprintf('group_beta_%s_all_layers_PCA_maxabs.mat', type));
% mat_zscore = fullfile(base_dataroot, sprintf('group_beta_%s_all_layers_PCA_zscore.mat', type));
% assert(exist(mat_maxabs,'file')==2 && exist(mat_zscore,'file')==2, ...
%     'Missing beta .mat files: %s or %s', mat_maxabs, mat_zscore);
% 
% Smax = load(mat_maxabs);
% Szn  = load(mat_zscore);
% 
% if isfield(Smax,'layer_wise_group_beta_maxabs')
%     beta_maxabs = Smax.layer_wise_group_beta_maxabs;   % [n_lyr x n_vox]
% elseif isfield(Smax,'group_beta_map_maxabs')
%     beta_maxabs = Smax.group_beta_map_maxabs;
% else
%     error('No expected variable in %s', mat_maxabs);
% end
% 
% if isfield(Szn,'layer_wise_group_beta_zscore')
%     beta_zscore = Szn.layer_wise_group_beta_zscore;    % [n_lyr x n_vox]
% elseif isfield(Szn,'group_beta_map_zscore')
%     beta_zscore = Szn.group_beta_map_zscore;
% else
%     error('No expected variable in %s', mat_zscore);
% end
% 
% assert(size(beta_maxabs,1)==n_lyr && size(beta_zscore,1)==n_lyr, 'Layer count mismatch in beta maps.');
% n_vox = size(beta_maxabs,2);

%% 3) Load Glasser labels (vertex-wise)
dlabel_path = '/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii';
glasser = ciftiopen(dlabel_path, 'wb_command');
labels = glasser.cdata(:);                % [n_vox x 1] integer labels
assert(numel(labels)==n_vox, ...
    'Glasser label vertex count (%d) != beta map vertex count (%d)', numel(labels), n_vox);

%% 4) Build per-layer masks and apply
% % ----- compute union only once -----
% roi_union = unique([rois_per_layer{:}]);              % 1 x K (all ROI ids across layers)
% if isempty(roi_union)
%     warning('ROI union is empty; union mask will be all false.');
% end
% m_union = ismember(labels, roi_union(:)'); 
% 
% % Preallocate
% union_masked_maxabs = zeros(size(beta_maxabs));       % [n_lyr x n_vox]
% union_masked_zscore = zeros(size(beta_zscore));       % [n_lyr x n_vox]
% union_bin_masks     = repmat(m_union', n_lyr, 1);     % same union mask row for each layer
% 
% masked_maxabs = zeros(size(beta_maxabs));             % [n_lyr x n_vox]
% masked_zscore = zeros(size(beta_zscore));             % [n_lyr x n_vox]
% bin_masks     = false(n_lyr, n_vox);
% 
% for li = 1:n_lyr
%     % ----- union (same mask for every layer) -----
%     union_masked_maxabs(li, :) = beta_maxabs(li, :) .* m_union';
%     union_masked_zscore(li, :) = beta_zscore(li, :) .* m_union';
% 
%     % ----- per-layer ROI mask -----
%     roi_ids = rois_per_layer{li};
%     if isempty(roi_ids)
%         warning('Layer %d: ROI list is empty; mask all false.', li);
%         m_layer = false(n_vox,1);
%     else
%         m_layer = ismember(labels, roi_ids(:)');      % [n_vox x 1]
%     end
%     bin_masks(li, :)     = m_layer';
%     masked_maxabs(li, :) = beta_maxabs(li, :) .* m_layer';
%     masked_zscore(li, :) = beta_zscore(li, :) .* m_layer';
% end
% 
% %% 5) Save masked dtseries (voxels x layers)
% % Load a template dtseries for geometry
% template_path = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
% template_file = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
% cii = ciftiopen([template_path, template_file], 'wb_command');
% 
% out = cii;
% out.diminfo{1,1}.models(3:end) = [];
% 
% % ----- union masked beta -----
% out.cdata = union_masked_maxabs';  % [n_vox x n_lyr]
% out.diminfo{1,1}.length = size(out.cdata,1);
% out.diminfo{1,2}.length = size(out.cdata,2);
% outfile = fullfile(base_dataroot, sprintf('Union_group_beta_%s_all_layers_PCA_maxabs_MASKED.dtseries.nii', type));
% ciftisavereset(out, outfile, 'wb_command'); fprintf('Saved: %s\n', outfile);
% 
% out.cdata = union_masked_zscore';
% out.diminfo{1,1}.length = size(out.cdata,1);
% out.diminfo{1,2}.length = size(out.cdata,2);
% outfile = fullfile(base_dataroot, sprintf('Union_group_beta_%s_all_layers_PCA_zscore_MASKED.dtseries.nii', type));
% ciftisavereset(out, outfile, 'wb_command'); fprintf('Saved: %s\n', outfile);
% 
% % ----- per-layer masked beta -----
% out.cdata = masked_maxabs';
% out.diminfo{1,1}.length = size(out.cdata,1);
% out.diminfo{1,2}.length = size(out.cdata,2);
% outfile = fullfile(base_dataroot, sprintf('group_beta_%s_all_layers_PCA_maxabs_MASKED.dtseries.nii', type));
% ciftisavereset(out, outfile, 'wb_command'); fprintf('Saved: %s\n', outfile);
% 
% out.cdata = masked_zscore';
% out.diminfo{1,1}.length = size(out.cdata,1);
% out.diminfo{1,2}.length = size(out.cdata,2);
% outfile = fullfile(base_dataroot, sprintf('group_beta_%s_all_layers_PCA_zscore_MASKED.dtseries.nii', type));
% ciftisavereset(out, outfile, 'wb_command'); fprintf('Saved: %s\n', outfile);
% 
% % % optional: per-layer binary masks
% % out.cdata = double(bin_masks)';
% % out.diminfo{1,1}.length = size(out.cdata,1);
% % out.diminfo{1,2}.length = size(out.cdata,2);
% % outfile = fullfile(base_dataroot, sprintf('group_beta_%s_ROImask_perLayer.dtseries.nii', type));
% % ciftisavereset(out, outfile, 'wb_command');
% % fprintf('Saved per-layer ROI masks: %s\n', outfile);
% 

%% 6) Load encoding performance -group 
Senc = load(fullfile(base_dataroot, [type '_avg_lambda5.mat']));
if isfield(Senc,'group_avg_corr')
    group_avg_corr = Senc.group_avg_corr;
elseif isfield(Senc,'concat_corr')
    group_avg_corr = Senc.concat_corr;   % fallback name
else
    error('Expected group_avg_corr (or concat_corr) in %s_avg_lambda5.mat', type);
end
assert(isequal(size(group_avg_corr,1), n_lyr) && isequal(size(group_avg_corr,2), n_vox), ...
       'group_avg_corr has size %dx%d; expected %dx%d', ...
       size(group_avg_corr,1), size(group_avg_corr,2), n_lyr, n_vox);

% Preallocate
union_masked_group_avg_corr = zeros(n_lyr, n_vox);
masked_group_avg_corr       = zeros(n_lyr, n_vox);

% ----- apply union mask (same for all layers) -----
for li = 1:n_lyr
    union_masked_group_avg_corr(li, :) = group_avg_corr(li, :) .* m_union';
end

% ----- apply per-layer masks -----
for li = 1:n_lyr
    roi_ids = rois_per_layer{li};
    if isempty(roi_ids)
        m_layer = false(n_vox,1);
    else
        m_layer = ismember(labels, roi_ids(:)');      % [n_vox x 1]
    end
    masked_group_avg_corr(li, :) = group_avg_corr(li, :) .* m_layer';
end

%% 7) Save masked encoding performance (voxels x layers) 
out = cii; out.diminfo{1,1}.models(3:end) = [];

% union-masked encoding performance
out.cdata = union_masked_group_avg_corr';
out.diminfo{1,1}.length = size(out.cdata,1);
out.diminfo{1,2}.length = size(out.cdata,2);
outfile = fullfile(base_dataroot, sprintf('Union_group_%s_MASKED.dtseries.nii', type));
ciftisavereset(out, outfile, 'wb_command'); fprintf('Saved: %s\n', outfile);

% per-layer masked encoding performance
out.cdata = masked_group_avg_corr';
out.diminfo{1,1}.length = size(out.cdata,1);
out.diminfo{1,2}.length = size(out.cdata,2);
outfile = fullfile(base_dataroot, sprintf('group_%s_MASKED.dtseries.nii', type));
ciftisavereset(out, outfile, 'wb_command'); fprintf('Saved: %s\n', outfile);

