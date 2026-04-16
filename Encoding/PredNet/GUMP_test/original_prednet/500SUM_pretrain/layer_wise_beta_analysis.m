%% ROI Analysis: Layer-wise Beta Gradient for V1 vs DMN
% FIXED VERSION - handles different CIFTI label structures
%
% Run AFTER group beta/correlation maps are created.

clc; clear; close all;

%% --- Configuration ---
BASE_SAVE_ROOT = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/kitti_pretrain_result/four_train_four_test/layer4/';
GLASSER_LABEL = '/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii';
OUTPUT_DIR = fullfile(BASE_SAVE_ROOT, 'ROI_analysis');

if ~exist(OUTPUT_DIR, 'dir'), mkdir(OUTPUT_DIR); end

% Add CIFTI tools
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

% Analysis parameters
num_layers = 4;  % L1, L2, L3, L4
FEATURE_TYPES = {'Ahat', 'E'};

%% --- Load Glasser Parcellation ---
fprintf('Loading Glasser parcellation...\n');
tic;
label_cii = ciftiopen(GLASSER_LABEL, 'wb_command');
toc;

roi_labels = label_cii.cdata;  % (n_vertices x 1)
n_vertices = length(roi_labels);
fprintf('  Total vertices: %d\n', n_vertices);

%% --- Explore CIFTI structure to find label table ---
fprintf('\nExploring CIFTI structure...\n');
fprintf('  diminfo fields:\n');
disp(label_cii.diminfo);

% Try different possible structures
label_table = [];
roi_names_all = {};
roi_keys_all = [];

% Method 1: Try diminfo{1,1}.maps.table
try
    label_table = label_cii.diminfo{1,1}.maps.table;
    fprintf('  Found label table via diminfo{1,1}.maps.table\n');
catch
    fprintf('  diminfo{1,1}.maps.table not found\n');
end

% Method 2: Try diminfo{2}.maps.table
if isempty(label_table)
    try
        label_table = label_cii.diminfo{2}.maps.table;
        fprintf('  Found label table via diminfo{2}.maps.table\n');
    catch
        fprintf('  diminfo{2}.maps.table not found\n');
    end
end

% Method 3: Try metadata structure
if isempty(label_table)
    try
        if isfield(label_cii, 'metadata')
            disp(label_cii.metadata);
        end
    catch
    end
end

% Method 4: Look for labels in diminfo{1,2}
if isempty(label_table)
    try
        label_table = label_cii.diminfo{1,2}.maps.table;
        fprintf('  Found label table via diminfo{1,2}.maps.table\n');
    catch
        fprintf('  diminfo{1,2}.maps.table not found\n');
    end
end

% Method 5: Explore all diminfo fields
if isempty(label_table)
    fprintf('\n  Exploring diminfo structure in detail:\n');
    for i = 1:length(label_cii.diminfo)
        fprintf('    diminfo{%d}:\n', i);
        if isstruct(label_cii.diminfo{i})
            fn = fieldnames(label_cii.diminfo{i});
            for j = 1:length(fn)
                fprintf('      .%s\n', fn{j});
            end
            % Check for maps
            if isfield(label_cii.diminfo{i}, 'maps')
                fprintf('      -> maps found, exploring:\n');
                maps = label_cii.diminfo{i}.maps;
                if isstruct(maps)
                    maps_fn = fieldnames(maps);
                    for k = 1:length(maps_fn)
                        fprintf('         .maps.%s\n', maps_fn{k});
                    end
                    if isfield(maps, 'table')
                        label_table = maps.table;
                        fprintf('      -> Found label table!\n');
                    end
                end
            end
        elseif iscell(label_cii.diminfo{i})
            fprintf('      (cell array with %d elements)\n', length(label_cii.diminfo{i}));
        end
    end
end

%% --- Alternative: Get unique labels directly ---
fprintf('\nGetting unique ROI labels from data...\n');
unique_labels = unique(roi_labels);
unique_labels(unique_labels == 0) = [];  % Remove background
fprintf('  Found %d unique ROI labels\n', length(unique_labels));
fprintf('  Label range: %d to %d\n', min(unique_labels), max(unique_labels));

%% --- Define ROI groups by label number (Glasser standard) ---
% If we can't get names, we use standard Glasser numbering
% Glasser parcellation: 1-180 left hemisphere, 181-360 right hemisphere
% V1 = 1 (L), 181 (R)
% V2 = 4 (L), 184 (R)
% etc.

% Standard Glasser indices (may need adjustment based on your specific file)
% These are approximate - you may need to verify with your parcellation

fprintf('\nDefining ROIs by label number...\n');

% If we have the label table, extract names
if ~isempty(label_table) && isstruct(label_table)
    fprintf('  Extracting names from label table...\n');
    roi_names_all = cell(length(label_table), 1);
    roi_keys_all = zeros(length(label_table), 1);
    for i = 1:length(label_table)
        if isfield(label_table(i), 'name')
            roi_names_all{i} = label_table(i).name;
        else
            roi_names_all{i} = sprintf('ROI_%d', i);
        end
        if isfield(label_table(i), 'key')
            roi_keys_all(i) = label_table(i).key;
        else
            roi_keys_all(i) = i;
        end
    end
    fprintf('  Extracted %d ROI names\n', length(roi_names_all));
    
    % Print first 10 for verification
    fprintf('  First 10 ROIs:\n');
    for i = 1:min(10, length(roi_names_all))
        fprintf('    %d: %s (key=%d)\n', i, roi_names_all{i}, roi_keys_all(i));
    end
else
    fprintf('  No label table found - using numeric labels\n');
    roi_keys_all = unique_labels;
    roi_names_all = arrayfun(@(x) sprintf('ROI_%d', x), roi_keys_all, 'UniformOutput', false);
end

%% --- Define ROI Groups ---
% Function to find ROI indices by name (partial match)
find_roi_by_name = @(pattern) roi_keys_all(cellfun(@(x) ~isempty(regexpi(x, pattern)), roi_names_all));

% Try to find V1, V2, DMN regions by name
V1_keys = find_roi_by_name('^[LR]_V1_');
if isempty(V1_keys)
    V1_keys = find_roi_by_name('V1');
end
if isempty(V1_keys)
    % Fallback to standard Glasser V1 indices
    V1_keys = [1, 181];  % Left and right V1
end

V2_keys = find_roi_by_name('^[LR]_V2_');
if isempty(V2_keys)
    V2_keys = find_roi_by_name('V2');
end
if isempty(V2_keys)
    V2_keys = [4, 184];
end

V3_keys = find_roi_by_name('^[LR]_V3_');
if isempty(V3_keys)
    V3_keys = find_roi_by_name('V3');
end
if isempty(V3_keys)
    V3_keys = [5, 185];
end

% DMN regions - try by name first
mPFC_keys = find_roi_by_name('10[rvd]|9[map]|32|24');
PCC_keys = find_roi_by_name('POS|23|31|RSC|PCV|7m');
Angular_keys = find_roi_by_name('PG[isp]');
LatTemp_keys = find_roi_by_name('TE[12][ap]|TG[dv]');

% Combine for DMN
DMN_keys = unique([mPFC_keys(:); PCC_keys(:); Angular_keys(:); LatTemp_keys(:)]);

% If no names found, use standard Glasser indices for DMN
if isempty(DMN_keys)
    fprintf('  Using standard Glasser indices for DMN...\n');
    % Approximate DMN indices in Glasser (you may need to adjust)
    % mPFC: ~10, 11, 12 and right hemisphere equivalents
    % PCC: ~23, 31 regions
    DMN_keys = [10, 11, 12, 23, 26, 31, 72, 73, ...  % Left
                190, 191, 192, 203, 206, 211, 252, 253];  % Right
end

% Higher visual
HigherVis_keys = find_roi_by_name('V4|MT|MST|FST|LO|V6|V7|V8');
if isempty(HigherVis_keys)
    HigherVis_keys = [6, 7, 8, 23, 24, 186, 187, 188, 203, 204];
end

% Frontoparietal
FP_keys = find_roi_by_name('8[CAB]|46|LIP|VIP|MIP|AIP|7[AP]');
if isempty(FP_keys)
    FP_keys = [40:50, 220:230];  % Approximate
end

fprintf('\nROI definitions:\n');
fprintf('  V1: %d vertices\n', sum(ismember(roi_labels, V1_keys)));
fprintf('  V2: %d vertices\n', sum(ismember(roi_labels, V2_keys)));
fprintf('  V3: %d vertices\n', sum(ismember(roi_labels, V3_keys)));
fprintf('  DMN: %d vertices (from %d regions)\n', sum(ismember(roi_labels, DMN_keys)), length(DMN_keys));
fprintf('  Higher Visual: %d vertices\n', sum(ismember(roi_labels, HigherVis_keys)));
fprintf('  Frontoparietal: %d vertices\n', sum(ismember(roi_labels, FP_keys)));

%% --- Create ROI Masks ---
V1_mask = find(ismember(roi_labels, V1_keys));
V2_mask = find(ismember(roi_labels, V2_keys));
EarlyVis_mask = find(ismember(roi_labels, [V1_keys(:); V2_keys(:); V3_keys(:)]));
DMN_mask = find(ismember(roi_labels, DMN_keys));
HigherVis_mask = find(ismember(roi_labels, HigherVis_keys));
FP_mask = find(ismember(roi_labels, FP_keys));

roi_names = {'V1', 'Early Visual', 'DMN', 'Higher Visual', 'Frontoparietal'};
roi_masks = {V1_mask, EarlyVis_mask, DMN_mask, HigherVis_mask, FP_mask};
n_rois = length(roi_names);

%% --- Load Beta Maps and Extract ROI Values ---
fprintf('\n=== Extracting ROI values from beta maps ===\n');

results = struct();

for f = 1:length(FEATURE_TYPES)
    feature_type = FEATURE_TYPES{f};
    fprintf('\nProcessing %s...\n', feature_type);
    
    beta_file = fullfile(BASE_SAVE_ROOT, feature_type, ['group_beta_map_', feature_type, '.mat']);
    corr_file = fullfile(BASE_SAVE_ROOT, feature_type, ['group_average_correlations_', feature_type, '.mat']);
    
    % Load and process beta map
    if exist(beta_file, 'file')
        loaded = load(beta_file);
        beta_map = loaded.layer_wise_group_beta;
        fprintf('  Beta map: [%d x %d]\n', size(beta_map, 1), size(beta_map, 2));
        
        roi_beta = nan(num_layers, n_rois);
        roi_beta_sem = nan(num_layers, n_rois);
        
        for lay = 1:min(num_layers, size(beta_map, 1))
            for r = 1:n_rois
                mask = roi_masks{r};
                if ~isempty(mask) && max(mask) <= size(beta_map, 2)
                    values = beta_map(lay, mask);
                    values = values(isfinite(values));
                    if ~isempty(values)
                        roi_beta(lay, r) = mean(values);
                        roi_beta_sem(lay, r) = std(values) / sqrt(length(values));
                    end
                end
            end
        end
        
        results.(feature_type).beta = roi_beta;
        results.(feature_type).beta_sem = roi_beta_sem;
        
        fprintf('  Beta values extracted\n');
    else
        fprintf('  Beta file not found: %s\n', beta_file);
    end
    
    % Load and process correlation map
    if exist(corr_file, 'file')
        loaded = load(corr_file);
        corr_map = loaded.group_avg_corr_matrix;
        fprintf('  Correlation map: [%d x %d]\n', size(corr_map, 1), size(corr_map, 2));
        
        roi_corr = nan(num_layers, n_rois);
        roi_corr_sem = nan(num_layers, n_rois);
        
        for lay = 1:min(num_layers, size(corr_map, 1))
            for r = 1:n_rois
                mask = roi_masks{r};
                if ~isempty(mask) && max(mask) <= size(corr_map, 2)
                    values = corr_map(lay, mask);
                    values = values(isfinite(values));
                    if ~isempty(values)
                        roi_corr(lay, r) = mean(values);
                        roi_corr_sem(lay, r) = std(values) / sqrt(length(values));
                    end
                end
            end
        end
        
        results.(feature_type).corr = roi_corr;
        results.(feature_type).corr_sem = roi_corr_sem;
        
        fprintf('  Correlation values extracted\n');
    else
        fprintf('  Correlation file not found: %s\n', corr_file);
    end
end

%% --- Save Results ---
save(fullfile(OUTPUT_DIR, 'roi_layer_results.mat'), 'results', 'roi_names', 'num_layers', ...
     'V1_keys', 'DMN_keys', 'roi_names_all', 'roi_keys_all');
fprintf('\nResults saved to: %s\n', fullfile(OUTPUT_DIR, 'roi_layer_results.mat'));

%% --- Print Results Table ---
fprintf('\n=== RESULTS TABLE ===\n');

if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    fprintf('\nPrediction  Beta Weights:\n');
    fprintf('%-10s', 'Layer');
    for r = 1:n_rois
        fprintf('%-18s', roi_names{r});
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 10 + 18*n_rois));
    
    for lay = 1:num_layers
        fprintf('%-10s', sprintf('L%d', lay));
        for r = 1:n_rois
            fprintf('%-18.6f', results.Ahat.beta(lay, r));
        end
        fprintf('\n');
    end
end

if isfield(results, 'E') && isfield(results.E, 'beta')
    fprintf('\nError (E) Beta Weights:\n');
    fprintf('%-10s', 'Layer');
    for r = 1:n_rois
        fprintf('%-18s', roi_names{r});
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 10 + 18*n_rois));
    
    for lay = 1:num_layers
        fprintf('%-10s', sprintf('L%d', lay));
        for r = 1:n_rois
            fprintf('%-18.6f', results.E.beta(lay, r));
        end
        fprintf('\n');
    end
end

%% --- Create Figures ---
fprintf('\n=== Creating Figures ===\n');

colors = struct();
colors.V1 = [0.8, 0.2, 0.2];
colors.EarlyVisual = [0.9, 0.4, 0.4];
colors.DMN = [0.2, 0.4, 0.8];
colors.HigherVisual = [0.9, 0.6, 0.2];
colors.Frontoparietal = [0.4, 0.7, 0.4];
color_array = [colors.V1; colors.EarlyVisual; colors.DMN; colors.HigherVisual; colors.Frontoparietal];

layer_labels = arrayfun(@(x) sprintf('L%d', x), 1:num_layers, 'UniformOutput', false);

%% Figure 1: Main Finding - V1 vs DMN Gradient
figure('Position', [100, 100, 500, 400], 'Color', 'w');
hold on;

if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    errorbar(1:num_layers, results.Ahat.beta(:, 1), results.Ahat.beta_sem(:, 1), ...
        '-o', 'Color', colors.V1, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.V1);
    errorbar(1:num_layers, results.Ahat.beta(:, 3), results.Ahat.beta_sem(:, 3), ...
        '-s', 'Color', colors.DMN, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.DMN);
end

yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('Prediction : V1 vs DMN Across Layers', 'FontSize', 16, 'FontWeight', 'bold');
legend({'V1', 'DMN'}, 'Location', 'best', 'FontSize', 12);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

saveas(gcf, fullfile(OUTPUT_DIR, 'figure1_Prediction_V1_vs_DMN.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure1_Prediction_V1_vs_DMN.fig'));
fprintf('  Saved: figure1_Prediction_V1_vs_DMN\n');

%% Figure 1b: Error - V1 vs DMN Across Layers
figure('Position', [100, 100, 500, 400], 'Color', 'w');
hold on;

if isfield(results, 'E') && isfield(results.E, 'beta')
    errorbar(1:num_layers, results.E.beta(:, 1), results.E.beta_sem(:, 1), ...
        '-o', 'Color', colors.V1, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.V1);
    errorbar(1:num_layers, results.E.beta(:, 3), results.E.beta_sem(:, 3), ...
        '-s', 'Color', colors.DMN, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.DMN);
end

yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('Error (E): V1 vs DMN Across Layers', 'FontSize', 16, 'FontWeight', 'bold');
legend({'V1', 'DMN'}, 'Location', 'best', 'FontSize', 12);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

saveas(gcf, fullfile(OUTPUT_DIR, 'figure1b_Error_V1_vs_DMN.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure1b_Error_V1_vs_DMN.fig'));
fprintf('  Saved: figure1b_Error_V1_vs_DMN\n');

%% Figure 1c: Prediction vs Error - V1 vs DMN (2 panels side by side)
figure('Position', [100, 100, 1000, 400], 'Color', 'w');

% Panel A: Prediction
subplot(1, 2, 1);
hold on;
if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    errorbar(1:num_layers, results.Ahat.beta(:, 1), results.Ahat.beta_sem(:, 1), ...
        '-o', 'Color', colors.V1, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.V1);
    errorbar(1:num_layers, results.Ahat.beta(:, 3), results.Ahat.beta_sem(:, 3), ...
        '-s', 'Color', colors.DMN, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.DMN);
end
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('Prediction : V1 vs DMN', 'FontSize', 14, 'FontWeight', 'bold');
legend({'V1', 'DMN'}, 'Location', 'best', 'FontSize', 11);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

% Panel B: Error
subplot(1, 2, 2);
hold on;
if isfield(results, 'E') && isfield(results.E, 'beta')
    errorbar(1:num_layers, results.E.beta(:, 1), results.E.beta_sem(:, 1), ...
        '-o', 'Color', colors.V1, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.V1);
    errorbar(1:num_layers, results.E.beta(:, 3), results.E.beta_sem(:, 3), ...
        '-s', 'Color', colors.DMN, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors.DMN);
end
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('Error (E): V1 vs DMN', 'FontSize', 14, 'FontWeight', 'bold');
legend({'V1', 'DMN'}, 'Location', 'best', 'FontSize', 11);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

sgtitle('V1 vs DMN: Prediction and Error Signals', 'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, fullfile(OUTPUT_DIR, 'figure1c_Prediction_Error_V1_vs_DMN_sidebyside.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure1c_Prediction_Error_V1_vs_DMN_sidebyside.fig'));
fprintf('  Saved: figure1c_Prediction_Error_V1_vs_DMN_sidebyside\n');

%% Figure 2: All ROIs
figure('Position', [100, 100, 600, 450], 'Color', 'w');
hold on;

if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    for r = 1:n_rois
        errorbar(1:num_layers, results.Ahat.beta(:, r), results.Ahat.beta_sem(:, r), ...
            '-o', 'Color', color_array(r, :), 'LineWidth', 2, 'MarkerSize', 8, ...
            'MarkerFaceColor', color_array(r, :));
    end
end

yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('Prediction  Beta Across Layers', 'FontSize', 16, 'FontWeight', 'bold');
legend(roi_names, 'Location', 'best', 'FontSize', 11);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

saveas(gcf, fullfile(OUTPUT_DIR, 'figure2_all_ROIs_gradient.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure2_all_ROIs_gradient.fig'));
fprintf('  Saved: figure2_all_ROIs_gradient\n');

%% Figure 3: Prediction vs Error in V1 AND DMN (2 panels)
figure('Position', [100, 100, 900, 400], 'Color', 'w');

% Define colors: Prediction=Blue, Error=Red
color_pred = [0.2, 0.4, 0.8];  % Blue
color_err = [0.8, 0.2, 0.2];   % Red

% Panel A: V1
subplot(1, 2, 1);
hold on;
if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    errorbar(1:num_layers, results.Ahat.beta(:, 1), results.Ahat.beta_sem(:, 1), ...
        '-o', 'Color', color_pred, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', color_pred);
end
if isfield(results, 'E') && isfield(results.E, 'beta')
    errorbar(1:num_layers, results.E.beta(:, 1), results.E.beta_sem(:, 1), ...
        '-s', 'Color', color_err, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', color_err);
end
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('V1: Prediction vs Error', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Prediction ', 'Error (E)'}, 'Location', 'best', 'FontSize', 11);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

% Panel B: DMN
subplot(1, 2, 2);
hold on;
if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    errorbar(1:num_layers, results.Ahat.beta(:, 3), results.Ahat.beta_sem(:, 3), ...
        '-o', 'Color', color_pred, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', color_pred);
end
if isfield(results, 'E') && isfield(results.E, 'beta')
    errorbar(1:num_layers, results.E.beta(:, 3), results.E.beta_sem(:, 3), ...
        '-s', 'Color', color_err, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', color_err);
end
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('DMN: Prediction vs Error', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Prediction ', 'Error (E)'}, 'Location', 'best', 'FontSize', 11);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

sgtitle('Prediction vs Error: V1 and DMN', 'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, fullfile(OUTPUT_DIR, 'figure3_V1_DMN_prediction_vs_error.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure3_V1_DMN_prediction_vs_error.fig'));
fprintf('  Saved: figure3_V1_DMN_prediction_vs_error\n');

%% Figure 3b: All four lines in one plot (V1=solid, DMN=dashed)
figure('Position', [100, 100, 600, 450], 'Color', 'w');
hold on;

% V1 Prediction (Blue, solid)
if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    errorbar(1:num_layers, results.Ahat.beta(:, 1), results.Ahat.beta_sem(:, 1), ...
        '-o', 'Color', color_pred, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', color_pred);
end
% V1 Error (Red, solid)
if isfield(results, 'E') && isfield(results.E, 'beta')
    errorbar(1:num_layers, results.E.beta(:, 1), results.E.beta_sem(:, 1), ...
        '-s', 'Color', color_err, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', color_err);
end
% DMN Prediction (Blue, dashed)
if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    errorbar(1:num_layers, results.Ahat.beta(:, 3), results.Ahat.beta_sem(:, 3), ...
        '--o', 'Color', color_pred, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
end
% DMN Error (Red, dashed)
if isfield(results, 'E') && isfield(results.E, 'beta')
    errorbar(1:num_layers, results.E.beta(:, 3), results.E.beta_sem(:, 3), ...
        '--s', 'Color', color_err, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
end

yline(0, 'k--', 'LineWidth', 1.5);
xlabel('PredNet Layer', 'FontSize', 14);
ylabel('Beta Weight (PC1)', 'FontSize', 14);
title('V1 vs DMN: Prediction and Error Signals', 'FontSize', 16, 'FontWeight', 'bold');
legend({'V1 Prediction', 'V1 Error', 'DMN Prediction', 'DMN Error'}, ...
    'Location', 'best', 'FontSize', 11);
set(gca, 'XTick', 1:num_layers, 'XTickLabel', layer_labels, 'FontSize', 12);
xlim([0.5, num_layers + 0.5]);
grid on; box on;
hold off;

saveas(gcf, fullfile(OUTPUT_DIR, 'figure3b_V1_DMN_all_signals.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure3b_V1_DMN_all_signals.fig'));
fprintf('  Saved: figure3b_V1_DMN_all_signals\n');

%% Figure 4: Bar plot L1 vs L4/L5
figure('Position', [100, 100, 450, 400], 'Color', 'w');

if isfield(results, 'Ahat') && isfield(results.Ahat, 'beta')
    bar_data = [results.Ahat.beta(num_layers, 1), results.Ahat.beta(num_layers, 3);
                results.Ahat.beta(1, 1), results.Ahat.beta(1, 3)];
    bar_sem = [results.Ahat.beta_sem(num_layers, 1), results.Ahat.beta_sem(num_layers, 3);
               results.Ahat.beta_sem(1, 1), results.Ahat.beta_sem(1, 3)];
    
    b = bar(bar_data, 'grouped');
    b(1).FaceColor = colors.V1;
    b(2).FaceColor = colors.DMN;
    
    hold on;
    ngroups = size(bar_data, 1);
    nbars = size(bar_data, 2);
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x, bar_data(:,i), bar_sem(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1.5);
    end
    yline(0, 'k--', 'LineWidth', 1);
    hold off;
    
    set(gca, 'XTickLabel', {sprintf('L%d (High)', num_layers), 'L1 (Low)'}, 'FontSize', 12);
    ylabel('Beta Weight (PC1)', 'FontSize', 14);
    title('Prediction: High vs Low Layer', 'FontSize', 16, 'FontWeight', 'bold');
    legend({'V1', 'DMN'}, 'Location', 'best', 'FontSize', 12);
    grid on; box on;
end

saveas(gcf, fullfile(OUTPUT_DIR, 'figure4_barplot_L1_vs_Ltop.png'));
saveas(gcf, fullfile(OUTPUT_DIR, 'figure4_barplot_L1_vs_Ltop.fig'));
fprintf('  Saved: figure4_barplot_L1_vs_Ltop\n');

fprintf('\n=== Analysis Complete ===\n');
fprintf('All figures saved to: %s\n', OUTPUT_DIR);