%% PCA Threshold Analysis: All Layers
% Find 90%, 95%, 99% variance threshold for all layers

clc; clear; close all;

%% --- Configuration ---
FEATURE_DATA_ROOT = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4';
SAVE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4/Results/Q1/PCA_analysis';

if ~exist(SAVE_DIR, 'dir'), mkdir(SAVE_DIR); end

SEGMENT = 'seg0';
NUM_LAYERS = 4;  % Layer 1-4 (Ahat0-3, E0-3)

%% --- Storage for results ---
results = struct();
thresholds = [0.90, 0.95, 0.99];

fprintf('%s\n', repmat('=', 1, 70));
fprintf('PCA Threshold Analysis: All Layers\n');
fprintf('%s\n', repmat('=', 1, 70));

%% --- Analyze Each Layer ---
for layer_idx = 1:NUM_LAYERS
    fprintf('\n%s\n', repmat('-', 1, 70));
    fprintf('LAYER %d (Ahat%d / E%d)\n', layer_idx, layer_idx-1, layer_idx-1);
    fprintf('%s\n', repmat('-', 1, 70));
    
    % --- Load Ahat ---
    ahat_file = fullfile(FEATURE_DATA_ROOT, [SEGMENT, '_Ahat.h5']);
    ahat_layer_name = ['/Ahat', num2str(layer_idx-1)];
    fprintf('Loading %s...\n', ahat_layer_name);
    ahat_4d = h5read(ahat_file, ahat_layer_name);
    fprintf('  Shape: [%s]\n', num2str(size(ahat_4d)));
    
    % --- Load E ---
    e_file = fullfile(FEATURE_DATA_ROOT, [SEGMENT, '_E.h5']);
    e_layer_name = ['/E', num2str(layer_idx-1)];
    fprintf('Loading %s...\n', e_layer_name);
    e_4d = h5read(e_file, e_layer_name);
    fprintf('  Shape: [%s]\n', num2str(size(e_4d)));
    
    % --- Reshape ---
    dims_ahat = size(ahat_4d);
    dims_e = size(e_4d);
    
    ahat_2d = single(reshape(ahat_4d, prod(dims_ahat(1:end-1)), dims_ahat(end)));
    e_2d = single(reshape(e_4d, prod(dims_e(1:end-1)), dims_e(end)));
    
    fprintf('  Ahat: [%d features x %d timepoints]\n', size(ahat_2d, 1), size(ahat_2d, 2));
    fprintf('  E: [%d features x %d timepoints]\n', size(e_2d, 1), size(e_2d, 2));
    
    clear ahat_4d e_4d;
    
    % --- Standardize ---
    ahat_mean = mean(ahat_2d, 2);
    ahat_std = std(ahat_2d, 0, 2);
    ahat_std(ahat_std == 0) = 1;
    ahat_std_2d = bsxfun(@minus, ahat_2d, ahat_mean);
    ahat_std_2d = bsxfun(@rdivide, ahat_std_2d, ahat_std);
    
    e_mean = mean(e_2d, 2);
    e_std = std(e_2d, 0, 2);
    e_std(e_std == 0) = 1;
    e_std_2d = bsxfun(@minus, e_2d, e_mean);
    e_std_2d = bsxfun(@rdivide, e_std_2d, e_std);
    
    clear ahat_2d e_2d ahat_mean ahat_std e_mean e_std;
    
    % --- Compute Eigenvalues ---
    fprintf('Computing eigenvalues...\n');
    
    % Ahat
    tic;
    [n_feat, n_time] = size(ahat_std_2d);
    if n_feat > n_time
        R = ahat_std_2d' * ahat_std_2d / n_feat;
    else
        R = ahat_std_2d * ahat_std_2d' / n_time;
    end
    ahat_eig = sort(max(eig(R), 0), 'descend');
    ahat_cumvar = cumsum(ahat_eig) / sum(ahat_eig);
    clear R;
    fprintf('  Ahat: %.1f sec\n', toc);
    
    % E
    tic;
    [n_feat, n_time] = size(e_std_2d);
    if n_feat > n_time
        R = e_std_2d' * e_std_2d / n_feat;
    else
        R = e_std_2d * e_std_2d' / n_time;
    end
    e_eig = sort(max(eig(R), 0), 'descend');
    e_cumvar = cumsum(e_eig) / sum(e_eig);
    clear R ahat_std_2d e_std_2d;
    fprintf('  E: %.1f sec\n', toc);
    
    % --- Find thresholds ---
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        
        ahat_pc = find(ahat_cumvar >= thresh, 1, 'first');
        e_pc = find(e_cumvar >= thresh, 1, 'first');
        
        if isempty(ahat_pc), ahat_pc = NaN; end
        if isempty(e_pc), e_pc = NaN; end
        
        results(layer_idx).ahat_threshold(i) = ahat_pc;
        results(layer_idx).e_threshold(i) = e_pc;
    end
    
    % Variance at 500 PCs
    if length(ahat_cumvar) >= 500
        results(layer_idx).ahat_var_500 = ahat_cumvar(500) * 100;
    else
        results(layer_idx).ahat_var_500 = ahat_cumvar(end) * 100;
    end
    
    if length(e_cumvar) >= 500
        results(layer_idx).e_var_500 = e_cumvar(500) * 100;
    else
        results(layer_idx).e_var_500 = e_cumvar(end) * 100;
    end
    
    % Store cumvar for plotting
    results(layer_idx).ahat_cumvar = ahat_cumvar;
    results(layer_idx).e_cumvar = e_cumvar;
    results(layer_idx).n_features = dims_ahat(1:end-1);
end

%% --- Summary Table ---
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('SUMMARY: PCs needed for each threshold\n');
fprintf('%s\n', repmat('=', 1, 70));

fprintf('\n--- 90%% Variance ---\n');
fprintf('+-------+----------------+----------------+------------+\n');
fprintf('| Layer |   Ahat (Pred)  |    E (Error)   | Difference |\n');
fprintf('+-------+----------------+----------------+------------+\n');
for layer_idx = 1:NUM_LAYERS
    ahat_pc = results(layer_idx).ahat_threshold(1);
    e_pc = results(layer_idx).e_threshold(1);
    if isnan(ahat_pc), ahat_str = 'N/A'; else, ahat_str = sprintf('%d', ahat_pc); end
    if isnan(e_pc), e_str = 'N/A'; else, e_str = sprintf('%d', e_pc); end
    if ~isnan(ahat_pc) && ~isnan(e_pc)
        diff_str = sprintf('%+d', e_pc - ahat_pc);
    else
        diff_str = 'N/A';
    end
    fprintf('| L%d    | %14s | %14s | %10s |\n', layer_idx, ahat_str, e_str, diff_str);
end
fprintf('+-------+----------------+----------------+------------+\n');

fprintf('\n--- 95%% Variance ---\n');
fprintf('+-------+----------------+----------------+------------+\n');
fprintf('| Layer |   Ahat (Pred)  |    E (Error)   | Difference |\n');
fprintf('+-------+----------------+----------------+------------+\n');
for layer_idx = 1:NUM_LAYERS
    ahat_pc = results(layer_idx).ahat_threshold(2);
    e_pc = results(layer_idx).e_threshold(2);
    if isnan(ahat_pc), ahat_str = 'N/A'; else, ahat_str = sprintf('%d', ahat_pc); end
    if isnan(e_pc), e_str = 'N/A'; else, e_str = sprintf('%d', e_pc); end
    if ~isnan(ahat_pc) && ~isnan(e_pc)
        diff_str = sprintf('%+d', e_pc - ahat_pc);
    else
        diff_str = 'N/A';
    end
    fprintf('| L%d    | %14s | %14s | %10s |\n', layer_idx, ahat_str, e_str, diff_str);
end
fprintf('+-------+----------------+----------------+------------+\n');

fprintf('\n--- 99%% Variance ---\n');
fprintf('+-------+----------------+----------------+------------+\n');
fprintf('| Layer |   Ahat (Pred)  |    E (Error)   | Difference |\n');
fprintf('+-------+----------------+----------------+------------+\n');
for layer_idx = 1:NUM_LAYERS
    ahat_pc = results(layer_idx).ahat_threshold(3);
    e_pc = results(layer_idx).e_threshold(3);
    if isnan(ahat_pc), ahat_str = 'N/A'; else, ahat_str = sprintf('%d', ahat_pc); end
    if isnan(e_pc), e_str = 'N/A'; else, e_str = sprintf('%d', e_pc); end
    if ~isnan(ahat_pc) && ~isnan(e_pc)
        diff_str = sprintf('%+d', e_pc - ahat_pc);
    else
        diff_str = 'N/A';
    end
    fprintf('| L%d    | %14s | %14s | %10s |\n', layer_idx, ahat_str, e_str, diff_str);
end
fprintf('+-------+----------------+----------------+------------+\n');

fprintf('\n--- Variance at current cap (500 PCs) ---\n');
fprintf('+-------+----------------+----------------+\n');
fprintf('| Layer |  Ahat Var (%%)  |   E Var (%%)    |\n');
fprintf('+-------+----------------+----------------+\n');
for layer_idx = 1:NUM_LAYERS
    fprintf('| L%d    | %14.1f | %14.1f |\n', layer_idx, ...
        results(layer_idx).ahat_var_500, results(layer_idx).e_var_500);
end
fprintf('+-------+----------------+----------------+\n');

%% --- Plot: All Layers Comparison ---
fprintf('\nGenerating plots...\n');

figure('Position', [100, 100, 1600, 800]);

% Colors for layers
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], ...
          [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]};

% Left: Ahat all layers
subplot(1, 2, 1);
hold on;
for layer_idx = 1:NUM_LAYERS
    cumvar = results(layer_idx).ahat_cumvar;
    n_plot = min(length(cumvar), 10000);
    plot(1:n_plot, cumvar(1:n_plot)*100, '-', 'LineWidth', 2, ...
        'Color', colors{layer_idx}, 'DisplayName', sprintf('L%d', layer_idx));
end
yline(90, '--', '90%', 'Color', 'r', 'LineWidth', 1.5);
yline(95, '--', '95%', 'Color', [1 0.5 0], 'LineWidth', 1.5);
yline(99, '--', '99%', 'Color', 'g', 'LineWidth', 1.5);
xline(500, ':', 'cap=500', 'Color', 'k', 'LineWidth', 2);
xlabel('Number of Principal Components', 'FontSize', 12);
ylabel('Cumulative Explained Variance (%)', 'FontSize', 12);
title('Ahat (Prediction): All Layers', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 11);
grid on;
ylim([0 105]);

% Right: E all layers
subplot(1, 2, 2);
hold on;
for layer_idx = 1:NUM_LAYERS
    cumvar = results(layer_idx).e_cumvar;
    n_plot = min(length(cumvar), 10000);
    plot(1:n_plot, cumvar(1:n_plot)*100, '-', 'LineWidth', 2, ...
        'Color', colors{layer_idx}, 'DisplayName', sprintf('L%d', layer_idx));
end
yline(90, '--', '90%', 'Color', 'r', 'LineWidth', 1.5);
yline(95, '--', '95%', 'Color', [1 0.5 0], 'LineWidth', 1.5);
yline(99, '--', '99%', 'Color', 'g', 'LineWidth', 1.5);
xline(500, ':', 'cap=500', 'Color', 'k', 'LineWidth', 2);
xlabel('Number of Principal Components', 'FontSize', 12);
ylabel('Cumulative Explained Variance (%)', 'FontSize', 12);
title('E (Error): All Layers', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 11);
grid on;
ylim([0 105]);

saveas(gcf, fullfile(SAVE_DIR, 'cumvar_all_layers.png'));
fprintf('  Saved: cumvar_all_layers.png\n');

%% --- Plot: Bar chart summary ---
figure('Position', [100, 100, 1400, 400]);

% 90% threshold bar chart
subplot(1, 3, 1);
ahat_90 = [results.ahat_threshold];
ahat_90 = ahat_90(1:3:end);  % Every 3rd element (90%)
e_90 = [results.e_threshold];
e_90 = e_90(1:3:end);

x = 1:NUM_LAYERS;
width = 0.35;
bar(x - width/2, ahat_90, width, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'k', 'DisplayName', 'Ahat');
hold on;
bar(x + width/2, e_90, width, 'FaceColor', [1 0.4 0.4], 'EdgeColor', 'k', 'DisplayName', 'E');
yline(500, '--', 'Current cap', 'Color', 'k', 'LineWidth', 2);
xlabel('Layer', 'FontSize', 12);
ylabel('PCs needed', 'FontSize', 12);
title('90% Variance Threshold', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
xticks(1:NUM_LAYERS);
xticklabels({'L1', 'L2', 'L3', 'L4'});
grid on;

% 95% threshold bar chart
subplot(1, 3, 2);
ahat_95 = [results.ahat_threshold];
ahat_95 = ahat_95(2:3:end);  % Every 3rd element starting from 2 (95%)
e_95 = [results.e_threshold];
e_95 = e_95(2:3:end);

bar(x - width/2, ahat_95, width, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'k', 'DisplayName', 'Ahat');
hold on;
bar(x + width/2, e_95, width, 'FaceColor', [1 0.4 0.4], 'EdgeColor', 'k', 'DisplayName', 'E');
yline(500, '--', 'Current cap', 'Color', 'k', 'LineWidth', 2);
xlabel('Layer', 'FontSize', 12);
ylabel('PCs needed', 'FontSize', 12);
title('95% Variance Threshold', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
xticks(1:NUM_LAYERS);
xticklabels({'L1', 'L2', 'L3', 'L4'});
grid on;

% Variance at 500 PCs bar chart
subplot(1, 3, 3);
ahat_var = [results.ahat_var_500];
e_var = [results.e_var_500];

bar(x - width/2, ahat_var, width, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'k', 'DisplayName', 'Ahat');
hold on;
bar(x + width/2, e_var, width, 'FaceColor', [1 0.4 0.4], 'EdgeColor', 'k', 'DisplayName', 'E');
xlabel('Layer', 'FontSize', 12);
ylabel('Variance Explained (%)', 'FontSize', 12);
title('Variance at 500 PCs', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
xticks(1:NUM_LAYERS);
xticklabels({'L1', 'L2', 'L3', 'L4'});
ylim([0 100]);
grid on;

saveas(gcf, fullfile(SAVE_DIR, 'pca_threshold_summary.png'));
fprintf('  Saved: pca_threshold_summary.png\n');

%% --- Save Results ---
save(fullfile(SAVE_DIR, 'pca_threshold_results.mat'), 'results', 'thresholds');
fprintf('  Saved: pca_threshold_results.mat\n');

%% --- Final Summary ---
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('FINAL SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 70));

fprintf('\nCurrent PCA cap: 500\n');
fprintf('\nAt 500 PCs, variance explained:\n');
for layer_idx = 1:NUM_LAYERS
    fprintf('  L%d: Ahat=%.1f%%, E=%.1f%%\n', layer_idx, ...
        results(layer_idx).ahat_var_500, results(layer_idx).e_var_500);
end

fprintf('\nTo reach 90%% variance, need:\n');
for layer_idx = 1:NUM_LAYERS
    ahat_pc = results(layer_idx).ahat_threshold(1);
    e_pc = results(layer_idx).e_threshold(1);
    fprintf('  L%d: Ahat=%d, E=%d\n', layer_idx, ahat_pc, e_pc);
end

fprintf('\nRecommendation:\n');
max_90 = max([results.ahat_threshold]);
max_90 = max(max_90(1:3:end));  % Max of 90% thresholds
fprintf('  For 90%% variance across all layers: cap >= %d\n', max_90);

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Analysis complete!\n');
fprintf('%s\n', repmat('=', 1, 70));