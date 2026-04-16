function [voxel_correlations, p_values] = auto_corr_jm(y_true, y_pred)
    [num_voxels, num_time] = size(y_true);
    voxel_correlations = zeros(num_voxels,1);
    p_values = zeros(num_voxels,1);
    for v = 1:num_voxels
        original_signal = y_true(v, :);  % 1xtime
        pred_signal = y_pred(v, :);      % 1xtime
        [coef, p] = corrcoef(original_signal, pred_signal);
        voxel_correlations(v) = coef(1, 2);
        p_values(v) = p(1, 2);
    end
end
