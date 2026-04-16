%% ratio calculation for sub avg for total seg 

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test';

% Load prediction+error correlation
total = load([saveroot, '/', 'correlations', '/','PCA', '/', '6layers', '/', 'ratio','/', 'total','/', 'layer1_avg_corr_svd0.90.mat']);

% Load prediction correlation
prediction = load([saveroot, '/', 'correlations', '/','PCA', '/', '6layers', '/', 'total','/', 'Ahat_avg_corr_svd0.90.mat']);

% Load error correlation
error = load([saveroot, '/', 'correlations', '/','PCA', '/', '6layers', '/',  'total','/', 'E_avg_corr_svd0.90.mat']);


p_ratio = (abs(predic_corr) ./ abs(correlations')) *100
e_ratio= (abs(error_corr) ./ abs(correlations')) *100

num_voxels = 5

% Initialize arrays to store results
coefficient_comparison = zeros(1, num_voxels);

for i = 1:num_voxels
     % Fit a linear regression model for the total data for the current voxel
            mdl_total = fitlm(X{1}(i), sub_cii1_4(i));
            
            % Fit a linear regression model for the prediction data for the current voxel
            mdl_prediction = fitlm(pre_Yhat(i), sub_cii1_4(i));

            % Fit a linear regression model for the error data for the current voxel
            mdl_error = fitlm(error_Yhat(i), sub_cii1_4(i));

            % Compare the coefficients to determine which predictor contributes more
            contribution_diff = mdl_prediction.Coefficients.Estimate(2) - mdl_error.Coefficients.Estimate(2);
    
    if contribution_diff > 0
        contribution_comparison(i) = 1; % Prediction is more dominant
    elseif contribution_diff < 0
        contribution_comparison(i) = -1; % Error is more dominant
    else
        contribution_comparison(i) = 0; % Equal contribution
    end
    
    contribution_values(i) = abs(contribution_diff);
end


%% corr - prof.meeting.used

% Initialize arrays to store results
coefficient_comparison = zeros(1, num_voxels);

for i = 1:num_voxels
     % Fit a linear regression model for the total data for the current voxel
            mdl_total = fitlm(X{1}(i), sub_cii1_4(i));
            
            % Fit a linear regression model for the prediction data for the current voxel
            mdl_prediction = fitlm(pre_Yhat(i), sub_cii1_4(i));

            % Fit a linear regression model for the error data for the current voxel
            mdl_error = fitlm(error_Yhat(i), sub_cii1_4(i));

            % Compare the coefficients to determine which predictor contributes more
            contribution_diff = mdl_prediction.Coefficients.Estimate(2) - mdl_error.Coefficients.Estimate(2);
    
    if contribution_diff > 0
        contribution_comparison(i) = 1; % Prediction is more dominant
    elseif contribution_diff < 0
        contribution_comparison(i) = -1; % Error is more dominant
    else
        contribution_comparison(i) = 0; % Equal contribution
    end
    
    contribution_values(i) = abs(contribution_diff);
end


% Analyze or visualize the contribution_comparison and contribution_values arrays
num_prediction_dominant = sum(contribution_comparison == 1);
num_error_dominant = sum(contribution_comparison == -1);
num_equal_contribution = sum(contribution_comparison == 0);

fprintf('Prediction is dominant for %.2f%% of voxels.\n', num_prediction_dominant / num_voxels * 100);
fprintf('Error is dominant for %.2f%% of voxels.\n', num_error_dominant / num_voxels * 100);
fprintf('Equal contribution for %.2f%% of voxels.\n', num_equal_contribution / num_voxels * 100);

% Display voxel indices with the highest contribution difference
[sorted_contributions, sorted_indices] = sort(contribution_values, 'descend');
fprintf('Top voxels with the highest contribution differences:\n');
for k = 1:num_voxels
    fprintf('Voxel %d: Difference = %.4f, Contribution Comparison: ', k, contribution_values(k));
    
    if contribution_comparison(k) == 1
        fprintf('Prediction Dominant\n');
    elseif contribution_comparison(k) == -1
        fprintf('Error Dominant\n');
    else
        fprintf('Equal Contribution\n');
    end
end
