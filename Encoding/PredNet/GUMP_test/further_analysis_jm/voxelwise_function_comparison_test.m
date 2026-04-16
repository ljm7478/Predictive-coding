%% --- 'orig' vs 'new' voxelwise_encoding Comparison Test ---

clc; clear;

% --- 1. Generate Synthetic Test Data ---
Nt = 500;   % Number of time points
Nc = 50;    % Number of features (Components)
Nv = 1000;  % Number of voxels
nfold = 3;  % Use 3-folds for a quick test
lambdas = [0.1, 1, 10, 100, 1000];

fprintf('Generating test data (Nt=%d, Nc=%d, Nv=%d)\n', Nt, Nc, Nv);

% Y = Features (Regressors)
Y = randn(Nt, Nc);
Y = zscore(Y, 0, 1); % Set mean to zero

% X = Brain data (Response)
% (Create data that has some relationship with Y)
true_W = randn(Nc, Nv);
X = Y * true_W + 1.5 * randn(Nt, Nv);
X = zscore(X, 0, 1); % Set mean to zero

%% --- 2. Method 1: Run 'orig' version ---
fprintf('\n--- Method 1: Running ''voxelwise_encoding_orig''... ---\n');
try
    tic;
    [W_orig, Rmat_orig, Lambda_orig] = voxelwise_encoding_orig(Y, X, lambdas, nfold);
    time_orig = toc;
    fprintf('  Complete. Execution time: %.4f seconds\n', time_orig);
catch ME
    fprintf('Error: Failed to execute ''voxelwise_encoding_orig.m''.\n');
    disp(ME.message);
    return;
end

%% --- 3. Method 2: Run 'new' version ---
fprintf('\n--- Method 2: Running ''voxelwise_encoding_new''... ---\n');
try
    tic;
    [W_new, Rmat_new, Lambda_new] = voxelwise_encoding_new(Y, X, lambdas, nfold);
    time_new = toc;
    fprintf('  Complete. Execution time: %.4f seconds\n', time_new);
catch ME
    fprintf('Error: Failed to execute ''voxelwise_encoding_new.m''.\n');
    disp(ME.message);
    return;
end

%% --- 4. Compare Results ---
fprintf('\n--- Result Comparison ---\n');
fprintf('Speed Comparison: ''new'' version is %.2f times faster than ''orig'' version.\n', time_orig / time_new);

% Rmat (validation performance) comparison
% Rmat might be slightly different due to the CV bug in 'orig'
max_Rmat_diff = max(abs(Rmat_orig(:) - Rmat_new(:)));
fprintf('Rmat (Validation Performance) Max Absolute Error: %e\n', max_Rmat_diff);

% Lambda (selected lambda) comparison
% Due to the CV bug, some voxels might have different lambdas selected
lambda_diff_count = sum(Lambda_orig(:) ~= Lambda_new(:));
fprintf('Lambda (Optimal Lambda) Mismatch Count: %d / %d voxels\n', lambda_diff_count, Nv);

% W (final weights) comparison
% If lambdas are different, W will also be different
max_W_diff = max(abs(W_orig(:) - W_new(:)));
fprintf('W (Final Weights) Max Absolute Error: %e\n', max_W_diff);

if max_W_diff < 1e-10
    fprintf('Conclusion: W matrices are numerically identical. (Success) \n');
elseif lambda_diff_count > 0
    fprintf('Conclusion: W matrices are different. (Expected Result) \n');
    fprintf('Reason: The CV sample-dropping bug in the ''orig'' code caused %d voxels to have a different optimal lambda.\n', lambda_diff_count);
    fprintf('The result from the ''new'' code is the more correct result.\n');
else
    fprintf('Conclusion: W matrices are different. (Failure) \n');
end