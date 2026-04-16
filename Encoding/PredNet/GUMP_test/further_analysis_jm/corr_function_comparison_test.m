%% --- Comparison Test for Two Correlation Calculation Methods ---

clc;
clear;

% --- 1. Generate Synthetic Test Data ---
Nt = 100; % Number of time points
Nv = 5000; % Number of voxels

fprintf('Generating test data (Nt=%d, Nv=%d)\n', Nt, Nv);

% A = "Actual" brain data (ground truth)
A = randn(Nt, Nv); 

% B = "Predicted" brain data
% Add noise to A to create a correlated signal
B = A + 0.5 * randn(Nt, Nv); 

%% --- 2. Method 1: amri_sig_corr (Custom Function) ---
% (Requires amri_sig_corr.m to be in the MATLAB path)
fprintf('Running Method 1: amri_sig_corr(A, B, ''mode'', ''auto'')...\n');
try
    tic; % Start timer
    % 'auto' mode compares corresponding columns (col 1 of A vs col 1 of B, etc.)
    R_amri = amri_sig_corr(A, B, 'mode', 'auto');
    toc; % Stop timer
    
    % amri_sig_corr returns a (1 x Nv) row vector, so reshape to (Nv x 1) column vector
    R_amri = R_amri(:); 
    
catch ME
    fprintf('Error: Could not find or execute amri_sig_corr.m.\n');
    fprintf('%s\n', ME.message);
    return;
end

%% --- 3. Method 2: diag(corr(...)) (Standard Function) ---
fprintf('Running Method 2: diag(corr(A, B))...\n');
tic; % Start timer

% corr(A, B) creates a large intermediate (Nv x Nv) matrix
R_corr_matrix = corr(A, B); 

% Extract only the diagonal elements, which we need
R_diag = diag(R_corr_matrix); 
toc; % Stop timer

%% --- 4. Compare Results ---
fprintf('--- Result Comparison ---\n');

% Check if both vectors were successfully computed
if exist('R_amri', 'var') && exist('R_diag', 'var')
    % Check if the sizes of the two vectors are identical
    if isequal(size(R_amri), size(R_diag))
        fprintf('Vector sizes are identical: ( %d x %d )\n', size(R_amri, 1), size(R_amri, 2));
        
        % Calculate the element-wise difference between vectors
        difference = R_amri - R_diag;
        
        % Find the maximum absolute difference (to check for floating-point errors)
        max_abs_diff = max(abs(difference));
        
        fprintf('Maximum absolute difference between vectors: %e\n', max_abs_diff);
        
        % Check if the difference is negligible (e.g., < 1e-10)
        if max_abs_diff < 1e-10
            fprintf('Result: The two methods are numerically identical. (Success) \n');
        else
            fprintf('Result: The two methods produced different results. (Failure) \n');
        end
        
    else
        fprintf('Error: The resulting vectors have different sizes.\n');
    end
else
    fprintf('Error: One of the methods failed, cannot compare.\n');
end