function [W, Rmat, Lambda] = voxelwise_encoding_new_gpu(Y, X, lambda, nfold)
% inputs:
%   - Y: Nt-by-Nc matrix, each column is a regressor (mean = zero)
%   - X: Nt-by-Nv matrix, each column is the response time series (mean=zero) of a voxel
%   - lambda: a vector, a candidate set of regularization parameters
%   - nfold: a scalar, the number of folds to do cross validation
% outputs:
%   - W: Nc-by-Nv matrix, each column is the optimal encoding weights of a voxel
%   - Rmat: Nv*(#lambda)*nfold array, validation accuracy (correlation)
%   - Lambda: optimal regularization parameters for each voxel

Nt = size(Y,1); % number of total time points
Nc = size(Y,2); % number of components
Nv = size(X,2); % number of voxels

% Keep on CPU as single
Y = single(Y);
X = single(X);

% Batch size for voxels (adjust based on GPU memory)
voxel_batch_size = 5000;

% Ensure folds are of integer size
fold_indices = round(linspace(1, Nt + 1, nfold + 1));
Rmat = zeros(Nv, length(lambda), nfold, 'single');

disp('Validating regularization parameters ..');
for nf = 1 : nfold
    % Define validation and training indices for this fold
    idx_v = fold_indices(nf) : fold_indices(nf+1)-1;
    idx_t = setdiff(1:Nt, idx_v);
    Y_t = Y(idx_t,:);
    Y_v = Y(idx_v,:);
    X_t = X(idx_t,:);
    X_v = X(idx_v,:);
    T = size(Y_t, 1);
    
    % Compute on CPU
    YTY = Y_t' * Y_t;
    YTX = Y_t' * X_t;
    
    for k = 1 : length(lambda)
        fprintf('  Fold: %d, Lambda: %d/%d\n', nf, k, length(lambda));
        lmb = lambda(k);
        
        % Solve on GPU in batches
        YTY_gpu = gpuArray(YTY + lmb * T * eye(Nc, 'single'));
        
        num_batches = ceil(Nv / voxel_batch_size);
        for b = 1:num_batches
            idx_start = (b-1) * voxel_batch_size + 1;
            idx_end = min(b * voxel_batch_size, Nv);
            voxel_idx = idx_start:idx_end;
            
            % Transfer batch to GPU
            YTX_batch = gpuArray(YTX(:, voxel_idx));
            
            % Closed-form solution for Ridge Regression weights
            M = YTY_gpu \ YTX_batch;
            
            % Predict on validation set
            Y_v_gpu = gpuArray(Y_v);
            X_vp = Y_v_gpu * M;
            X_v_batch = gpuArray(X_v(:, voxel_idx));
            
            % Calculate correlation for batch
            R = diag(corr(gather(X_v_batch), gather(X_vp)));
            R(isnan(R)) = 0;
            Rmat(voxel_idx, k, nf) = R;
            
            % Clear GPU memory
            clear YTX_batch M X_vp X_v_batch Y_v_gpu;
        end
        clear YTY_gpu;
    end
end

% Choose optimal regularization parameter for each voxel
[~, max_idx] = max(mean(Rmat, 3), [], 2);
Lambda = lambda(max_idx);

% Training with optimal regularization parameters on all training data
disp('Training final encoding models with optimal parameters..');
YTY = Y' * Y;
YTX = Y' * X;
W = zeros(Nc, Nv, 'single');

unique_lambdas = unique(Lambda);
for i = 1:length(unique_lambdas)
    lmb = unique_lambdas(i);
    fprintf('  Training for lambda = %.2f\n', lmb);
    
    voxel_indices = find(Lambda == lmb);
    YTY_gpu = gpuArray(YTY + lmb * Nt * eye(Nc, 'single'));
    
    % Process in batches
    num_voxels_this_lambda = length(voxel_indices);
    num_batches_lambda = ceil(num_voxels_this_lambda / voxel_batch_size);
    
    for b = 1:num_batches_lambda
        idx_start = (b-1) * voxel_batch_size + 1;
        idx_end = min(b * voxel_batch_size, num_voxels_this_lambda);
        batch_voxel_idx = voxel_indices(idx_start:idx_end);
        
        YTX_batch = gpuArray(YTX(:, batch_voxel_idx));
        M = YTY_gpu \ YTX_batch;
        W(:, batch_voxel_idx) = gather(M);
        
        clear YTX_batch M;
    end
    clear YTY_gpu;
end

end