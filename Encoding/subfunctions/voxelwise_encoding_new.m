function [W, Rmat, Lambda] = voxelwise_encoding_new(Y, X, lambda, nfold)
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

% Ensure folds are of integer size
fold_indices = round(linspace(1, Nt + 1, nfold + 1));

Rmat = zeros(Nv,length(lambda), nfold,'single');

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
    YTY = Y_t'*Y_t;

    for k = 1 : length(lambda)
        fprintf('  Fold: %d, Lambda: %d/%d\n', nf, k, length(lambda));
        lmb = lambda(k);

        % Closed-form solution for Ridge Regression weights
        M = (YTY + lmb*T*eye(Nc)) \ (Y_t' * X_t);

        % Predict on validation set
        X_vp = Y_v * M; % predicted

        % Calculate correlation for all voxels at once
        R = diag(corr(X_v, X_vp));
        R(isnan(R)) = 0; % Replace NaN correlations with 0
        Rmat(:,k,nf) = R;
    end
end

% Choose optimal regularization parameter for each voxel
[~, max_idx] = max(mean(Rmat, 3), [], 2);
Lambda = lambda(max_idx);

% Training with optimal regularization parameters on all training data
disp('Training final encoding models with optimal parameters..');
YTY = Y'*Y;
W = zeros(Nc, Nv, 'single');

unique_lambdas = unique(Lambda);
for i = 1:length(unique_lambdas)
    lmb = unique_lambdas(i);
    fprintf('  Training for lambda = %.2f\n', lmb);

    % Efficiently solve for all voxels that share this optimal lambda
    voxel_indices = (Lambda == lmb);
    M = (YTY + lmb*Nt*eye(Nc)) \ (Y' * X(:, voxel_indices));
    W(:,voxel_indices) = M;
end

end
