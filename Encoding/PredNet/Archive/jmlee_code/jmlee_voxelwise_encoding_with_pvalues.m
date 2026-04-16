function [significant_W] = jmlee_voxelwise_encoding_with_pvalues(Y, X, lambda, nfold, threshold)

    Nt = size(Y,1);
    Nc = size(Y,2);
    dT = Nt/nfold;
    dT = fix(dT);
    T = Nt - dT;
    Nv = size(X,2);

    Rmat = zeros(Nv,length(lambda), nfold,'single');

    % Initialize a cell array to store significant W matrices
    significant_W = cell(Nv, 1);

    disp('Validating regularization parameters ..');
    for nf = 1 : nfold
        idx_t = 1:Nt;
        idx_v = (nf-1)*dT+1:nf*dT;
        idx_t(idx_v) = [];

        Y_t = Y(idx_t,:);
        Y_v = Y(idx_v,:);
        YTY = Y_t' * Y_t;
        X_v = X(idx_v,:);
        X_t = X(idx_t,:);

        % Initialize W_mat to store W matrices for each voxel and lambda
        W_mat = cell(Nv, length(lambda));

        for k = 1 : length(lambda)
            disp(['fold: ', num2str(nf),' lambda: ',num2str(k)]);
            lmb = lambda(k);
            M = (YTY + lmb * T * eye(Nc)) \ Y_t'; 

            R = zeros(Nv,1);
            v1 = 1;
            while (v1 <= Nv)
                v2 = min(v1+4999,Nv);    
                W = M * X_t(:,v1:v2);
                X_vp = Y_v * W; % predicted
                X_vg = X_v(:,v1:v2); % ground truth
                R(v1:v2) = amri_sig_corr(X_vg, X_vp, 'mode', 'auto');
                v1 = v2 + 1;

                % Store W matrices for each voxel and lambda
                for vv = v1:v2
                    W_mat{vv, k} = W(:, vv - v1 + 1);
                end
            end
            Rmat(:, k, nf) = R;
        end

        % Perform t-test and save significant W matrices
        for v = 1:Nv
            W_mat_v = cell(1, length(lambda));  % Initialize for each voxel
            for k = 1:length(lambda)
                [~, ~, ~, stats] = ttest(W_mat{v, k});

                % Check if the t-test is significant based on the p-value
                if stats.tstat < 0 && stats.tstat < -threshold || ...
                   stats.tstat > 0 && stats.tstat > threshold
                    W_mat_v{k} = W_mat{v, k};
                end
            end

            % Assign the non-empty W matrices to the significant_W cell array
            non_empty_indices = ~cellfun('isempty', W_mat_v);
            if any(non_empty_indices)
                if isempty(significant_W{v, 1})
                    significant_W{v, 1} = cat(2, W_mat_v{non_empty_indices});
                else
                    significant_W{v, 1} = [significant_W{v, 1}, cat(2, W_mat_v{non_empty_indices})];
                end
            end
        end
    end
end
