function ratio_map = pe_p_ratio_map(pred_corr, pe_corr, epsilon)

% pred_corr, pe_corr: [n_layers x n_voxels] correlation maps
% epsilon: small constant to avoid division by zero (default 1e-4)

if nargin < 3
    epsilon = 1e-4;
end

ratio_map = zeros(size(pred_corr));

for l = 1:size(pred_corr,1)
    ratio_map(l,:) = (pred_corr(l,:) + epsilon) ./ (pe_corr(l,:) + epsilon);
end

% Optionally visualize
figure;
imagesc(ratio_map);
colorbar;
xlabel('Voxel');
ylabel('Layer');
title('Prediction / PE ratio map');
colormap('hot');

end
