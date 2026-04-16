function layer_region_crossmatrix(avg_corr_layers, region_labels, region_names)

% avg_corr_layers: [n_layer x n_voxels] matrix
% region_labels: vector length n_voxels (e.g., Glasser labels)
% region_names: cell array of region names corresponding to unique(region_labels)

n_layers = size(avg_corr_layers,1);
unique_regions = unique(region_labels(region_labels>0)); % skip 0
n_regions = length(unique_regions);

crossmat = zeros(n_layers, n_regions);

for r = 1:n_regions
    vox_in_region = (region_labels == unique_regions(r));
    for l = 1:n_layers
        crossmat(l,r) = mean(avg_corr_layers(l, vox_in_region));
    end
end

% Plot heatmap
figure;
imagesc(crossmat);
colorbar;
xlabel('Region');
ylabel('Layer');
set(gca, 'XTick', 1:n_regions, 'XTickLabel', region_names, 'XTickLabelRotation', 45);
set(gca, 'YTick', 1:n_layers, 'YTickLabel', arrayfun(@(x) sprintf('Layer %d', x), 1:n_layers, 'UniformOutput', false));
title('Layer x Region Encoding Correlation');
colormap('viridis'); % needs cmocean or viridis colormap, or default

end
