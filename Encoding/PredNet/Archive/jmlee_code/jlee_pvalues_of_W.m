%% Step4:W
clc; clear
disp('Load model featuremaps')

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/4layers_32channel/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

features = { 'Ahat'} %, 'E'};
svd_folder = [saveroot,'/', 'X','/', '4layers_32channel', '/'];

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:3
        for seg = 1:8
            filename = sprintf('%s/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', svd_folder, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end

end

%Voxelwise_encoding 
lambda = [0.1:0.2:0.9];
nfold = 9; %from paper 

%load Y
load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 1:15
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
   for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        for seg = 1:8
            disp(['Segment: ', num2str(seg)]);
           for layer = 0:3
                disp(['Layer: ', num2str(layer)]);
                 % Y calculate 
                switch seg
                    case 1
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 2
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 3
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{4}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 4
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 5
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 6
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{5},sub_cii{i}{7}, sub_cii{i}{8});
                    case 7
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{5},sub_cii{i}{6}, sub_cii{i}{8});
                    case 8
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{5},sub_cii{i}{6}, sub_cii{i}{7});
                end
                 
                % X calculate
                X_tot = {};
                for s = 1:8
                    if s ~= seg
                        disp(['s is ', num2str(s)])
                        disp([sprintf('%s%d_seg%d', feature, layer, s)])
                        X_tot{end+1} = concatenated.(sprintf('%s%d_seg%d', feature, layer, s));
                    end
                end
                X = cat(1, X_tot{:});
                
                %voxelwise encoding
                [significant_W] = jmlee_voxelwise_encoding_with_pvalues(X, Y', lambda, nfold);
                
                % Save W-pvalues (feature x voxel)
                folder_name = fullfile(saveroot, 'significant_W', '4layers_32channel', '10k', sprintf('seg%d', seg));
                if ~exist(folder_name, 'dir')
                    mkdir(folder_name);
                end

                name = sprintf('%s_%s%d_significant_W.mat', subject, feature, layer);
                save(fullfile(folder_name, name), 'significant_W', '-v7.3');
            end
        end
    end
end


