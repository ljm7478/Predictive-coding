
%% Step1:  W calculation  (prediction+error)
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

features = {'Ahat','E'};

ratio={}; 
for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature);
    for layer = 0:5
        for seg = 1:8
            filename = sprintf('%s/X/6layers_per_seg/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', saveroot, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end
end

%concat prediction & error
for layer = 0:5
    for seg = 1:8 
        prediction_data = concatenated.(sprintf('%s%d_seg%d', features{1}, layer, seg));
        error_data = concatenated.(sprintf('%s%d_seg%d', features{2}, layer, seg));
        
        name = sprintf('layer%d_seg%d', layer, seg);
        ratio.(name) = cat(2, prediction_data, error_data);
        eval([name, '= ratio.(name);']);
        
        % Calculate corrcoef to check if concatenation is correct
        dim = size(prediction_data);
        prediction_corr = corrcoef(ratio.(name)(:, 1:dim(2)), prediction_data);
        error_corr = corrcoef(ratio.(name)(:, dim(2)+1:end), error_data);
        
        % Display correlation coefficients
        fprintf('Layer %d, Segment %d:\n', layer, seg);
        fprintf('Prediction Correlation:\n');
        disp(prediction_corr);
        fprintf('Error Correlation:\n');
        disp(error_corr)
    end
end

%Voxelwise_encoding
lambda = [0.1:0.2:0.9];
nfold = 9; %from paper

%load Y
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
subjectIDs = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};


for i = 1:15
    subject = subjectIDs{i};
    disp(['Voxelwise_encoding start sub', num2str(subject), sprintf(' layer%d_seg%d', layer, seg)])
    for seg = 1:8
        disp(['Segment: ', num2str(seg)]);
        for layer = 0:5
            disp(['Layer: ', num2str(layer)])

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

            % X calculate (concat training set)
            X_tot = {};
            concatenated_data={};
            for s = 1:8
                disp([sprintf('current test segment is %s', num2str(seg))])
                if s ~= seg
                    disp([sprintf('layer%d_seg%d', layer, s)])
                    concatenated_data{end+1} = ratio.(sprintf('layer%d_seg%d', layer, s));

                    X_tot{s} = ratio.(sprintf('layer%d_seg%d', layer, s));

                    % Calculate corrcoef to check correct structure
                    corr = corrcoef(X_tot{s}, ratio.(sprintf('layer%d_seg%d', layer, s)));

                    % Display correlation coefficients
                    fprintf('Layer %d, Segment %d:\n', layer, s);
                    disp(corr);
                end
            end

            % Combine the cell array into a single matrix
            X = cat(1, X_tot{:});

            % Calculate corrcoef to check if concatenation is correct
            A = size(concatenated_data{1}, 1);
            B = size(concatenated_data{2}, 1);
            C = size(concatenated_data{3}, 1);
            D = size(concatenated_data{4}, 1);
            E = size(concatenated_data{5}, 1);
            F = size(concatenated_data{6}, 1);
            G = size(concatenated_data{7}, 1);
   
            total_corr1 = corrcoef(X((1:A), :), concatenated_data{1});
            total_corr2 = corrcoef(X((A+1:A+B), :), concatenated_data{2});
            total_corr3 = corrcoef(X((A+B+1:A+B+C), :), concatenated_data{3});
            total_corr4 = corrcoef(X((A+B+C+1:A+B+C+D), :), concatenated_data{4});
            total_corr5 = corrcoef(X((A+B+C+D+1:A+B+C+D+E), :), concatenated_data{5});
            total_corr6 = corrcoef(X((A+B+C+D+E+1:A+B+C+D+E+F), :), concatenated_data{6});
            total_corr7 = corrcoef(X((A+B+C+D+E+F+1:A+B+C+D+E+F+G), :), concatenated_data{7});
 

            fprintf('Total Correlation 1:\n');
            disp(total_corr1);
            fprintf('Total Correlation 2:\n');
            disp(total_corr2);
            fprintf('Total Correlation 3:\n');
            disp(total_corr3);
            fprintf('Total Correlation 4:\n');
            disp(total_corr4);
            fprintf('Total Correlation 5:\n');
            disp(total_corr5);
            fprintf('Total Correlation 6:\n');
            disp(total_corr6);
            fprintf('Total Correlation 7:\n');
            disp(total_corr7);

            %voxelwise encoding
            [W, Rmat, Lambda] = voxelwise_encoding(X, Y', lambda, nfold);

            % save W (feature x voxel)
            folder_name = [saveroot,'/', 'W','/','PCA','/', '6layers', '/', 'ratio','/', sprintf('seg%d',seg)]
            if not(exist(folder_name,'dir'))
                mkdir(folder_name)
            end

            name = [num2str(subject), '_', sprintf('layer%d_seg%d', layer, seg), '_W_svd0.9.mat']
            save([folder_name,'/', name], 'W', '-v7.3');
        end
    end
end


%% Step2: Yhat calc - total P+E - single segment

% X model featuremaps 
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

features = {'Ahat','E'};

ratio={}; 
for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature);
    for layer = 0:5
        for seg = 1:8
            filename = sprintf('%s/X/6layers_per_seg/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', saveroot, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end
end

%concat prediction & error
for layer = 0:5
    for seg = 1:8 
        prediction_data = concatenated.(sprintf('%s%d_seg%d', features{1}, layer, seg));
        error_data = concatenated.(sprintf('%s%d_seg%d', features{2}, layer, seg));
        
        name = sprintf('layer%d_seg%d', layer, seg);
        ratio.(name) = cat(2, prediction_data, error_data);
        eval([name, '= ratio.(name);']);
        
        % Calculate corrcoef to check if concatenation is correct
        dim = size(prediction_data);
        prediction_corr = corrcoef(ratio.(name)(:, 1:dim(2)), prediction_data);
        error_corr = corrcoef(ratio.(name)(:, dim(2)+1:end), error_data);
        
        % Display correlation coefficients
        fprintf('Layer %d, Segment %d:\n', layer, seg);
        fprintf('Prediction Correlation:\n');
        disp(prediction_corr);
        fprintf('Error Correlation:\n');
        disp(error_corr)
    end
end

disp('make Y_hat')
subjectIDs = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 1:15
    subject = subjectIDs{i}
    disp(['Voxelwise_encoding start sub ', num2str(subject)])

    for seg=1:8

        for layer=0:5

            % X calculate
            X = {};
            for s = 1:8
                if s == seg
                    disp(['s is ', num2str(s)])
                    disp([sprintf('layer%d_seg%d', layer, s)])
                    X = ratio.(sprintf('layer%d_seg%d', layer, s));
                end
            end

            % save Y_hat
            folder_name = [saveroot,'/', 'Y_hat','/6layers_per_seg', '/', 'ratio','/', sprintf('seg%d',seg),'/']
            if not(exist(folder_name,'dir'))
                mkdir(folder_name)
            end

            W = load([saveroot,'/', 'W', '/PCA','/', '6layers', '/', 'ratio','/',  sprintf('seg%d',seg), '/', num2str(subject), '_', sprintf('layer%d_seg%d', layer, seg), '_W_svd0.9.mat']);
            name = 'Y_pred';
            eval([name, '= X * W.W;']);

            Yhat = eval(name);
            name2 = strcat(num2str(subject),'_', sprintf('layer%d_seg%d', layer, seg), '_pred_svd0.90.mat')

            save([folder_name,  name2], 'Yhat', '-v7.3');
        end
    end
end

%% step2: Yhat - prediction/ error single segment
clc; clear
disp('correaltion  compute start')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

subjectIDs = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

features = {'Ahat','E'};

ratio={}; 
for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature);
    for layer = 0:5
        for seg = 1:8
            filename = sprintf('%s/X/6layers_per_seg/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', saveroot, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end
end

%concat prediction & error
for layer = 0:5
    for seg = 1:8 
        prediction_data = concatenated.(sprintf('%s%d_seg%d', features{1}, layer, seg));
        error_data = concatenated.(sprintf('%s%d_seg%d', features{2}, layer, seg));
        
        name = sprintf('layer%d_seg%d', layer, seg);
        ratio.(name) = cat(2, prediction_data, error_data);
%         eval([name, '= ratio.(name);']);
        
        % Calculate corrcoef to check if concatenation is correct
        dim = size(prediction_data);
        prediction_corr = corrcoef(ratio.(name)(:, 1:dim(2)), prediction_data);
        error_corr = corrcoef(ratio.(name)(:, dim(2)+1:end), error_data);
        
        % Display correlation coefficients
        fprintf('Layer %d, Segment %d:\n', layer, seg);
        fprintf('Prediction Correlation:\n');
        disp(prediction_corr);
        fprintf('Error Correlation:\n');
        disp(error_corr)
    end
end

for i = 1:15
    subject = subjectIDs{i}
    
    disp(['correalting' , subject])
    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature);
        for layer= 0:5
            for seg = 1:8
                %folder
                prediction_folder_name = [saveroot,'/', 'ratio','/6layers_per_seg/', 'Y_hat_P','/', sprintf('seg%d',seg),'/']
                if not(exist(prediction_folder_name,'dir'))
                    mkdir(prediction_folder_name)
                end
                
                error_folder_name = [saveroot,'/', 'ratio','/6layers_per_seg/', 'Y_hat_E','/', sprintf('seg%d',seg),'/']
                if not(exist(error_folder_name,'dir'))
                    mkdir(error_folder_name)
                end
                
                
                %load concatenated prediction+error W
                W = load([saveroot,'/', 'ratio', '/6layers_per_seg/', 'W','/',  sprintf('seg%d',seg), '/', num2str(subject), '_', sprintf('layer%d_seg%d', layer, seg), '_W_svd0.9.mat']);
                W = W.W;
                
                % prediction
                concat_data = ratio.(sprintf('layer%d_seg%d', layer, seg));
                name= sprintf('layer%d_seg%d', layer, seg)
                concat_data_dim = size(concat_data)
                
                prediction_data = concatenated.(sprintf('%s%d_seg%d', features{1}, layer, seg));
                pre_dim=size(prediction_data)
                
                %segmentation into each prediction for each layer + seg
                pre_W = W(1:pre_dim(2),:);
                
                %pre_Yhat make
                pre_Yhat = prediction_data * pre_W;
                
                name = [num2str(subject), '_', sprintf('Ahat%d_seg%d', layer, seg), '_W_svd0.9.mat']
                save([prediction_folder_name,'/', name], 'pre_Yhat', '-v7.3');
                
                % Error
                %segmentation into each error for each layer + seg
                err_W = W((pre_dim(2)+1:end),:);
                
                error_data = concatenated.(sprintf('%s%d_seg%d', features{2}, layer, seg));
                err_dim=size(error_data)
                
                %pre_error make
                pre_Error = error_data * err_W;
                
                name = [num2str(subject), '_', sprintf('E%d_seg%d', layer, seg), '_W_svd0.9.mat']
                save([error_folder_name,'/', name], 'pre_Error', '-v7.3');
            end
        end
    end
end

%% ratio map calculation - timeseries regression

clc; 

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

subjectIDs = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

features = {'Ahat','E'};

PT_layer_array={};
ET_layer_array={};
PE_layer_array={};


P_seg_array={};
E_seg_array={};
PE_seg_array={};


load(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test', '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

for i = 1:15
    subject = subjectIDs{i} 
    disp(['correalting' , subject])
    
    % total Y
    sub_cii1_4 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
    dim = size(sub_cii1_4)
    total_time = dim(2)
    total_voxel=dim(1)
    
    for layer= 0:5
        for seg = 1:8
           
            %load Yhat_prediction
            prediction_folder_name = [saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'Y_hat_P','/', sprintf('seg%d',seg),'/']
            pre_Yhat = load([prediction_folder_name,'/', subject, '_', sprintf('Ahat%d_seg%d', layer, seg), '_W_svd0.9.mat'])
            P_seg_array{seg}= pre_Yhat.pre_Yhat;
            

            %load Yhat_error
            error_folder_name = [saveroot,'/', 'ratio','/', '6layers_per_seg', '/','Y_hat_E','/', sprintf('seg%d',seg),'/']
            pre_Error = load([error_folder_name,'/', subject, '_', sprintf('E%d_seg%d', layer, seg), '_W_svd0.9.mat'])
            E_seg_array{seg}=pre_Error.pre_Error;
           
            
            %load total prediction+error
            total_folder_name = [saveroot,'/Y_hat/6layers_per_seg/ratio','/', sprintf('seg%d',seg),'/']
            pre_total = load([total_folder_name,'/', subject, '_', sprintf('layer%d_seg%d', layer, seg), '_pred_svd0.90.mat'])
            PE_seg_array{seg}=pre_total.Yhat;
           
        end
        
        %concatenat Yhat for movie1-4 : voxel by time
        pre_Yhat_total= cat(1, P_seg_array{:});
        pre_Error_total= cat(1, E_seg_array{:});
        pre_total_total= cat(1, PE_seg_array{:});
        
        
        % linear regression calc begins
        PT_array = zeros(1, total_voxel);
        ET_array = zeros(1, total_voxel);
        PE_array = zeros(1, total_voxel);
        
        for v = 1:total_voxel
            
            % Fit a linear regression model for the total prediction+error data for the current voxel
            Yhat_tot = pre_total_total(:,v);
            Y = sub_cii1_4(v,:);
            
            mdl_total = fitlm(Y, Yhat_tot);

            % Fit a linear regression model for the prediction data for the current voxel
            Yhat_pre = pre_Yhat_total(:,v);
            Y = sub_cii1_4(v,:);
            
            mdl_prediction = fitlm(Y, Yhat_pre);
            
            % Fit a linear regression model for the error data for the current voxel
            Yhat_err = pre_Error_total(:,v);
            Y = sub_cii1_4(v,:);
            
            mdl_error = fitlm(Y, Yhat_err);
    
            
            % coefficient divide
            
            PT = mdl_prediction.Coefficients.Estimate(2) / mdl_total.Coefficients.Estimate(2);
            
            ET = mdl_error.Coefficients.Estimate(2) / mdl_total.Coefficients.Estimate(2);
            
            PE = mdl_prediction.Coefficients.Estimate(2) / mdl_error.Coefficients.Estimate(2);
            
            
            % Store the calculated ratios in the arrays
            PT_array(v) = PT;
            ET_array(v) = ET;
            PE_array(v) = PE;
        end
        % Concatenate ratio for each layer
        PT_layer_array{layer+1} = PT_array;
        ET_layer_array{layer+1} = ET_array;
        PE_layer_array{layer+1} = PE_array;
    end
%     make save dir
%     folder_name = [saveroot,'/', 'ratio', '/', '6layers', '/', 'ratio_map_time', '/', 'PE', '/']
%     if not(exist(folder_name,'dir'))
%         mkdir(folder_name)
%     end

    %save layer_array for each subject
    PT_layer_array = cat(1, PT_layer_array{:});
    PT_name = [subject, '_PT_ratio.mat']
    save([saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'ratio_map', '/', 'PT','/', PT_name], 'PT_layer_array', '-v7.3');
    
    ET_layer_array = cat(1, ET_layer_array{:});
    ET_name = [subject, '_ET_ratio.mat']
    save([saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'ratio_map', '/', 'ET','/', ET_name], 'ET_layer_array', '-v7.3');
    
    PE_layer_array = cat(1, PE_layer_array{:});
    PE_name = [subject, '_PE_ratio.mat']
    save([saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'ratio_map', '/', 'PE','/', PE_name], 'PE_layer_array', '-v7.3');
    
    %initialize
    PT_layer_array={};
    ET_layer_array={};
    PE_layer_array={};
end


%% ratio map calculation - timeseries regression - Yhat comparison

clc; 

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

subjectIDs = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

features = {'Ahat','E'};

PT_layer_array={};
ET_layer_array={};
PE_layer_array={};


P_seg_array={};
E_seg_array={};
PE_seg_array={};


load(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test', '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

for i = 1:15
    subject = subjectIDs{i} 
    disp(['correalting' , subject])
    
    % total Y
    sub_cii1_4 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
    dim = size(sub_cii1_4)
    total_time = dim(2)
    total_voxel=dim(1)
    
    for layer= 0:5
        for seg = 1:8
           
            %load Yhat_prediction
            prediction_folder_name = [saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'Y_hat_P','/', sprintf('seg%d',seg),'/']
            pre_Yhat = load([prediction_folder_name,'/', subject, '_', sprintf('Ahat%d_seg%d', layer, seg), '_W_svd0.9.mat'])
            P_seg_array{seg}= pre_Yhat.pre_Yhat;
            

            %load Yhat_error
            error_folder_name = [saveroot,'/', 'ratio','/', '6layers_per_seg', '/','Y_hat_E','/', sprintf('seg%d',seg),'/']
            pre_Error = load([error_folder_name,'/', subject, '_', sprintf('E%d_seg%d', layer, seg), '_W_svd0.9.mat'])
            E_seg_array{seg}=pre_Error.pre_Error;
           
            
            %load total prediction+error
            total_folder_name = [saveroot,'/Y_hat/6layers_per_seg/ratio','/', sprintf('seg%d',seg),'/']
            pre_total = load([total_folder_name,'/', subject, '_', sprintf('layer%d_seg%d', layer, seg), '_pred_svd0.90.mat'])
            PE_seg_array{seg}=pre_total.Yhat;
           
        end
        
        %concatenat Yhat for movie1-4 : voxel by time
        pre_Yhat_total= cat(1, P_seg_array{:});
        pre_Error_total= cat(1, E_seg_array{:});
        pre_total_total= cat(1, PE_seg_array{:});
        
        
        % linear regression calc begins
        PT_array = zeros(1, total_voxel);
        ET_array = zeros(1, total_voxel);
        PE_array = zeros(1, total_voxel);
        
        for v = 1:total_voxel
           
            disp([sprintf('current voxel is %d', v)])
            % Fit a linear regression model for the prediction data for the current voxel
            Yhat_pre = pre_Yhat_total(:,v);
            Yhat_tot = pre_total_total(:,v);
            
            mdl_prediction = fitlm(Yhat_tot, Yhat_pre);
            
            % Fit a linear regression model for the error data for the current voxel
            Yhat_err = pre_Error_total(:,v);
            Yhat_tot = pre_total_total(:,v);
            
            mdl_error = fitlm(Yhat_tot, Yhat_err);

            % coefficient divide
            
            PT = mdl_prediction.Coefficients.Estimate(2) ;
            ET = mdl_error.Coefficients.Estimate(2) ;
            
            % Store the calculated ratios in the arrays
            PT_array(v) = PT;
            ET_array(v) = ET;
        end
        % Concatenate ratio for each layer
        PT_layer_array{layer+1} = PT_array;
        ET_layer_array{layer+1} = ET_array;
    end
%     make save dir
%     folder_name = [saveroot,'/', 'ratio', '/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', 'PT', '/']
%     if not(exist(folder_name,'dir'))
%         mkdir(folder_name)
%     end

    %save layer_array for each subject
    PT_layer_array = cat(1, PT_layer_array{:});
    PT_name = [subject, '_PT_ratio.mat']
    save([saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', 'PT','/', PT_name], 'PT_layer_array', '-v7.3');
    
    ET_layer_array = cat(1, ET_layer_array{:});
    ET_name = [subject, '_ET_ratio.mat']
    save([saveroot,'/', 'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', 'ET','/', ET_name], 'ET_layer_array', '-v7.3');

    %initialize
    PT_layer_array={};
    ET_layer_array={};
    PE_layer_array={};
end

%% Step 6: avg .mat -total
clc;clear

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

subjectIDs = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};


% types={'ET', 'PE', 'PT'};
types={'ET', 'PT'};


for t = 1:length(types)
        type = types{t}
        sum_values=0;
    for i = 1:15
        subject = subjectIDs{i}

        % Load correlation
        value = load([saveroot,'/',  'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/',  type, '/',  strcat(subject,'_', sprintf('%s_ratio.mat', type))]);
        value=value.(sprintf('%s_layer_array', type));

        sum_values = sum_values + value;
    end
    Y_hat = sum_values / 15;
    Y_hat=Y_hat(2:end, :);
    save([saveroot,'/',  'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', type, '/', sprintf('avg_%s_ratio.mat', type)], 'Y_hat', '-v7.3');
    clear Y_hat
end

%% P/E ratio after calculating avg PT, ET ratio.mat

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

%%avg ratio
% % Load PT ratio
PT = load([saveroot,'/',  'ratio/6layers_per_seg/ratio_map_Yhat/PT/avg_PT_ratio.mat']);
PT=PT.Y_hat;

% Load ET ratio
ET = load([saveroot,'/',  'ratio/6layers_per_seg/ratio_map_Yhat/ET/avg_ET_ratio.mat']);
ET = ET.Y_hat;

%divide PE
PE = PT ./ ET;
Y_hat=PE;

% P>E=1
% Y_hat(Y_hat > 1) = 1; 
% Y_hat(Y_hat < 1) = -1; 

% %save PE ratio
% save([saveroot,'/',  'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', 'PE', '/', 'PE_ratio.mat'], 'Y_hat', '-v7.3');
save([saveroot,'/',  'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', 'PE', '/', 'avg_PE_ratio.mat'], 'Y_hat', '-v7.3');

%%ind
% Load PT ratio
% subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:length(subjects)
%     subject = subjects{i}
%         
%         PT = load([saveroot,'/',  sprintf('ratio/6layers_per_seg/ratio_map_Yhat/PT/%s_PT_ratio.mat', subject)]);
%         PT=PT.PT_layer_array;
% 
%         % Load ET ratio
%         ET = load([saveroot,'/',  sprintf('ratio/6layers_per_seg/ratio_map_Yhat/ET/%s_ET_ratio.mat', subject)]);
%         ET = ET.ET_layer_array;
% 
%         %divide PE
%         PE = PT ./ ET;
%         Y_hat=PE;
% 
%         % P>E=1
%         Y_hat(Y_hat > 1) = 1; 
%         Y_hat(Y_hat < 1) = -1; 
%         
%         %save PE ratio
%         name=sprintf('%s_PE_ratio.mat', subject);
%         save([saveroot,'/',  'ratio','/', '6layers_per_seg', '/', 'ratio_map_Yhat', '/', 'PE', '/', name], 'Y_hat', '-v7.3');
% end
%% ratio_dtseries - individual sub
clc; clear
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab/'));

subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

ciidir = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
ciidir = sprintf([ciidir, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii']);
cii = ciftiopen(ciidir, 'wb_command');
cii.diminfo{1, 1}.models(3:end) = [];
cii.diminfo{1, 1}.length = 18722;
cii.diminfo{1, 2}.length = 1;

ratio_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/ratio/6layers_per_seg/ratio_map_Yhat/';
saveroot='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/ratio/6layers_per_seg/dt_result_Yhat/';

% types={'ET', 'PE', 'PT'};
% types={'ET',  'PT'};
types={'PE'};
 
layer_array={};


for i = 1:length(subjects)
    subject = subjects{i}
    for t = 1:length(types)
        
        type = types{t}
        disp(sprintf('%s_%s_ratio.mat', subject, type))
        
        ratio = load(sprintf('%s%s/%s_%s_ratio.mat', ratio_dir,type, subject, type));
%         ratio = ratio.(sprintf('%s_layer_array', type));
        ratio = ratio.Y_hat;
        
        ratio = ratio';
        
        cii.cdata = ratio;
        
        save_file_name = sprintf([saveroot, '%s_%s_ratio.dtseries.nii'], subject, type)
        ciftisavereset(cii, save_file_name, 'wb_command');
    end
end

%% ratio_dtseries - avg
clc; clear
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab/'));

subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

% ciidir = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
% ciidir = sprintf([ciidir, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii']);
% cii = ciftiopen(ciidir, 'wb_command');
% cii.diminfo{1, 1}.models(3:end) = [];
% cii.diminfo{1, 1}.length = 18722;
% cii.diminfo{1, 2}.length = 1;

ratio_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/ratio/6layers_per_seg/ratio_map_Yhat/';
saveroot='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/ratio/6layers_per_seg/dt_result_Yhat/';

% types={'ET', 'PE', 'PT'};
% types={'ET',  'PT'};
types={'PE'};
 
layer_array={};


X={};
for t = 1:length(types)
    type = types{t}
    
    % Load cifti -10k
    folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
    files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

    file_path = fullfile(folder_path, files.name);
    cii = ciftiopen(file_path, 'wb_command');
    cii = rmfield(cii, 'cdata');

    % Load correlation
%     ratio = load(sprintf('%s%s/avg_%s_ratio.mat', ratio_dir, type, type));
    ratio = load(sprintf('%s%s/%s_ratio.mat', ratio_dir, type, type));
    ratio = ratio.Y_hat;
    
    % Update cdata and diminfo
    cii.cdata = ratio';
    cii.diminfo{1, 1}.length = size(ratio', 1);
    cii.diminfo{1, 2}.length = size(ratio', 2);
    cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal

%     % Concatenate cdata for each feature
%     if isempty(X)
%         X = cii;
%     else
%         X.cdata = cat(2, X.cdata, cii.cdata);
%     end
%     save_file_name = sprintf('%savg_%s_ratio.dtseries.nii',saveroot, type)
    save_file_name = sprintf('%s%s_ratio.dtseries.nii',saveroot, type)
    ciftisavereset(cii, save_file_name, 'wb_command');
%     
%     %initalize X
%     X={};
end

