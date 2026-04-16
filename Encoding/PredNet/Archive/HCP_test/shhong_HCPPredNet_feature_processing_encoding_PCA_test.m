%% Step2: PCA - E 
clc;clear

%Titanic trained
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/4layers';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test';

addpath(genpath('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/code/'));

%500SUM trained
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_HCP/result/h5';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';
% % % % % % % % % % SVD-updating algorithm % % % % % % % % %

Niter = 2; % number of iteration to compute principle component
percps = {0.90, 0.92, 0.94, 0.96, 0.99}; % explain 99% of the variance of every movie segments % 

features = {'Ahat','E'}; %  


for i = 2:3
    disp(['layer is ', num2str(i)])

    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        k0 = 0;

        %load lay_feat
        disp('load regularized lay_feat')

        secpath = [saveroot,'/', 'feature_maps', '/', sprintf('%s%d_1-4_concatenated_feature.h5', feature, i)]
        lay_feat = h5read(secpath,'/data');
        dim = size(lay_feat)

        for k = 1 : length(percps)

            percp = percps{k};

            for iter =  1  : Niter
                disp(['iteration #: ', num2str(iter)])
                disp('amri_sig_isvd start')

                if iter == 1
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
                else
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
                end  

                disp(['Iteration ', num2str(iter), ' completed for', feature, num2str(i)]);
            end

            disp('diag(s) begins')

            s = diag(S);

            disp('start saving SVD_layer');
            % save principal components
            save([saveroot, '/PCA_test/X/', sprintf('%s%d_feature_maps_svd_%s.mat', feature, i, num2str(percp))], 'B', 's', '-v7.3');
            disp(['SVD_layer for', feature, ' saved']);

%             clear B s S lay_feat
        end
    end
end
