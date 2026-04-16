%%
clc;clear

% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 

subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

% ciidir = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
% ciidir = sprintf([ciidir, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii']);
% cii = ciftiopen(ciidir, 'wb_command');
% cii.diminfo{1, 1}.models(3:end) = [];
% cii.diminfo{1, 1}.length = 18722;
% cii.diminfo{1, 2}.length = 1;

isc_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/isc/';
Yhat_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/correlations/4layers_32channel/10k/total/';
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/';

% splits = {'3', '5'};
types = {'bonferroni', 'fdr_bh'};
percentiles = {'25', '15'}; % 

for_avg_bonf_25 = zeros(18722,1);
% for_avg_fdr_bh_25 = zeros(18722,1);
for_avg_bonf_15 = zeros(18722,1);
% for_avg_fdr_15 = zeros(18722,1);

features = {'Ahat', 'E'};

for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)

    for layer = 1:3
        bonf_25_sum_values = 0;
        bonf_15_sum_values = 0;
       
        for i = 1:length(subjects)
            subject = subjects{i}
            
            corr = load([dataroot,'/', 'GUMP_test','/', 'correlations','/', '4layers_32channel','/','10k', '/','total', '/', 'intersub_isc_ROI_mask', '/', sprintf('%s_%s%d_ROI.mat', subject, feature, layer)]).masked_corr;
            
            for j = 1
                type = types{j};

                for k = 1:length(percentiles)

                    percentile = percentiles{k};

                    if strcmp(type,'bonferroni') && strcmp(percentile,'25')
                        isc_data = load(sprintf([isc_dir, 'isc_gump_%s_bonferroni_percentile-%s.mat'], subject, percentile)).data.';
                        cdata = corr./isc_data(18722,1);
                        for_avg_bonf_25 = for_avg_bonf_25 + cdata;
                    end


                    if strcmp(type,'bonferroni') && strcmp(percentile,'15')
                        isc_data = load(sprintf([isc_dir, 'isc_gump_%s_bonferroni_percentile-%s.mat'], subject, percentile)).data.';
                        cdata = corr./isc_data(18722,1);
                        for_avg_bonf_15  = for_avg_bonf_15 + cdata;
                    end

                end
            end
        end     
        avg_bonf_25{layer} = for_avg_bonf_25 / length(subjects);
        avg_bonf_15{layer} = for_avg_bonf_15 / length(subjects);
    end
    folder_name = [dataroot, 'OHBM','/', '4layers_32channel','/', 'mat_file', '/', 'yes_intersubject_mask', '/', 'ROI_threshold', '/']
    avg_bonf_25_filename = sprintf('R_bonf_25_%s.mat', feature)
    avg_bonf_15_filename = sprintf('R_bonf_15_%s.mat', feature)

    save([folder_name, avg_bonf_25_filename], 'avg_bonf_25', '-v7.3');
    save([folder_name, avg_bonf_15_filename], 'avg_bonf_15', '-v7.3');

    %initalize
    avg_bonf_25 = {};
    avg_bonf_15 = {};
end



% avg_bonf_25 = for_avg_bonf_25/length(subjects);
% cii.cdata = avg_bonf_25;
% 
% save_file_name = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/OHBM/4layers_32channel/Ahat1_bonferroni_percentile-25.dtseries.nii';
% ciftisavereset(cii, save_file_name, 'wb_command');
% 
% avg_bonf_15 = for_avg_bonf_15/length(subjects);
% cii.cdata = avg_bonf_15;
% 
% save_file_name = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/OHBM/4layers_32channel/Ahat1_bonferroni_percentile-15.dtseries.nii';
% ciftisavereset(cii, save_file_name, 'wb_command');

