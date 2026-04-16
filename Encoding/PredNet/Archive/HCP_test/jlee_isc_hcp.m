%% isc - 4layers
clc; clear
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab/'));

subjects = {'100610', '116726', '135124', '156334', '169343', '178243', '191841', '201515', '249947', '380036', '463040', '725751', '818859', '901139', '102311'};

ciidir = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
ciidir = sprintf([ciidir, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii']);
cii = ciftiopen(ciidir, 'wb_command');
cii.diminfo{1, 1}.models(3:end) = [];
cii.diminfo{1, 1}.length = 18722;
cii.diminfo{1, 2}.length = 1;

isc_dir = '/local_raid4/03_user/sunghyoung/01_project/04_noise_ceiling/results/isc/hcp/';
corr_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/correlations/PCA/4layers/total/';
saveroot='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/dt_result/4layers/bonf_25/';

%bonferroni 25% fixed
for_avg_bonf_25 = zeros(18722,1);
 
layer_array={};

for layer= 0:3
    for i = 1:length(subjects)
        subject = subjects{i};
        disp(sprintf('%s_E%d_corr_svd0.90.mat', subject, layer))
        corr = load(sprintf('%s%s_E%d_corr_svd0.90.mat', corr_dir, subject, layer)).correlations;
        
        isc_data = load(sprintf([isc_dir, 'isc_hcp_%s_bonferroni_percentile-25.mat'], subject)).data.';
        cii.cdata = corr./isc_data(18722,1);
        
        save_file_name = sprintf([saveroot, '%s_E%d_bonferroni_25p.dtseries.nii'], subject, layer)
        ciftisavereset(cii, save_file_name, 'wb_command');
        
        for_avg_bonf_25 = for_avg_bonf_25 + cii.cdata;  
    end
    avg_bonf_25 = for_avg_bonf_25/length(subjects);
    
    layer_array{layer+1} = avg_bonf_25;
end
layer_array = cat(2, layer_array{:});
cii.cdata = layer_array;
    
save_file_name = 'E_avg_bonferroni_25p.dtseries.nii'
ciftisavereset(cii, [saveroot, save_file_name], 'wb_command');

%% isc - 6layers
clc; clear
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab/'));

subjects = {'100610', '116726', '135124', '156334', '169343', '178243', '191841', '201515', '249947', '380036', '463040', '725751', '818859', '901139', '102311'};

ciidir = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
ciidir = sprintf([ciidir, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii']);
cii = ciftiopen(ciidir, 'wb_command');
cii.diminfo{1, 1}.models(3:end) = [];
cii.diminfo{1, 1}.length = 18722;
cii.diminfo{1, 2}.length = 1;

isc_dir = '/local_raid4/03_user/sunghyoung/01_project/04_noise_ceiling/results/isc/hcp/';
corr_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/correlations/PCA/6layers/total/';
saveroot='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/dt_result/6layer/bonf_25/total/';

%bonferroni 25% fixed
for_avg_bonf_25 = zeros(18722,1);
 
layer_array={};

for layer= 0:5
    for i = 1:length(subjects)
        subject = subjects{i};
        disp(sprintf('%s_Ahat%d_corr_svd0.90.mat', subject, layer))
        corr = load(sprintf('%s%s_Ahat%d_corr_svd0.90.mat', corr_dir, subject, layer)).correlations;
        
        isc_data = load(sprintf([isc_dir, 'isc_hcp_%s_bonferroni_percentile-25.mat'], subject)).data.';
        cii.cdata = corr./isc_data(18722,1);
        
%         save_file_name = sprintf([saveroot, '%s_Ahat%d_bonferroni_25p.dtseries.nii'], subject, layer)
%         ciftisavereset(cii, save_file_name, 'wb_command');
        
        for_avg_bonf_25 = for_avg_bonf_25 + cii.cdata;  
    end
    avg_bonf_25 = for_avg_bonf_25/length(subjects);
    
    layer_array{layer+1} = avg_bonf_25;
end
layer_array = cat(2, layer_array{:});
cii.cdata = layer_array;
    
save_file_name = 'Ahat_avg_bonferroni_25p.dtseries.nii'
ciftisavereset(cii, [saveroot, save_file_name], 'wb_command');
