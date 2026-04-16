addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab/'));

saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';
subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
features = {'Ahat', 'E'}

% Load cifti -10k
folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

file_path = fullfile(folder_path, files.name);
cii = ciftiopen(file_path, 'wb_command');
cii = rmfield(cii, 'cdata');

cii.diminfo{1, 1}.length = 18722;
cii.diminfo{1, 2}.length = 4;
cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal

lambda1 = [0.1:0.2:0.9];
lambda2 = logspace(1, 20, 20);
lambda3 = logspace(1, 7, 7);
lambda4 = logspace(1, 3, 20);
lambda5 = logspace(1, 4, 20);
lambdas = {lambda1, lambda2, lambda3, lambda4, lambda5};

X={} ;
for i = 1:15
    subject = subjectIDs(i)
   
    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for seg = 1:4
            for j = 2:length(lambdas)
                for layer=0:3

                    if isfile(sprintf([saveroot,'/lambda/W/PCA/lambda%s/seg%s/%s_%s%s_lambda_svd0.9_s3.mat'],num2str(j), num2str(seg), num2str(subject), feature, num2str(layer)))
                        
                        %load correaltion
                        load(sprintf([saveroot,'/lambda/W/PCA/lambda%s/seg%s/%s_%s%s_lambda_svd0.9_s3.mat'],num2str(j), num2str(seg), num2str(subject), feature, num2str(layer)));

                        % Update cdata and diminfo
                        cii.cdata = log10(Lambda.');
                        
                        % Save as dtseries
                        save_file_name = sprintf(['%s_%s%s_lambda.dtseries.nii'], num2str(subject), feature, num2str(layer));
                        ciftisavereset(cii, fullfile(sprintf([saveroot, '/lambda/W/PCA/lambda%s/seg%s/', save_file_name], num2str(j), num2str(seg))), 'wb_command');
                        
                    end
                end
            end
        end 
    end
end