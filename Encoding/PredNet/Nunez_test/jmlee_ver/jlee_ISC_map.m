%%load fMRI avg(seg18) for sub1,2,3

sub1 = load('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject1_fmri/training_fmri.mat').fmri;
sub1_avg = (sub1.data1+sub1.data2) / 2;
sub1_avg = reshape(sub1_avg, size(sub1_avg,1), size(sub1_avg,2)*size(sub1_avg,3));

sub2 = load('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject2_fmri/training_fmri.mat').fmri;
sub2_avg = (sub2.data1+sub2.data2) / 2;
sub2_avg = reshape(sub2_avg, size(sub2_avg,1), size(sub2_avg,2)*size(sub2_avg,3));

sub3 = load('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject3_fmri/training_fmri.mat').fmri;
sub3_avg = (sub3.data1+sub3.data2) / 2;
sub3_avg = reshape(sub3_avg, size(sub3_avg,1), size(sub3_avg,2)*size(sub3_avg,3));

%combine for all subs
combined_data = cat(3, sub1_avg, sub2_avg, sub3_avg);
combined_data = reshape(combined_data, size(combined_data,2), size(combined_data,1), size(combined_data,3));

dim = size(combined_data) %time x voxels x subjects

% 'combined_data' is a Time x Voxels x Subjects matrix
num_voxels = size(combined_data, 2);
num_subjects = size(combined_data, 3);
ISC_map = zeros(num_voxels, 1);

for v = 1:num_voxels
    temp_data = squeeze(combined_data(:, v, :));  % Time x Subjects matrix for voxel v
    corr_matrix = corr(temp_data);       % Correlation matrix between all subjects for voxel v
    ISC_map(v) = mean(corr_matrix(triu(true(num_subjects), 1)));  % Average upper triangular values excluding diagonal
end
mask = ISC_map > 0.25;  % Create a logical mask
