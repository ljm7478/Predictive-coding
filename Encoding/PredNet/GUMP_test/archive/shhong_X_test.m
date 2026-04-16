Ahat1_seg1_3 = load('Ahat1_feature_maps_seg1-3_svd_0.90_ver2.mat');
Ahat1_seg3_1 = load('Ahat1_feature_maps_seg3-1_svd_0.90_ver2.mat');
Ahat2_seg1_3 = load('Ahat2_feature_maps_seg1-3_svd_0.90_ver2.mat');
Ahat2_seg3_1 = load('Ahat2_feature_maps_seg3-1_svd_0.90_ver2.mat');

secpath1 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers_32channel/h5/run1_Ahat1.h5';
run1_Ahat1 = h5read(secpath1,'/predimg');

secpath2 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers_32channel/h5/run2_Ahat1.h5';
run2_Ahat1 = h5read(secpath2,'/predimg');

secpath3 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers_32channel/h5/run3_Ahat1.h5';
run3_Ahat1 = h5read(secpath3,'/predimg');

lay_feat13_1 = cat(4, run1_Ahat1, run2_Ahat1, run3_Ahat1);
lay_feat31_1 = cat(4, run3_Ahat1, run2_Ahat1, run1_Ahat1);

Ahat1_mean_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/tmp/Ahat1_seg1-3_feature_maps_avg.h5';
Ahat1_mean = h5read(Ahat1_mean_path, '/data');
Ahat1_std_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/tmp/Ahat1_seg1-3_feature_maps_std.h5';
Ahat1_std = h5read(Ahat1_std_path, '/data');

lay_feat13_1 = bsxfun(@minus, lay_feat13_1, Ahat1_mean);
lay_feat13_1 = bsxfun(@rdivide, lay_feat13_1, Ahat1_std);
dim13_1 = size(lay_feat13_1);
lay_feat13_1 = reshape(lay_feat13_1, prod(dim13_1(1:end-1)),dim13_1(end));

lay_feat31_1 = bsxfun(@minus, lay_feat31_1, Ahat1_mean);
lay_feat31_1 = bsxfun(@rdivide, lay_feat31_1, Ahat1_std);
dim31_1 = size(lay_feat31_1);
lay_feat31_1 = reshape(lay_feat31_1, prod(dim31_1(1:end-1)),dim31_1(end));

X_Ahat1_seg1_3 = lay_feat13_1'*Ahat1_seg1_3.B/sqrt(size(Ahat1_seg1_3.B,1));
X_Ahat1_seg3_1 = lay_feat31_1'*Ahat1_seg3_1.B/sqrt(size(Ahat1_seg3_1.B,1));

% ----------------------------------------------------------------------------

secpath4 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers_32channel/h5/run1_Ahat2.h5';
run1_Ahat2 = h5read(secpath4,'/predimg');

secpath5 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers_32channel/h5/run2_Ahat2.h5';
run2_Ahat2 = h5read(secpath5,'/predimg');

secpath6 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers_32channel/h5/run3_Ahat2.h5';
run3_Ahat2 = h5read(secpath6,'/predimg');

lay_feat13_2 = cat(4, run1_Ahat2, run2_Ahat2, run3_Ahat2);
lay_feat31_2 = cat(4, run3_Ahat2, run2_Ahat2, run1_Ahat2);

Ahat2_mean_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/tmp/Ahat2_seg1-3_feature_maps_avg.h5';
Ahat2_mean = h5read(Ahat2_mean_path, '/data');
Ahat2_std_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/tmp/Ahat2_seg1-3_feature_maps_std.h5';
Ahat2_std = h5read(Ahat2_std_path, '/data');

lay_feat13_2 = bsxfun(@minus, lay_feat13_2, Ahat2_mean);
lay_feat13_2 = bsxfun(@rdivide, lay_feat13_2, Ahat2_std);
dim13_2 = size(lay_feat13_2);
lay_feat13_2 = reshape(lay_feat13_2, prod(dim13_2(1:end-1)),dim13_2(end));

lay_feat31_2 = bsxfun(@minus, lay_feat31_2, Ahat2_mean);
lay_feat31_2 = bsxfun(@rdivide, lay_feat31_2, Ahat2_std);
dim31_2 = size(lay_feat31_2);
lay_feat31_2 = reshape(lay_feat31_2, prod(dim31_2(1:end-1)),dim31_2(end));

X_Ahat2_seg1_3 = lay_feat13_2'*Ahat2_seg1_3.B/sqrt(size(Ahat2_seg1_3.B,1));
X_Ahat2_seg3_1 = lay_feat31_2'*Ahat2_seg3_1.B/sqrt(size(Ahat2_seg3_1.B,1));
