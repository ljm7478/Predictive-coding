%%
clc; clear

dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/4layers_32c_input_change';

features = {'run1_E','run2_E','run3_E', 'run4_E', 'run5_E', 'run6_E', 'run7_E', 'run8_E'};

for lay = 1:3
    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        feature_name = feature(6);
        disp(feature_name)
        
        seg_name = feature(1:4);
        disp(seg_name)
        
        %load lay_feat
        disp(sprintf('load %s%d lay_feat for avg', feature, lay))

        secpath = [dataroot, '/', 'h5', '/', seg_name, '/', sprintf('%s%d.h5', feature_name, lay)]
        lay_feat = h5read(secpath,'/predimg');

        half_channel = size(lay_feat,1);
        half_channel = half_channel/2;

        lay_feat_1st_half= lay_feat(1:half_channel,:,:,:);
        lay_feat_2nd_half = lay_feat(half_channel+1:end,:,:,:);

        lay_feat = lay_feat_1st_half+lay_feat_2nd_half;
        dim = size(lay_feat)

        h5create([dataroot,  '/', 'h5', '/', seg_name, '/', sprintf('%s%d_3c.h5', feature_name, lay)],'/predimg',...
            [size(lay_feat)],'Datatype','single');
        h5write([dataroot,  '/', 'h5', '/', seg_name, '/', sprintf('%s%d_3c.h5', feature_name, lay)],'/predimg', lay_feat);
    end
end