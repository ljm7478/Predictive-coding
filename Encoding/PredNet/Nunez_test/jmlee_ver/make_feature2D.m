%% make feaure2D
clc;clear;
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/h5/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Nunez_test/';

layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 
for lay = 1 : length(layername)
    lay_feat_concatenated=[];
    for seg = 1 : 18
        disp(['Seg: ', num2str(seg)]);
        secpath = [dataroot,'seg', num2str(seg), '_Ahat', '.h5']
        lay_feat = h5read(secpath,[layername{lay}]);
        dim = size(lay_feat);
        Nf = dim(end); % number of frames
        if seg == 1
           lay_feat_concatenated = zeros([dim(1:end-1),Nf*18],'single'); 
        end
        lay_feat_concatenated(:,:,:,(seg-1)*Nf+1:seg*Nf) = lay_feat;
    end
    dim = size(lay_feat_concatenated);
    Nu = prod(dim(1:end-1));% number of units
    Nf = dim(end); % number of time points
    lay_feat_concatenated = reshape(lay_feat_concatenated, Nu, Nf)';
    dim = size(lay_feat_concatenated)
    
    save(fullfile(saveroot, [layername{lay}(2:end),'.mat']),...
        'lay_feat_concatenated', '-v7.3');
    disp(['Saved concatenated features for layer ', num2str(lay)]);
end