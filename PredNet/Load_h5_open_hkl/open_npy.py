import numpy as np 
import os

base_dir = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/pt_kitti_ft_Lall_nt150/layer4/1frame_prediction/run1"

data1=np.load(os.path.join(base_dir, 'Ahat1.npy'))
print(data.shape)