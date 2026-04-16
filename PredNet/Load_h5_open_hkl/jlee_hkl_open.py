import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

# file_path = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/02_HCP/hkl/X_7T_MOVIE3_CC2_v2.hkl'
# file_path = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/hkl/X_Gump.hkl'
# file_path1 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/hkl/X_seg1.hkl'
# haha='/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction/hkl/X_Frequency.hkl'
# file_path2 = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_ball_animation/hkl/X_random_test.hkl'
# hkl_file = hkl.load(haha)
# hkl_file2 = hkl.load(file_path2)

train_file = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/00_KITTI_pre-train/hkl/X_train.hkl'
val_file = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/00_KITTI_pre-train/hkl/X_val.hkl'

# train_sources = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/hkl/sources_train.hkl'
# val_file = '/local_raid1/03_user/jungmin/02_data/02_Movie/Concat6Movie/hkl/X_train.hkl'
# val_sources ='/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/hkl/sources_val.hkl'

# result = (hkl_file == hkl_file2).all()
# print(result)

with open(train_file, 'r') as file:
    # Load the HKL file
    hkl_file = hkl.load(file)

    num_zeros = np.count_nonzero(hkl_file == 0)
    num_non_zeros = np.count_nonzero(hkl_file != 0)

    print("Number of zeros:", num_zeros)
    print("Number of non-zeros:", num_non_zeros)
    print('finish')

# to check titanic hkl (train and validation) 
# with open(val_file, 'r') as file: 
#     hkl_file = hkl.load(file)

#     num_images = hkl_file.shape[0]

#     random_index = np.random.randint(0, num_images)

#     random_image = hkl_file[random_index]

#     plt.imshow(random_image)
#     plt.title("Random Image (Index: {})".format(random_index))
#     plt.axis('off') 
#     plt.show()

