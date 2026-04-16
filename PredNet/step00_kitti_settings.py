# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.

# #AlexNet from WenH Encoding model dir 
# DATA_DIR = './data/03_AlexNet_WenH_Encoding/current_use_check_230406/'
# HICKLE_DUMP = './data/03_AlexNet_WenH_Encoding/current_use_check_230406/hkl/'
# WEIGHTS_DIR = './data/03_AlexNet_WenH_Encoding/current_use_check_230406/weight/'

# GUMP 
# GUMP_HICKLE_DUMP = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/hkl'
GUMP_HICKLE_DUMP = './data/01_gump/hkl'
GUMP_DATA_DIR = '/combinelab2/03_user/jungmin/02_data/01_Gump/02_movie/segment/frames'

# #HCP 
# HCP_RESULT_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/6layers/'
# HCP_HICKLE_DUMP = './data/02_HCP/hkl/'
# HCP_DATA_DIR ='/combinelab/03_user/jungmin/02_data/02_HCP/'

#Titanic for Train ONLY
Titanic_WEIGHTS_DIR = './data/02_Titanic/weights/'
Titanic_HICKLE_DUMP = './data/02_Titanic/hkl/'
# Titanic_DATA_DIR = '/combinelab/03_user/jungmin/02_data/03_Titanic/frames_for_make_hkl/'
# Titanic_RESULTS_DIR_HCP = './data/02_Titanic/results/HCP/'
# Titanic_RESULTS_DIR_GUMP = './data/02_Titanic/result/GUMP/'

# #Movie 6concat (500sum, citizenfour, pulpfiction, littlemisssunshine, usualsuspect, DP)
# Concat_DATA_DIR = '/local_raid1/03_user/jungmin/02_data/02_Movie/Concat6Movie'
# Concat_HICKLE_DUMP = './data/06_NNDB/concat6movies/hkl/'
# Concat_WEIGHTS_DIR ='./data/06_NNDB/concat6movies/weights/'

#500SUM+Citizenfour
Concat_DATA_DIR = '/local_raid1/03_user/jungmin/02_data/02_Movie/NNDB/06_concat_500SUM_CIT/'
Concat_HICKLE_DUMP = './data/06_NNDB/CIT_500SUM/hkl/'
Concat_WEIGHTS_DIR ='./data/06_NNDB/CIT_500SUM/weights/'

#simple ball animation stimuli created by jlee
# ball_DATA_DIR = './data/03_ball_animation/stimulus/frames/'
# ball_HICKLE_DUMP='./data/03_ball_animation/hkl/'
# ball_WEIGHTS_DIR='./data/03_ball_animation/weights/'

## AlexNet stimuli
# AlexNet_DATA_DIR = '/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/stimuli/video_fmri_dataset/test/'
# AlexNet_HICKLE_DUMP='./data/04_AlexNet/hkl/'
# AlexNet_WEIGHTS_DIR='./data/04_AlexNet/weights/'

##jmlee_stimulus
# Jlee_DATA_DIR = '/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/01_stimuli/PredNetExp'
# Jlee_HICKLE_DUMP = './data/05_jmlee_stimuli/PredNetExp'
# Jlee_WEIGHTS_DIR = './data/05_jmlee_stimuli/PredNetExp'

# ##jmlee_stimulus - naturalistic
# Naturalitic_DATA_DIR = '/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/01_stimuli/Naturalistic/'
# N_Prediction_Error_HICKLE_DUMP = './data/05_jmlee_stimuli/Naturalistic/prednet_CNN_like_model/Prediction_Error/hkl/'
# N_Prediction_Error_WEIGHTS_DIR = './data/05_jmlee_stimuli/Naturalistic/prednet_CNN_like_model/Prediction_Error/weights'
# N_Prediction_HICKLE_DUMP = './data/05_jmlee_stimuli/Naturalistic/prednet_CNN_like_model/Prediction/hkl/'
# N_Prediction_WEIGHTS_DIR='./data/05_jmlee_stimuli/Naturalistic/prednet_CNN_like_model/Prediction/weights/'

# ##jmlee_stimulus - gabor
# Gabor_DATA_DIR = '/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/01_stimuli/Gabor/'
# G_Prediction_Error_HICKLE_DUMP = './data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction_Error/hkl/'
# G_Prediction_Error_WEIGHTS_DIR = './data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction_Error/weights'
# G_Prediction_HICKLE_DUMP = './data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction/hkl/'
# G_Prediction_WEIGHTS_DIR='./data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction/weights/'

##Cam-CAN_stimulus 
CC_DATA_DIR = '/combinelab2/03_user/jungmin/02_data/'
CC_HICKLE_DUMP = './data/07_CamCAN/hkl/'

##Budapest_stimulus 
BP_DATA_DIR = '/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_video'
BP_HICKLE_DUMP = './data/08_Budapest/hkl/'