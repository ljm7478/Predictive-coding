import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# Load Ahat and E arrays
A_name='Ahat3_tmo'
E_name='E3_tmo'
Ahat = np.load('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction_Error/result/24fps/prednet_CNN_like_ver2/Frequency/{}.npy'.format(A_name)) 
E = np.load('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/Gabor/prednet_CNN_like_model/Prediction_Error/result/24fps/prednet_CNN_like_ver2/Frequency/{}.npy'.format(E_name))

# Take the mean across channels
Ahat_mean = Ahat.mean(axis=-1)  # Shape: (1, 4368, 128, 160)
E_mean = E.mean(axis=-1)        # Shape: (1, 4368, 128, 160)

# 1. Compute similarity metrics
rmse = np.sqrt(((Ahat_mean - E_mean) ** 2).mean())
mae = np.abs(Ahat_mean - E_mean).mean()

# 2. Cosine similarity and Pearson correlation for trend analysis
Ahat_flat = Ahat_mean.flatten()
E_flat = E_mean.flatten()

cos_sim = cosine(Ahat_flat, E_flat)
pearson_corr, _ = pearsonr(Ahat_flat, E_flat)

print("RMSE: {}, MAE: {}".format(rmse, mae))
print("Cosine Similarity: {}, Pearson Correlation: {}".format(cos_sim, pearson_corr))

# 3. Visualize trends over time
plt.figure(figsize=(12, 5))

# Plot mean Ahat and E over time (averaging over spatial dimensions)
plt.subplot(1, 2, 1)
Ahat_time_mean = Ahat_mean.mean(axis=(2, 3))[0]  # Shape: (4368,)
E_time_mean = E_mean.mean(axis=(2, 3))[0]        # Shape: (4368,)

plt.plot(Ahat_time_mean, label='Mean Ahat over time')
plt.plot(E_time_mean, label='Mean E over time')
plt.legend()
plt.title('Mean Ahat vs. Mean E over time')

# Plot the difference (Ahat - E) for the first timestep
plt.subplot(1, 2, 2)
diff = Ahat_mean[0, 0] - E_mean[0, 0]  # First timestep
plt.imshow(diff, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Difference (Ahat - E) for first timestep')
plt.show()
