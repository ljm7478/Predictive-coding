# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Define File Paths ---
path_500SUMCIT = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/weights/CNN_like/pt_kitti_ft_Lall_nt150/layer4/history.csv"
path_kitti = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/weights/prednet/pt_kitti_ft_Lall_nt150/layer4/history.csv"

output_filename = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_training_loss_check/pretrain/Finetuned_Titanic_comparison_2versions.png'

# --- 2. Load Data ---
try:
    # Load the CSV files into pandas DataFrames
    df1 = pd.read_csv(path_500SUMCIT)
    df2 = pd.read_csv(path_kitti)
except IOError as e:
    print ("Error: File not found.")
    print ("Details: %s" % e)
    exit()

# --- 3. Pre-process Data ---
# Add an 'epoch' column based on the row index (assuming 1-based indexing)
df1['epoch'] = range(1, len(df1) + 1)
df2['epoch'] = range(1, len(df2) + 1)

# --- 4. Calculate Synchronized Y-Axis Limits ---
# Find the absolute min and max loss values across ALL THREE dataframes
all_losses = pd.concat([
    df1['loss'], df1['val_loss'],
    df2['loss'], df2['val_loss']
])
min_loss = 0
max_loss = all_losses.max()

# Create a tuple for the Y-axis limits, adding 5% padding to the max value
ylim = (min_loss, max_loss * 1.05)

# --- 5. Create Plots ---
# Create a figure with three subplots (1 row, 3 columns)
# Increased figsize width to accommodate 3 plots
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(24, 6))

# --- Plot 1: Intermediate ---
ax1.plot(df1['epoch'], df1['loss'], label='Training Loss')
ax1.plot(df1['epoch'], df1['val_loss'], label='Validation Loss')
ax1.set_title('CNN-like KITTI Finetuned')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(ylim)

# --- Plot 2: Final 2 ---
ax3.plot(df2['epoch'], df2['loss'], label='Training Loss')
ax3.plot(df2['epoch'], df2['val_loss'], label='Validation Loss')
ax3.set_title('Original KITTI Finetune')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True)
ax3.set_ylim(ylim)

# --- 6. Finalize and Save ---
fig.suptitle('Training Log Comparison Original vs CNN-like', fontsize=16)


# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
plt.savefig(output_filename)

print ("Comparison plot saved as '%s'" % output_filename)