import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Define File Paths ---
# Path to your single history.csv file
log_path = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/weights/CNN_like/pt_kitti_ft_Lall_nt150/layer5_nopool/history.csv"
output_filename = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_training_loss_check/finetune/pt_kitti_ft_CNN-like_5layers_nopool.png'
# Optional: Title for the plot
plot_title = '500SUMCIT finetuning (Fixed  Hyperparameter Ver)'

# --- 2. Load Data ---
if not os.path.exists(log_path):
    print("Error: File not found at {}".format(log_path))
    exit()

try:
    df = pd.read_csv(log_path)
    print("Successfully loaded log file.")
except Exception as e:
    print("Error reading CSV file: {}".format(e))
    exit()

# --- 3. Pre-process Data ---
# Add an 'epoch' column based on row index (1-based)
df['epoch'] = range(1, len(df) + 1)

# --- 4. Plotting ---
plt.figure(figsize=(10, 6))

# Plot Training Loss
plt.plot(df['epoch'], df['loss'], label='Training Loss (loss)', linewidth=2)

# Plot Validation Loss (if available)
if 'val_loss' in df.columns:
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss (val_loss)', linewidth=2)

plt.title(plot_title, fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MAE)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Optional: Set Y-axis limit to focus on the convergence (e.g., ignore initial spike)
# plt.ylim(0, 0.1) 

# Save the plot
output_dir = os.path.dirname(output_filename)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print("Plot saved to: {}".format(output_filename))