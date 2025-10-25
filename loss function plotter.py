import re
import os
import matplotlib.pyplot as plt

# Path to your slurm.out file
slurm_out_file = "slurm-3233.out"

# Specify the steps you want to plot
steps_to_plot = 17

# Destination directory for saving plots
dest_dir = "plots"
os.makedirs(dest_dir, exist_ok=True)

# Regular expression to extract the running_time
time_pattern = re.compile(r"running_time='(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'") 

# Read the slurm.out file
with open(slurm_out_file, 'r') as f:
    log_data = f.read()

# Extract running_time
match = time_pattern.search(log_data)
if match:
    running_time = match.group(1)
else:
    raise ValueError("Could not find running_time in the log file.")

# Create sub-directory for plots based on running_time
plot_subdir = os.path.join(dest_dir, running_time)
os.makedirs(plot_subdir, exist_ok=True)
plot_sub_subdir = os.path.join(plot_subdir, str(steps_to_plot))
os.makedirs(plot_sub_subdir, exist_ok=True)

# Regular expression to extract the relevant data
pattern = re.compile(r"Step: (\d+); Epoch: (\d+);.*?Kd_loss: (-?[\d\.]+);.*?Pd_loss: (-?[\d\.]+);.*?Rd_loss: (-?[\d\.]+);.*?Conf_loss: (-?[\d\.]+);.*?Div_loss: (-?[\d\.]+);.*?ide_loss: (-?[\d\.]+);.*?triplet_loss: (-?[\d\.]+);.*?angular_loss: (-?[\d\.]+);")

# Extract the data
steps, epochs, kd_loss, pd_loss, rd_loss, conf_loss, div_loss, ide_loss, triplet_loss, angular_loss, total_loss = [], [], [], [], [], [], [], [], [], [], []

for match in pattern.finditer(log_data):
    step = int(match.group(1))
    if step == steps_to_plot and match.group(2) != 'Test':
        steps.append(step)
        epochs.append(int(match.group(2)))
        kd_loss.append(float(match.group(3)))
        pd_loss.append(float(match.group(4)))
        rd_loss.append(float(match.group(5)))
        conf_loss.append(float(match.group(6)))
        div_loss.append(float(match.group(7)))
        ide_loss.append(float(match.group(8)))
        triplet_loss.append(float(match.group(9)))
        angular_loss.append(float(match.group(10)))
        temp_var = float(match.group(3)) + float(match.group(4)) + float(match.group(5)) + float(match.group(6)) + float(match.group(7)) + float(match.group(8)) + float(match.group(9)) + float(match.group(10))
        total_loss.append(temp_var)

# Plot and save the data for each loss function
losses = {
    'Kd_loss': kd_loss,
    'Pd_loss': pd_loss,
    'Rd_loss': rd_loss,
    'Conf_loss': conf_loss,
    'Div_loss': div_loss,
    'ide_loss': ide_loss,
    'triplet_loss': triplet_loss,
    'angular_loss': angular_loss,
    'total_loss': total_loss
}

for loss_name, loss_values in losses.items():
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, label=loss_name)
    plt.xlabel('Epoch')
    plt.ylabel(loss_name)
    plt.title(f'{loss_name} Over Epochs for Specified Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_sub_subdir, f'{loss_name}.png'))
    plt.close()

print(f"Plots saved to {plot_sub_subdir}")
