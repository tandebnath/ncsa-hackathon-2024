```python
import matplotlib.pyplot as plt
import os

def plot_histogram(energy_densities, labels, output_folder):
    # Separate energy densities by label
    energy_densities_label_0 = [energy for energy, label in zip(energy_densities, labels) if label == 0]
    energy_densities_label_1 = [energy for energy, label in zip(energy_densities, labels) if label == 1]

    # Plot histogram
    plt.hist(energy_densities_label_0, bins=50, alpha=0.5, label='Label 0')
    plt.hist(energy_densities_label_1, bins=50, alpha=0.5, label='Label 1')
    plt.xlabel('Energy Density')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save histogram to output folder
    output_path = os.path.join(output_folder, 'energy_density_histogram.png')
    plt.savefig(output_path)
    plt.close()
```