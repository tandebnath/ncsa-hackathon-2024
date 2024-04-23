```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

def plot_spectrogram_and_save_rgb(stft_data_ch1, stft_data_ch2, stft_data_ch3, output_folder, filename):
    # Generate spectrograms for each channel
    _, _, Zxx_ch1 = stft_data_ch1
    _, _, Zxx_ch2 = stft_data_ch2
    _, _, Zxx_ch3 = stft_data_ch3
    
    # Convert to magnitude (amplitude) spectrograms
    spectrogram_ch1 = np.abs(Zxx_ch1)
    spectrogram_ch2 = np.abs(Zxx_ch2)
    spectrogram_ch3 = np.abs(Zxx_ch3)
    
    # Normalize each channel to the range [0, 1]
    spectrogram_ch1 /= np.max(spectrogram_ch1)
    spectrogram_ch2 /= np.max(spectrogram_ch2)
    spectrogram_ch3 /= np.max(spectrogram_ch3)
    
    # Stack spectrograms to create an RGB image
    rgb_image = np.stack((spectrogram_ch1, spectrogram_ch2, spectrogram_ch3), axis=-1)
    
    # Ensure the image data is in the range [0, 1] for all channels
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Plot the RGB spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(rgb_image, aspect='auto', origin='lower')
    plt.axis('off')  # No axis for a cleaner look
    
    # Save the RGB spectrogram to the output folder
    output_path = f"{output_folder}/{filename}.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
```