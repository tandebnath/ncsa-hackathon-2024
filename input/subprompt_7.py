```python
import numpy as np
from scipy.fft import fft, fftfreq

def compute_energy_density(signal, sampling_rate):
    # Perform Fourier Transform
    signal_fft = fft(signal)
    frequencies = fftfreq(len(signal), d=1/sampling_rate)
    
    # Compute the power spectral density
    power_spectral_density = np.abs(signal_fft)**2
    
    # Integrate the power spectral density over the frequency band
    energy_density = np.trapz(power_spectral_density, frequencies)
    
    return energy_density
```