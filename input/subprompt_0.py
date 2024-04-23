```python
import numpy as np
from scipy.signal import butter, sosfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfilt(sos, data, axis=1)
    return filtered_data

# Example usage:
# fs = 2048  # Sampling rate
# lowcut = 25.0  # Low cut frequency in Hz
# highcut = 500.0  # High cut frequency in Hz
# data = np.random.randn(3, n)  # Replace with your 3xn data array
# filtered_signal = bandpass_filter(data, lowcut, highcut, fs)
```