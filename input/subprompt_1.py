```python
import numpy as np
from scipy.signal import stft

def perform_stft(time_series_data, fs=2048, window='hann', nperseg=256, noverlap=None):
    f, t, Zxx = stft(time_series_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx
```