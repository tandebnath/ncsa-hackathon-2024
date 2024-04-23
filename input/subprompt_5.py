```python
import numpy as np
from scipy.signal import correlate

def compute_cross_correlation(time_series_data, labels):
    # Filter data with label 1
    data_with_label_1 = time_series_data[labels == 1]

    num_channels = data_with_label_1.shape[1]
    time_lags = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                correlation = correlate(data_with_label_1[:, i], data_with_label_1[:, j], mode='full')
                lag = np.argmax(correlation) - (len(data_with_label_1) - 1)
                time_lags[i, j] = lag

    return time_lags
```