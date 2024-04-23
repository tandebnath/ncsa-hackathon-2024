```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import detrend

def animate_phase_space(time_series_data, output_folder, fps=30, duration=5):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    num_frames = fps * duration

    def update(frame):
        for i in range(3):
            axs[i].clear()
            signal = time_series_data[:, i]
            time_derivative = np.gradient(detrend(signal))
            axs[i].plot(signal[:frame], time_derivative[:frame])
            axs[i].set_title(f'Channel {i+1}')
            axs[i].set_xlabel('Signal')
            axs[i].set_ylabel('Time Derivative')
        return axs

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False, repeat=False)
    output_path = f"{output_folder}/phase_space_animation.mp4"
    ani.save(output_path, writer='ffmpeg', fps=fps)

# Example usage:
# time_series_data = np.random.rand(1000, 3)  # Replace with actual 3-channel time series data
# output_folder = 'path/to/output/folder'
# animate_phase_space(time_series_data, output_folder)
```