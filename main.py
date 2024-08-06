import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def calculate_frames(signal, frame_size, overlap):
    step_size = frame_size - overlap
    num_frames = 1 + (len(signal) - frame_size) // step_size
    frames = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        start_idx = i * step_size
        frames[i, :] = signal[start_idx:start_idx + frame_size]

    return frames

def compute_energy(frames):
    return np.sum(frames ** 2, axis=1)

def compute_zero_crossings(frames):
    zero_crossings = np.sum(np.diff(np.sign(frames), axis=1) != 0, axis=1)
    return zero_crossings

def plot_results(signal, frames, energy, zero_crossings, frame_size):
    time_axis = np.linspace(0, len(signal) / fs, num=len(signal))
    frames_1d = frames.flatten()
    frames_time_axis = np.linspace(0, len(frames_1d) / fs, num=len(frames_1d))
    energy_time_axis = np.linspace(0, len(energy) * frame_size / fs, num=len(energy))
    zero_crossings_time_axis = np.linspace(0, len(zero_crossings) * frame_size / fs, num=len(zero_crossings))

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time_axis, signal)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(frames_time_axis, frames_1d)
    plt.title('Frames Matrix in 1D')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 3)
    plt.plot(energy_time_axis, energy)
    plt.title('Energy Vector')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy')

    plt.subplot(4, 1, 4)
    plt.plot(zero_crossings_time_axis, zero_crossings)
    plt.title('Zero Crossing Vector')
    plt.xlabel('Time [s]')
    plt.ylabel('Zero Crossings')

    plt.tight_layout()
    plt.show()

# Load the input speech signal
filename = 'Test.wav'  # Replace with your file path
fs, signal = wavfile.read(filename)

# Parameters
frame_size = 256  # Adjust frame size as needed
overlap = frame_size // 2  # 50% overlap

# Ensure the signal is mono
if len(signal.shape) > 1:
    signal = signal[:, 0]

# Compute frames
frames = calculate_frames(signal, frame_size, overlap)
print("Frames Matrix:")
print(frames)

# Compute energy and zero crossings
energy = compute_energy(frames)
print("Energy Vector:")
print(energy)

zero_crossings = compute_zero_crossings(frames)
print("Zero Crossings Vector:")
print(zero_crossings)

# Plot the results
plot_results(signal, frames, energy, zero_crossings, frame_size)