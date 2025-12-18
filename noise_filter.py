import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 1000  # sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second duration

# Signal: 5 Hz + 50 Hz
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

plt.plot(t, x)
plt.title("Original Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

from scipy.signal import butter, filtfilt

cutoff = 10      # cutoff frequency in Hz
order = 4        # filter order

b, a = butter(order, cutoff / (fs / 2), btype='low')

y = filtfilt(b, a, x)

plt.figure()
plt.plot(t, x, label='Original')
plt.plot(t, y, label='Filtered')
plt.legend()
plt.title("Filtered vs Original")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()
# FFT of original and filtered signal

# FFT computation
X = np.fft.fft(x)
Y = np.fft.fft(y)

# Frequency axis
freqs = np.fft.fftfreq(len(x), 1/fs)

# Take only positive frequencies
idx = freqs >= 0
freqs_pos = freqs[idx]

# Magnitude spectra
X_mag = np.abs(X[idx])
Y_mag = np.abs(Y[idx])

# Plot FFT of original signal
plt.figure()
plt.stem(freqs_pos, X_mag)
plt.title("Frequency Spectrum of Original Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 100)
plt.grid(True)

# Plot FFT of filtered signal
plt.figure()
plt.stem(freqs_pos, Y_mag)
plt.title("Frequency Spectrum of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 100)
plt.grid(True)

plt.show()