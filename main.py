import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
import math

N = 8

t = np.linspace(0,6,20*1024) #  time axis



third_octave = np.array([174.61,220,174.61,164.81,174.61,220,164.81,146.83])
fourth_octave = np.zeros(N)



tis = np.array([0, 0.4, 0.8, 1.2, 3.5, 3.7, 4.2, 4.5])
Tis = np.array([0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 2.0])

 


x = 0

for i in range(N):
    x += np.reshape([(t >= tis[i]) & (t < tis[i] + Tis[i]) ],t.shape) * (np.sin( 2 * np.pi * third_octave[i] * t ) + np.sin( 2 * np.pi * fourth_octave[i] * t ))



   
N = 6*1024
f = np. linspace(0 , 512 , int(N/2)) # frequency axis


# fourier transform of x

x_f = fft(x)
x_f = 2/N * np.abs(x_f [0:np.int(N/2)])


# adding noise

fn1 , fn2 = np. random. randint(0, 512, 2)
n = np.sin(2*np.pi*fn1*t)+np.sin(2*np.pi*fn2*t)


xn = x+n

# fourier transform of signal with added noise

xn_f = fft(xn)
xn_f = 2/N * np.abs(xn_f [0:np.int(N/2)])


# noise cancellation

max_indicies = np.where(xn_f>math.ceil(np.max(x_f)))
fi1 = max_indicies[0][0] # index of first frequency
fi2 = max_indicies[0][1] # index of second frequency


fn1_found = int(f[fi1])
fn2_found = int(f[fi2])


x_filtered = xn - (np.sin(2*np.pi*fn1_found*t)+np.sin(2*np.pi*fn2_found*t))


sd.play(x_filtered, 9*1024)

x_filtered_f = fft(x_filtered)
x_filtered_f = 2/N * np.abs(x_filtered_f [0:np.int(N/2)])


# Plotting all graphs

plt.figure()

plt.subplot(3,1,1)
plt.title('Original Signal (Time Domain)')
plt.plot(t,x) # oiginal signal in time domain

plt.subplot(3,1,2)
plt.title('Noisy Signal (Time Domain)')
plt.plot(t,xn) # noisy signal in time domain

plt.subplot(3,1,3)
plt.title('Filtered Signal (Time Domain)')
plt.plot(t,x_filtered) # filtered signal in time domain


plt.figure()
plt.subplot(3,1,1) 
plt.title('Original Signal (Frequency Domain)')
plt.plot(f,x_f) # original signal in frequency domain


plt.subplot(3,1,2)
plt.title('Noisy Signal (Frequency Domain)')
plt.plot(f,xn_f) # noisy signal in frequency domain

plt.subplot(3,1,3)
plt.title('Filtered Signal (Frequency Domain)')
plt.plot(f,x_filtered_f) # filtered signal in time domain

