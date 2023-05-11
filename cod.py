import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.fft import fft, fftfreq
import pandas as pd
import binascii

# criação de seno
def cont_sin(time, sample_rate, frequency):
    time = time
    sample = sample_rate
    f = frequency
    t = np.linspace(time, time+0.1, sample)
    signal = np.sin(2*np.pi*f*t)
    return t, signal


# -------------------MODULAÇÃO-------------------

# padrões de frequencia
Fs = 10000
T = 0
fc1 = 1300  # 1
fc2 = 1700  # 0

# construindo mensagem
ascii_message = 'hello comp'
bin_message = ''.join(format(ord(i), '08b') for i in ascii_message)
print(bin_message)
len_message = len(ascii_message)

# preenche vetor freq_bin_message com a frequencia de cada elemento
freq_bin_message = np.zeros(len(bin_message))
for i in range(len(bin_message)):
    if bin_message[i] == '1':
        freq_bin_message[i] = fc1
    else:
        freq_bin_message[i] = fc2

# cria sinal modulado em frequência
signal = np.zeros(0)
t = np.zeros(0)
for i in range(len(freq_bin_message)):
    time, sin_signal = cont_sin(T, Fs, freq_bin_message[i])
    signal = np.hstack([signal, sin_signal])
    t = np.hstack([t, time + T])
    T += 0.1

plt.plot(t, signal)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title(r'Plot of CT signal')
plt.xlim([0, 0.001])
plt.show()

# espectro do sinal
T = t[1] - t[0] # calcular o período do sinal 0.001 -> 1/T = 1000
N = signal.size

f = fftfreq(len(signal), T)
frequencias = f[:N // 2]
amplitudes = np.abs(fft(signal))[:N // 2] * 1 / N

print("Value in index ", np.argmax(amplitudes), " is %.2f" % amplitudes[np.argmax(amplitudes)])
print("Freq: ", frequencias[np.argmax(amplitudes)])
plt.plot(frequencias, amplitudes)
plt.grid()
plt.xlim([1000, 2000])
plt.show()


