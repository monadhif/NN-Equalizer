"""
The idea is to find the Rotation transform using as input the rotated samples(200 samples with 2 
columns(I,Q) float values) and as output the normal ones 
The problem is the input shape, i put it input_dim = (2,) because my input is a dataset of 8192 rows 
and 2 columns but i got an error
"""
import numpy as np
import numcomm #chez orange 
import itertools
import matplotlib.pyplot as plt
from scipy import signal
from math import *
from numpy.random import rand, randn
from scipy import special
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.linalg import toeplitz
from datetime import datetime
from packaging import version







QAM_ORDER = 16
N_BYTES = 4096
# h = np.array([0.2,0.9,0.3]) #impulse response of the channel
h = np.array(
    [0.2 + 1j * 0.1, 0.9 + 1j * 0.1, 0.3 + 1j * 0.1]
)  # impulse response of the channel #,0.1,0.2
L = h.size
K = 1
QAM_modulator = numcomm.modems.QAM(order=QAM_ORDER)
# We generate integer between 0 & 255 = One byte or 8 bits
bits = np.array(np.random.randint(0, 255, N_BYTES), dtype=np.uint8)  # As integer
bits_representation = numcomm.tools.bytestobits(bits, 1)  # As bits
x = QAM_modulator.modulate(bits)
N = len(x)
# (noisy_symbols, real_SNR_dB) = AWGN_noise(samples=symbols,target_SNR_dB=20)
x_with_zeros = np.concatenate((np.zeros(L - 1), x, np.zeros(L - 1)))
x_matrix = []
for i in range(N + L - 1):
    x_matrix.append(x_with_zeros[i : i + L])
x_matrix = np.array(x_matrix)
chanOut = np.convolve(x, h)
m = len(chanOut)
(yn, real_SNR_dB, noise_power) = AWGN_noise(samples=chanOut, target_SNR_dB=20)
########################################################################################

hAutoCorr = np.convolve(h, np.flip(h))
L_m = len(hAutoCorr)
m = L_m // 2
print(m, "    ", L_m)
h1_m = np.concatenate((hAutoCorr[m:L_m], np.zeros(2 * K + 1 - L_m + m)))
h2_m = np.concatenate((hAutoCorr[::-1][m:L_m], np.zeros(2 * K + 1 - L_m + m)))
hM1 = toeplitz(h1_m, h2_m)
print(hM1.shape)
hM = hM1 + noise_power * np.eye(2 * K + 1)
d_m = np.zeros(2 * K + 1)
print(d_m)
d_m[K - 1 : K + L - 1] = np.flip(h)
c_mmse = np.matmul(np.linalg.inv(hM), d_m.T)
print(hM1)
#################################################################################
# array([-0.24677222,  1.23519444, -0.35920783])
yFilt_mmse1 = np.convolve(yn, c_mmse)  # chanOut
# yFilt_mmse = yFilt_mmse[K+3:len(yFilt_mmse)]
print(yFilt_mmse.shape)
yFilt_mmse = np.convolve(yFilt_mmse1, np.ones(1))  # convolution
print(yFilt_mmse == yFilt_mmse1)
ySamp_mmse = np.array(yFilt_mmse[0:N], dtype=np.complex64)
print(ySamp_mmse.shape)
initial_bits = QAM_modulator.demodulate(ySamp_mmse)  # demodulate x
print(initial_bits.shape)
print("x.shape: ", x.shape)
###########################################################################
def new_model(m):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                input_shape=(m, 1, 2),
                filters=2,
                kernel_size=(3, 1),
                padding="valid",
                activation="linear",
            ),
        ]
    )
    model.compile(
        optimizer="sgd", loss="mean_squared_error", metrics=[]
    )  # optimizer='adam',#'mean_squared_error'

    return model


x_n = np.array(
    [[np.real(symbol), np.imag(symbol)] for symbol in x], dtype=np.float32
).reshape(1, -1, 1, 2)
y_n = np.array(
    [[np.real(symbol), np.imag(symbol)] for symbol in yn], dtype=np.float32
).reshape(1, -1, 1, 2)
print(y_n.shape)
print(x_n.shape)


w = np.array(
    [
        np.array(
            [
                [[[c_mmse[2], 0], [0, c_mmse[2]]]],
                [[[c_mmse[1], 0], [0, c_mmse[1]]]],
                [[[c_mmse[0], 0], [0, c_mmse[0]]]],
            ]
        ),
        np.array([0, 0]),
    ]
)

#################################################################################

tf.random.set_seed(1)
np.random.seed(1)
model = new_model(m)
# model.layers[0].set_weights(w)
model.summary()
model.fit(y_n, x_n, shuffle=True, epochs=3000, batch_size=1, verbose=1)

