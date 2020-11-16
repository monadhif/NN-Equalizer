"""
Brief: Multi_path equalizer using convolutional neural network
Author: Mona DHIFLAOUI <mona.dhiflaoui@gmail.com>
Internship 2020
"""
import numpy as np
import numcomm #chez orange 
import itertools
from numpy.random import rand, randn
import tensorflow as tf
from scipy.linalg import toeplitz
from Rayleigh_Rician import Ray_model

def AWGN_noise(samples, target_SNR_dB):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    samples : [complex]
        [M-QAM samples]
    target_SNR_dB : [integer]
        [The SNR value]
    """
    # Calculate signal power and convert to dB
    sig_power = np.abs(np.correlate(samples, samples, "valid") / samples.size).astype(
        np.float64
    )
    sig_power_db = 10 * np.log10(sig_power)

    # Calculate noise according to average signal power
    noise_power_db = sig_power_db - target_SNR_dB
    noise_power = 10 ** (noise_power_db / 10)

    # Generate samples of white gaussian) noise
    mean_noise = 0
    noise_samples_I = np.random.normal(
        mean_noise, np.sqrt(noise_power / 2), len(samples)
    )
    noise_samples_Q = np.random.normal(
        mean_noise, np.sqrt(noise_power / 2), len(samples)
    )
    #total_noise_samples = noise_samples_I + noise_samples_Q
    # Noise up the original signal
    noisy_samples = samples + noise_samples_I + 1j * noise_samples_Q

    # Compute real SNR
    real_noise_power = np.abs(
        np.correlate(noise_samples_I, noise_samples_I, "valid") / noise_samples_I.size
    )
    real_SNR_dB = 10 * np.log10(sig_power / real_noise_power)[0]
    return (np.array((noisy_samples), dtype=np.complex64), real_SNR_dB, noise_power)

def generate_samples(QAM_ORDER, N_BYTES):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    QAM_ORDER : [integer (4^n) n ={1,2,3,..}]
        [The order of the modulation QAM]
    N_BYTES : [integer]
        [The number of integers that will be generated  between 0 & 255 using numcomm  library]
    target_SNR_dB : [integer]
        [The SNR value]
    """

    QAM_modulator = numcomm.modems.QAM(order=QAM_ORDER)
    # We generate integer between 0 & 255 = One byte or 8 bits
    bits = np.array(np.random.randint(0, 255, N_BYTES), dtype=np.uint8)  # As integer
    #bits_representation = numcomm.tools.bytestobits(bits, 1)  # As bits
    symbols = QAM_modulator.modulate(bits)
    return (symbols)

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


def NN_MMSE(new_model, N_BYTES,QAM_ORDER, target_SNR,L):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    new_model : [Neural Network]
        [The model of CNN that will be used]
    N_BYTES : [Integer]
        [The number of integers that will be generated  between 0 & 255 using numcomm  library]
    QAM_ORDER : [integer]
        [the order of the QAM]
    target_SNR : [Integer]
        [the SNR value]
    L : [Integer]
        [Number of the path in the channel]
    """
    h = Ray_model(L).flatten() # Example h = [0.2 ,0.9, 0.3]
    symbols = generate_samples(QAM_ORDER, N_BYTES) #for example QAM_ORDER =16,  N_BYTES = 4096
    chanOut = np.convolve(symbols, h)
    m = len(chanOut)
    (received_sym, real_SNR_dB, noise_power) = AWGN_noise(samples=chanOut, target_SNR_dB=target_SNR)
    data = np.array([[np.real(symbol), np.imag(symbol)] for symbol in symbols ], dtype=np.float32).reshape(1, -1, 1, 2)
    received_data = np.array([[np.real(symbol), np.imag(symbol)] for symbol in received_sym], dtype=np.float32).reshape(1, -1, 1, 2)
    np.random.seed(1)
    model = new_model(m)
    # model.layers[0].set_weights(w)
    model.summary()
    model.fit(received_data, data , shuffle=True, epochs=3000, batch_size=1, verbose=1)
    return(model)
