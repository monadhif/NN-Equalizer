"""
Brief: Multi_path equalizer using convolutional neural network
Author: Mona DHIFLAOUI <mona.dhiflaoui@gmail.com>
Internship 2020
"""
import numpy as np
import numcomm  # Orange Library
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
def Analy_MMSE(QAM_ORDER, N_BYTES, target_SNR,L, K):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    QAM_ORDER : [integer]
        [the order of the QAM]
    N_BYTES : [integer]
        [The number of integers that will be generated  between 0 & 255 using numcomm  library]
    target_SNR : [integer]
        [the SNR value]
    L : [integer]
        [Number of the path in the channel]
    K : [integer]
        [Number of taps in the equalizer]

    Returns
    -------
    c_mmse : the analytical MMSE coefficient
    ySamp_mmse : the estimated sympols using MMSE
    estim_initial_bits : the demodulation of the  estimated sympols
    """

    QAM_modulator = numcomm.modems.QAM(order=QAM_ORDER)
    h = Ray_model(L).flatten() # Example h = [0.2 ,0.9, 0.3]
    symbols = generate_samples(QAM_ORDER, N_BYTES) #for example QAM_ORDER =16,  N_BYTES = 4096
    N = len(symbols )
    chanOut = np.convolve(symbols, h)
    (received_sym, real_SNR_dB, noise_power) = AWGN_noise(samples=chanOut, target_SNR_dB=target_SNR)
    #auto-correlation matrix of h
    hAutoCorr = np.convolve(h, np.flip(h).conj().T)
    L_m = len(hAutoCorr)
    m = L_m // 2
    #toeplitz matrix of the auto-correlation matrix of h
    h1_m = np.concatenate((hAutoCorr[m:L_m], np.zeros(2 * K + 1 - L_m + m)))
    h2_m = np.concatenate((hAutoCorr[::-1][m:L_m], np.zeros(2 * K + 1 - L_m + m)))
    hM1 = toeplitz(h1_m, h2_m)

    hM = hM1 + noise_power * np.eye(2 * K + 1)
    d_m = np.zeros(2 * K + 1)
    d_m [K-1 : K+L-1] = np.flip(h)
    #The MMSE coefficient
    c_mmse = np.matmul(np.linalg.inv(hM), d_m.conj().T)
    #equalize the received symbols
    yFilt_mmse = np.convolve(received_sym, c_mmse)   
    ySamp_mmse = np.array(yFilt_mmse[0:N], dtype=np.complex64)# estimated symbols

    estim_initial_bits = QAM_modulator.demodulate(ySamp_mmse)  # demodulate the estimated symbols
    return (c_mmse, ySamp_mmse, estim_initial_bits,symbols, received_sym )





