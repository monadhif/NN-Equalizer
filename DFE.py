"""
Brief: DFE equalizer using BPSK symbols
Author: Mona DHIFLAOUI <mona.dhiflaoui@gmail.com>
Internship 2020
"""
import numpy as np
import numcomm #chez orange 
import itertools
import matplotlib.pyplot as plt
from scipy import signal
from math import *
from numpy.random import rand, randn
from numpy import linalg as LA




def decision_feedback(N_BYTES,training_len, snr_dB, ff_filter_len,fb_filter_len,data_len,fade_var_1D,chan_len, QAM_ORDER):

    #snr parameters
    snr = 10**(0.1*snr_dB)
    noise_var_1D = 2*(2*fade_var_1D*chan_len)/(2*snr) # noise variance
    #X = [-1 - 1j,-1- 1j,1 + 1j,1 - 1j]
    training_a =  np.random.randint(2, size=2*N_BYTES)
    X= 1-2*training_a[::2] + 1j*(1-2*training_a[1::2])
    #impulse response of the channel
    fade_chan = [0.9+0.9*1j ,0.1+0.1*1j, 0.1+0.1*1j ,0.1+0.1*1j, 0.1+0.1*1j]
    fade_chan = fade_chan/ LA.norm(fade_chan)
    chan_len = len(fade_chan)
    #awgn
    noise = np.random.normal(0, sqrt(noise_var_1D), training_len+chan_len-1)+ np.random.normal(0, sqrt(noise_var_1D), training_len+chan_len-1)
    chanOut= np.convolve(fade_chan,X)+noise
    #(chan_op, real_SNR_dB,noise_power ) = AWGN_noise(samples=chanOut,target_SNR_dB=snr_dB)
    chan_op = chanOut.reshape(1,-1)
    # ------------ LMS update of taps------------------------------------------
    ff_filter = np.zeros(ff_filter_len,dtype =np.complex64)# feedforward filter initialization
    fb_filter = np.zeros(fb_filter_len,dtype =np.complex64)# feedback filter initialization
    ff_filter_ip = np.zeros(ff_filter_len,dtype =np.complex64)#feedforward filter input vector
    fb_filter_ip = np.zeros(fb_filter_len,dtype =np.complex64)# feedback filter input vector
    fb_filter_op = 0# feedback filter output symbol
    #estimating the autocorrelation of received sequence at zero lag
    Rvv0 = np.matmul(chan_op,chan_op.conj().T)/(N_BYTES+chan_len-1)
    #maximum step size
    max_step_size = 1/(ff_filter_len*(Rvv0)+fb_filter_len*(2))
    step_size = 0.125*max_step_size
    for i1 in range(N_BYTES-ff_filter_len+1 ):
        ff_filter_ip[1:ff_filter_len]=ff_filter_ip[0:ff_filter_len-1]
        ff_filter_ip[0] = chan_op[i1]
        ff_filter_op =  np.matmul(ff_filter,ff_filter_ip.T)   #  feedforward filter output
    
        ff_and_fb = ff_filter_op-fb_filter_op 
        error = ff_and_fb-X[i1]
   
        temp1 =np.real(ff_and_fb)<0
        temp2 = np.imag(ff_and_fb)<0 
        quantizer_op = 1-2*temp1 + 1j*(1-2*temp2)
        #LMS update
        ff_filter=ff_filter-step_size*error*ff_filter_ip.conj()
        fb_filter=fb_filter+step_size*error*fb_filter_ip.conj()
         
        fb_filter_ip[1:fb_filter_len]=fb_filter[0:fb_filter_len-1]
        fb_filter_ip[0] = quantizer_op;    
        fb_filter_op = np.matmul(fb_filter,fb_filter_ip.T)
    snr_dB =range(0,20)
    ber = np.zeros(len(snr_dB))
    #-------    data transmission phase----------------------------
    # source
    data_a =  np.random.randint(2, size=2*data_len)
    data_seq= 1-2*data_a [::2] + 1j*(1-2*data_a [1::2])
    for ii in range (0,len(snr_dB)):#len(snr_dB)
        snr = 10**(0.1*snr_dB[ii])
        noise_var_1D = 2*(2*fade_var_1D*chan_len)/(2*snr)
        noise = np.random.normal(0, sqrt(noise_var_1D), data_len+chan_len-1)+ np.random.normal(0, sqrt(noise_var_1D), data_len+chan_len-1)
        #channel output
    
        chan_op = np.convolve(fade_chan,data_seq)+noise
        dec_seq = np.zeros(data_len-ff_filter_len+1,dtype =np.complex64)# output from dfe
        ff_filter_ip = np.zeros(ff_filter_len,dtype =np.complex64)#feedforward filter input
        fb_filter_ip =np.zeros(fb_filter_len,dtype =np.complex64)# feedback filter input
        fb_filter_op = 0 #feedback filter output symbol
        for i1 in range(data_len-ff_filter_len+1):
            ff_filter_ip[1:ff_filter_len]=ff_filter_ip[0:ff_filter_len-1]
            ff_filter_op =  np.matmul(ff_filter,ff_filter_ip.T)   #  feedforward filter output
            ff_and_fb = ff_filter_op-fb_filter_op 
            #hard decision
            temp1 = np.real(ff_and_fb)<0
            temp2 = np.imag(ff_and_fb)<0
            dec_seq[i1] = 1-2*temp1 +1j*(1-2*temp2)
         
            fb_filter_ip[1:fb_filter_len]=fb_filter[0:fb_filter_len-1]
            fb_filter_ip[0] = dec_seq[i1]
            fb_filter_op =np.matmul(fb_filter,fb_filter_ip.T)
        
        dec_a = np.zeros(2*(data_len-ff_filter_len+1))
        dec_a[::2] = np.real(dec_seq)<0
        dec_a[1::2] = np.imag(dec_seq)<0
        ber[ii] = np.count_nonzero(dec_a-data_a[0:2*(data_len-ff_filter_len+1)])/(2*(data_len-ff_filter_len+1))
    return(dec_a,ber)





     
