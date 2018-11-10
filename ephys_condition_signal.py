import numpy as np
from scipy import signal,stats

def filter(ephys,freq_range,filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='bandpass',fs=30e3):
	## ephys = samples x trials or channels
	
    

    # notch:
    #notch = 60
    #notch_bw = 100 # notch filter q factor
    #notch_f = notch/(fs/2)
    #notch_q = notch_f/(notch_bw/(fs/2))
    # ?? 

    # design Elliptic filter:

    [b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def median_rejection(ephys,channels,exclude=[]):
    #channels = np.arange(ephys.shape[-1]) + 1 ### channels should start with 1. ## channels are the last dimension of ephys
    if channels[0] == 0:
        channels = channels + 1

    if exclude == []:
        keep_channels = channels
    else:
        for index in sorted(exclude,reverse=True):
            keep_channels = np.delete(channels,index-1) ### subtracting 1 b/c added it to channels.  
    print 'channels ===== ', channels
    if len(ephys.shape) == 3: ### assuming ephys is samples x trials x channels
        
        for trial in range(ephys.shape[1]):
            median = np.median(ephys[:,trial,keep_channels-1],axis=1) ## axis 0 = samples 
            ### subtract 1 from keep_channels b/c index starts w/ 0 whereas channels/keep_channels have 1 added on to them
            for ch in channels:
                ephys[:,trial,ch] -= median

    elif len(ephys.shape) == 2: ### ephys shape = samples x channels 
        median = np.median(ephys[:,keep_channels-1],axis=1)
        for ch in channels:
            ephys[:,ch-1] -= median

    return ephys

def rectify(ephys):
	return ephys**2

def downsample(ephys,downfact): ### downfact = downsample factor; ephys shape = samples * trials * channels
	return signal.decimate(ephys,downfact,ftype='fir',axis=0)