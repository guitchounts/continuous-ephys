import numpy as np
from scipy import signal
import h5py

############## get __filtered raw traces__ in different bands: requires to load raw LFPs:
# TO DO: the code for this... 

def get_lfp(lfp_path,channel_name):

    ### path is the directory w/o the file name. 
    ### channel name is the filename, e.g. Ch_40

    num_chans = len(channel_name)

    for ch in range(num_chans):

        lfp_file = open(lfp_path + str(channel_name[ch]),"rb")

        print 'LFP file path = ', lfp_path + str(channel_name[ch])
        lfp = np.fromfile(lfp_file,dtype=np.int16)
        
        lfp_file.close()

        #filtered_lfp = filter(lfp,[1, 800]) 
        filtered_lfp = lfp

        if ch == 0:
            all_lfps = filtered_lfp
        else:
            all_lfps = np.vstack([all_lfps, filtered_lfp])

    print 'shape of all_lfps = ', all_lfps.shape
    return all_lfps




def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):

    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace



############## get __power__ in different bands: requires to load spectrogram


def get_power_bands(lfp_spec,freqs):


    freq_bands = [ [0,4],[4,8],[8,12],[12,30],[30,60],[60,150] ]

    lfp_power = np.zeros([64*6,lfp_spec.shape[2]])  ## 64 channels x 4 bands

    counter = 0

    for ch in range(64):    
        
        for freq_band in freq_bands:
            power = get_power(lfp_spec[ch,:,:],freq_band,freqs)

            lfp_power[counter,:] = power
            counter += 1
        #power_0_4 = get_power(lfp_spec[ch,:,:],[0,4],freqs)
        #power_4_8 = get_power(lfp_spec[ch,:,:],[4,8],freqs)
        #power_8_12 = get_power(lfp_spec[ch,:,:],[8,12],freqs)
        #power_15_40 = get_power(lfp_spec[ch,:,:],[15,40],freqs)
        #power_40_100 = get_power(lfp_spec[ch,:,:],[40,100],freqs)
        #lfp_power[ch*4:(ch+1)*4,:] = power_0_4,power_5_15,power_15_40,power_40_100


    
    return lfp_power


def get_freq_idx(freqs,desired_freq): # make desired_freq a tuple, e.g. (0,4)
    idx = []
    for counter,value in enumerate(freqs):
        if  desired_freq[0] <= value <= desired_freq[1]:
            #yield counter
            idx.append(counter)
    return idx


def get_power(spec,freq_range,freqs):

    idx = get_freq_idx(freqs,freq_range)

    power = np.mean(spec[idx,:],0)

    return power


if __name__ == "__main__":

    lfp_file = h5py.File('lfp_spec.mat','r')

    lfp_spec = lfp_file['lfp_spec'][:]
    freqs = lfp_file['f'][:]

    print 'Shape of lfp spec = ', lfp_spec.shape

    lfp_file.close()

    lfp_power = get_power_bands(lfp_spec,freqs)

    print 'Shape of lfp_power = ', lfp_power .shape

    power_file = h5py.File('lfp_power.hdf5','w')

    power_file.create_dataset('lfp_power', data=lfp_power)

    power_file.close()





