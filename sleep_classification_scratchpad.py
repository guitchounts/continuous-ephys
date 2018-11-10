
import h5py
import sys
import os
import numpy as np
import re
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
from scipy import signal,stats
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import pylab
import seaborn as sns
import pandas as pd
import numpy.matlib
from sklearn.decomposition import PCA
from matplotlib import gridspec
from sklearn import preprocessing

####### File sorting functions: ########
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def get_raw_files():
    raw_files = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".raw.kwd"):
            raw_files.append(file)
            
    sort_nicely(raw_files)   
    return raw_files


def get_rel_traces(file):

    #relevant_ch = [0, 28, 29, 67, 68, 69]    # take one LFP, EEG, and EMG channel, as well as Accelerometer channels:]

    #for file in raw_files:
    
    input_h5file = h5py.File(file,'a')


    # read input file and select the appropriate channels:
    input_data = np.array(input_h5file['/recordings/0/data'])
    #input_data = np.array(input_data[:,relevant_ch])

    dt = np.dtype([('lfp',np.int64),('eeg',np.int64),('emg',np.int64),('acc1',np.int64),('acc2',np.int64),('acc3',np.int64)])

    data = np.zeros(len(input_data),dtype=dt)

    #input_data = np.array(input_data[:,relevant_ch],dtype=dt)    
    #[lfp,eeg,emg,acc1,acc2,acc3] = input_data[:,relevant_ch].T
    data['lfp'] = np.mean(input_data[:,[0,4,48]],axis=1)
    data['eeg'] = np.mean(input_data[:,[29,30]],axis=1)
    data['emg'] = np.mean(input_data[:,[28,31]],axis=1)
    data['acc1'] = input_data[:,67]
    data['acc2'] = input_data[:,68]
    data['acc3'] = input_data[:,69]


    fs = 30e3
    time = np.arange(len(data))/fs

    ##### CLOSE the FILE?!!!! 
   
    return data,fs,time,dt



#f, Pxx_den = signal.periodogram(input_data[:,0], fs)
#plt.semilogy(f, Pxx_den)
#plt.ylim([1e-8, 1e4])
#plt.xlim([0, 3e2])


def demean_detrend(trace):
    # demean:
    trace = trace - np.mean(trace)

    # detrend:
    trace = signal.detrend(trace)

    return trace


def filter_trace(trace,freq_range,filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='bandpass'):

    # notch:
    #notch = 60
    #notch_bw = 100 # notch filter q factor
    #notch_f = notch/(fs/2)
    #notch_q = notch_f/(notch_bw/(fs/2))
    # ?? 

    # design Elliptic filter:

    [b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,trace)
    return filtered_trace



#freq_range_high = [25, 100]
#[b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range_high],btype='bandpass')
#lfp_25_100 = signal.filtfilt(b,a,lfp)


# In[38]:

def get_specgram(trace,n_samples):

    Pxx, freqs, t, plot = plt.specgram(
        trace,
        NFFT=n_samples, 
        Fs=fs, 
        detrend=pylab.detrend_none,
        window=pylab.window_hanning,
        noverlap=int(n_samples * 0.5))
    plt.ylim([0, 1e2])
    return Pxx, freqs, t



def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]

def windowed_avg(trace,window_size):
    output = []
    x = slidingWindow(trace,window_size,window_size/2)
    for value in x:
        output.append(np.mean(value))

    return output

def windowed_auc(trace,window_size):
    output = []
    x = slidingWindow(trace,window_size,window_size/2)
    for value in x:
        output.append(np.trapz(value))
    return output

# make generator for the index of freqs (and Pxx) to return indeces that included wanted frequenies:
def get_freq_idx(freqs,desired_freq): # make desired_freq a tuple, e.g. (0,4)
    idx = []
    for counter,value in enumerate(freqs):
        if  desired_freq[0] <= value <= desired_freq[1]:
            #yield counter
            idx.append(counter)
    return idx


def get_power(trace,freq_range,n_samples):
    # In[101]:
    Pxx, freqs, t = get_specgram(trace,n_samples)

    idx = get_freq_idx(freqs,freq_range)
        #idx_0_4 = get_freq_idx(freqs,(0,4))
        #idx_5_30 = get_freq_idx(freqs,(5,30))
        #idx_30_100 = get_freq_idx(freqs,(30,100))

    #power = np.mean(Pxx[idx,:],0)
    print 'shape of Pxx[idx,:] = ', Pxx[idx,:].shape
    power = np.trapz(Pxx[idx,:],axis=0)
        #power_0_4 = np.mean(Pxx[idx_0_4,:],0)
        #power_5_30 = np.mean(Pxx[idx_5_30,:],0) # take mean on 0-th dim of Pxx -> this gives power over time. 
        #power_30_100 = np.mean(Pxx[idx_30_100,:],0)


    print 'shape of power = ', power.shape

    return power

def get_windowed_data(data):

    # 1. make structured array to hold data:
    dt_windowed = np.dtype([('lfp_delta',np.int64),('lfp_theta',np.int64),('lfp_gamma',np.int64),('eeg_delta',np.int64),('eeg_theta',np.int64),('eeg_gamma',np.int64),('emg',np.int64),('acc1',np.int64),('acc2',np.int64),('acc3',np.int64)])

    window_size = 2**16 
    Pxx, freqs, t = get_specgram(data['lfp'],window_size) # only doing this to get the size of our windowed data array
    windowed_data = np.zeros(len(t),dtype=dt_windowed)


    # 2. get EEG and LFP at the 3 different frequencies:

    freqs = {'delta':(0,4),'theta':(4,8),'gamma':(25,100)}

    for counter, channel in enumerate(data.dtype.names):

        # normalize the data:
        data[channel] = stats.zscore(data[channel]) 
        
        # for LFP and EEG channels, take spectrogram at different frequencies:
        if channel == 'lfp' or channel == 'eeg':
            for freq_range in freqs:
                windowed_data[str(channel + '_' + freq_range)] = get_power(data[channel],freqs[freq_range],window_size)
                print str(channel + '_' + freq_range)
        # for EMG and Acc channels, get windowed average:
        elif channel == 'emg' or channel == 'acc1' or channel == 'acc2' or channel == 'acc3':
            windowed_data[channel] = windowed_auc(data[channel],window_size)
            print channel
        else:
            print 'you chose a funky channel name'


    return windowed_data


if __name__ == "__main__":

    #0. make save folder:
    save_folder = './sleep_traces/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #1. gather KWD files. 
    raw_files = get_raw_files()
    #print raw_files
    ###### Then, for each file: 

    #2. get appropriate lfp, emg, eeg, acc traces
    for file_idx,file in enumerate(raw_files):
        
        save_name = save_folder + 'windowed_data_exp_' + str(file_idx+1)

        if os.path.isfile((save_name+'.npz')) == True:
            print 'Experiment %i exists. Skipping...' % (file_idx+1)
        else:
            print 'Processing experiment %i...' % (file_idx+1)

            
            data,fs,time,dt = get_rel_traces(file)

            

            #3. demean, detrend, and filter the LFP and EEG into different bands [delta, theta, gamma]
                    #lfp = demean_detrend(lfp)
                    #eeg = demean_detrend(eeg)
                    #for trace in [lfp,eeg,emg,acc1,acc2,acc3]:
                    #     demean_detrend(trace)
            

            for channel in dt.names:

                data[channel] = demean_detrend(data[channel])

                #data[channel] = filter_trace(data[channel],[0,100])
                #data[channel] = stats.zscore(data[channel])
                

                Pxx, freqs, t = get_specgram(data[channel],n_samples=2**16)

                # if power_data is not a variable, make it, with length of Pxx:
                if 'power_data' in globals():
                    print 'found variable power_data...'
                    power_data[channel] = np.vstack([Pxx.T, freqs]).T
                else:
                    print 'creating variable power_data...'
                    power_data = np.zeros([len(freqs),Pxx.shape[1]+1],dtype=dt)
                    power_data[channel] = np.vstack([Pxx.T, freqs]).T


            # OPTION: filter:
                #lfp_0_4 = filter_trace(lfp,[0,4],filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='lowpass')
                #lfp_4_12 = filter_trace(lfp,[4,12])
                #lfp_25_100 = filter_trace(lfp,[25,100])

                #eeg_0_4 = filter_trace(eeg,[0,4],filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='lowpass')
                #eeg_4_12 = filter_trace(eeg,[4,12])
                #eeg_25_100 = filter_trace(eeg,[25,100])

           


            #4. get power over time for the different frequencies 

            windowed_data = get_windowed_data(data)

            #lfp_power_0_4 = get_power(data['lfp'],(0,4))
            #lfp_power_4_12 = get_power(data['lfp'],(4,12))
            #lfp_power_25_100 = get_power(data['lfp'],(25,100))

            #eeg_power_0_4 = get_power(data['eeg'],(0,4))
            #eeg_power_4_12 = get_power(data['eeg'],(4,12))
            #eeg_power_25_100 = get_power(data['eeg'],(25,100))

            #5. save windowed data file under experiment # name. 
            
            np.savez_compressed(save_name,windowed_data=windowed_data,power_data=power_data,t=t)

            ####### PLOT STUFF #########
            print("Plotting stuff...")


            idx_0_25 = get_freq_idx(freqs,(0,25))
            Pxx_lfp_0_25 = power_data[idx_0_25,0:-1]['lfp']
            Pxx_lfp_0_25_intgr = stats.zscore(np.trapz(Pxx_lfp_0_25,axis=0))
            print 'shape of Pxx_lfp_0_25_intgr = ', Pxx_lfp_0_25_intgr.shape
            #make Pandas dataframe out of Pxx for plotting heatmap
            b = pd.DataFrame(data=Pxx_lfp_0_25,
                index = freqs[idx_0_25],
                columns = t )

            
            pca = PCA(n_components=2,whiten=True)
            broadband_lfp_pca = pca.fit(Pxx_lfp_0_25.T).transform(Pxx_lfp_0_25.T)

            ######## the figure: #######

            fig = plt.figure(figsize=(20, 10)) 
            ax1 = sns.heatmap(b,robust=True,xticklabels=1000, yticklabels=10,cbar=False)
            ax1.invert_yaxis()

            sns.set_style("white",{'axes.linewidth' : 0.01})
            #sns.xkcd_rgb["pale red"]
            ax2 = ax1.twinx()
            ax2 = sns.tsplot(data=stats.zscore(np.trapz(Pxx_lfp_0_25,axis=0)),value='Integrated LFP',color='green',linewidth=0.1)
            #ax2.invert_yaxis()
            ax2.yaxis.set_visible(False)
            
            pos1 = ax1.get_position()
            new_pos = pos1
            new_pos.y0 -= 0.1 #new_pos.y0 
            ax2.set_position(new_pos)
            
            ax3 = ax1.twinx()
            ax3 = sns.tsplot(data=windowed_data['emg'],value='EMG',color='blue',linewidth=0.1)
            ax3.yaxis.set_visible(False)

            sns.despine()

            fig.savefig((save_name+".pdf"))

            del power_data
        

        #########
        # things to save from each experiment kwd file:
        # 1. power at different freqencies for the LFP and EEG channel (length 273 vector, if window size is 2^16)
        # 2. [normalized magnitude] of EMG channel
        # 3. 

