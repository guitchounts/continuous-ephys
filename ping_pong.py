

from scipy import stats,signal
import numpy as np
import sys,os
import seaborn as sns
import matplotlib.pylab as plt
sns.set_style("white",{'axes.linewidth' : 0.01})
import pandas as pd



#sys.path.append('/Users/guitchounts/Dropbox (coxlab)/Scripts/Repositories/analysis-tools')
sys.path.append('/Volumes/Mac HD/Dropbox (coxlab)/Scripts/Repositories/analysis-tools')
import OpenEphys

#sys.path.append('/Volumes/Mac HD/Dropbox (coxlab)/Scripts/Repositories/continuous-ephys')
#import simple_spikedetect

def filter(ephys,freq_range,filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='bandpass',fs=30e3):

    ## notch filter first:

    [b,a] = signal.iirnotch(60/(fs/2),Q=30)
    ephys = signal.filtfilt(b,a,ephys,axis=0)

    # design Elliptic filter:
    [b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace



def get_ttls(data_path):

    print 'data_path  = ', data_path
    events_data = OpenEphys.load('all_channels.events')


    ########## LOAD THE TTLs!!! #############
    #temp = np.intersect1d((np.where(events_data['channel'] == 0)[0]), (np.where(events_data['eventId'] == 1)[0]))
    ttl0_on = events_data['timestamps'][np.intersect1d((np.where(events_data['channel'] == 0)[0]), (np.where(events_data['eventId'] == 1)[0]))]

    #temp = np.intersect1d((np.where(events_data['channel'] == 0)[0]), (np.where(events_data['eventId'] == 0)[0]))
    ttl0_off = events_data['timestamps'][np.intersect1d((np.where(events_data['channel'] == 0)[0]), (np.where(events_data['eventId'] == 0)[0]))]

    #temp = np.intersect1d((np.where(events_data['channel'] == 1)[0]), (np.where(events_data['eventId'] == 1)[0]))
    ttl1_on = events_data['timestamps'][np.intersect1d((np.where(events_data['channel'] == 1)[0]), (np.where(events_data['eventId'] == 1)[0]))]

    #temp = np.intersect1d((np.where(events_data['channel'] == 1)[0]), (np.where(events_data['eventId'] == 0)[0]))
    ttl1_off = events_data['timestamps'][np.intersect1d((np.where(events_data['channel'] == 1)[0]), (np.where(events_data['eventId'] == 0)[0]))]


    return ttl0_on, ttl0_off, ttl1_on, ttl1_off




def plot_responses(data_path,save_path):

    ttl0_on, ttl0_off, ttl1_on, ttl1_off = get_ttls(data_path)

    fs = 3e4

    time_axis = np.arange(-.250,.750,1/fs)

    channels = range(64) # [0,1] #

    allchans_binoc = dict()

    ######### LOAD EPHYS 
    for ch in channels:


        ephys_data = OpenEphys.load('100_CH%i.continuous' % (ch+1))

        filt_data = filter(ephys_data['data'],[1,200])


    ###### take ttl0_on times (in samples), get corresponding chunks from the ephys (250ms before that time to 750ms after)
        offset_time = ephys_data['timestamps'][0]

        right_led_ephys = np.zeros([ttl1_on.shape[0],int(1.0*fs)]) 
        for idx,on_time in enumerate(ttl1_on):
            offset_ontime = on_time - offset_time
            #print idx,on_time,offset_ontime
            if offset_ontime-int(.250*fs)>0:
                if filt_data.shape[0] - offset_ontime > 0:
                    if offset_ontime+int(.750*fs) <filt_data.shape[0] :
                        right_led_ephys[idx,:] = filt_data[int(offset_ontime-int(.250*fs)):int(offset_ontime+int(.750*fs))]


        left_led_ephys = np.zeros([ttl0_on.shape[0],int(1.0*fs)]) 
        for idx,on_time in enumerate(ttl0_on):
            offset_ontime = on_time - offset_time
            #print idx,on_time,offset_ontime
            if offset_ontime-int(.250*fs)>0:
                if filt_data.shape[0] - offset_ontime > 0:
                    if offset_ontime+int(.750*fs) <filt_data.shape[0] :
                    
                        left_led_ephys[idx,:] = filt_data[int(offset_ontime-int(.250*fs)):int(offset_ontime+int(.750*fs))]
          


        mean_left_led = np.mean(left_led_ephys,axis=0)
        #std_left_led = np.std(left_led_ephys,axis=0)
        sem_left_led = stats.sem(left_led_ephys,axis=0)

        mean_right_led = np.mean(right_led_ephys,axis=0)
        #std_right_led = np.std(right_led_ephys,axis=0)
        sem_right_led = stats.sem(right_led_ephys,axis=0)

        ##### binocularity index for each electrode:

        left_max = np.max(abs(mean_left_led))
        right_max = np.max(abs(mean_right_led))
        binocularity = (left_max - right_max) / (left_max + right_max)
        print 'binocularity for ch %i = %6f' % ((ch+1),binocularity)
        
        allchans_binoc[str(ch+1)] = [binocularity]
        
        
        plot = 1
        if plot == 1:
            f, axarr = plt.subplots(2, sharex=True,sharey=True) #plt.figure(dpi=600)
            f.dpi = 600
            f.suptitle('binocularity for ch %i = %6f' % (ch,binocularity))
            #ax = sns.tsplot(data=mean_left_led_ch22,time=time_axis, linewidth=0.1)

            axarr[0].plot(time_axis,mean_left_led,linewidth=0.1)
            axarr[0].fill_between(time_axis, mean_left_led-sem_left_led, mean_left_led+sem_left_led,alpha=0.1)
            axarr[0].set_ylabel('Left LED Response (uV)')

            axarr[1].plot(time_axis,mean_right_led,linewidth=0.1)
            axarr[1].fill_between(time_axis, mean_right_led-sem_right_led, mean_right_led+sem_right_led,alpha=0.1)
            axarr[1].set_ylabel('Right LED Response (uV)')

            axarr[1].set_xlabel('Time (sec)')
            axarr[1].set_xticks

            start, end = axarr[1].get_xlim()
            stepsize = 0.1
            axarr[1].xaxis.set_ticks(np.arange(start, end, stepsize))

            # add a square wave showing when the stimulus happened:
            plt.plot([-0.25,0,0,0.25,0.25,0.75],[-75,-75,-50,-50,-75,-75],color='black')

            f.savefig(save_path + 'ch%i.pdf'%(ch+1),dpi=600)


    #total_binocularity = pd.DataFrame.from_dict(allchans_binoc)
    #total_binocularity.to_pickle('binocularity')


if __name__ == "__main__":

    data_path = os.getcwd()

    save_path = './ping_pong_responses/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    plot_responses(data_path,save_path)



