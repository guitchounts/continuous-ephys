import sys
sys.path.append('/Volumes/Mac HD/Dropbox (coxlab)/Scripts/Repositories/continuous-ephys/open-ephys analysis/')
import OpenEphys
import numpy as np
import matplotlib.pyplot as plt
#eventsfile = "/Volumes/labuser/Desktop/EPHYS/test data/10HzDigitalInput/2015-05-11_14-06-08_diginputtest/all_channels.events"
#eventsfile = "/Volumes/GG Data Raid/Ephys/2015-04-29_17-35-04_digitalinputtest/all_channels_4.events"
#eventsfile = "/Volumes/labuser/Desktop/EPHYS/test data/MWorksPixelInput/2015-05-11_15-53-23_digintest/all_channels.events"
#eventsfile = "/Volumes/labuser/Desktop/EPHYS/test data/MWorksPixelInput/2015-05-12_12-00-22_digintest/all_channels.events"
#eventsfile = "/Volumes/labuser/Desktop/EPHYS/test data/MWorksPixelInput/2015-05-12_17-51-29_captest/all_channels_2.events"
eventsfile = "/Volumes/labuser/Desktop/EPHYS/Data/grat03/2015-05-30_12-46-18/all_channels.events"

events_data = OpenEphys.load(eventsfile)



pixel_ch1 = [] # np.zeros((len(events_data['timestamps']),1))
pixel_ch2 =  [] #np.zeros((len(events_data['timestamps']),1))
pixel_ch1_time = [] #np.zeros((len(events_data['timestamps']),1))
pixel_ch2_time = []

counter = 0
while counter < len(events_data['timestamps']): # go thru all timestamps
	if events_data['channel'][counter] == 3:
		if events_data['eventId'][counter] == 1:
			pixel_ch1.append(1)
			pixel_ch1_time.append(events_data['timestamps'][counter])
		elif events_data['eventId'][counter] == 0:
			pixel_ch1.append(0)
			pixel_ch1_time.append(events_data['timestamps'][counter])
		else:
			pass
		 
	elif events_data['channel'][counter] == 4:
		if events_data['eventId'][counter] == 1:
			pixel_ch2.append(1)
			pixel_ch2_time.append(events_data['timestamps'][counter])
		elif events_data['eventId'][counter] == 0:
			pixel_ch2.append(0)
			pixel_ch2_time.append(events_data['timestamps'][counter])
		else:
			pass
		
	else:
		pass

	counter += 1

fs = 30000
time1 = [i/fs for i in pixel_ch1_time]
time2 = [i/fs for i in pixel_ch2_time]

from scipy.signal import butter, lfilter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='stop')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = lfilter(b, a, data)
    filt_data = filtfilt(b,a,data)
    return filt_data
#filt_data = butter_bandpass_filter(pixel_ch1,200,300,fs)

from scipy import fft, ifft
def notchfilter(data):
    """ Filters the data using notch filter

        Description:
            Digital filter which returns the filtered signal using 60Hz
            notch filter. Transforms the signal into frequency domain
            using the fft function of the Scipy Module. Then, suppresses
            the 60Hz signal by equating it to zero. Finally, transforms
            the signal back to time domain using the ifft function.
        Input:
            ECGdata -- list of integers (ECG data)
        Output:
            ifft(fftECG) -- inverse fast fourier transformed array of filtered ECG data
    """         
    fft_data = fft(data)
    for i in range(len(fft_data)):
        if 200<i<300: fft_data[i]=0
    return ifft(fft_data)
filt_data = notchfilter(pixel_ch1)




ax1 = plt.subplot(2, 1, 1)
plt.plot(time1, pixel_ch1,'-*')
ax2 = plt.subplot(2, 1, 2,sharex=ax1)
plt.plot(time2, pixel_ch2,'-*')


plt.show()
print pixel_ch2[0:50]
print pixel_ch2_time[0:50]
print time[0:50]
"""
pixel_ch1 = np.zeros((len(events_data['timestamps']),1))
pixel_ch2 = np.zeros((len(events_data['timestamps']),1))
counter = 0
while counter < len(events_data['timestamps']): # go thru all timestamps
    if events_data['channel'][counter] == 3:
        pixel_ch1[counter] = 1
    elif events_data['channel'][counter] == 4:
        pixel_ch2[counter] = 1
    else:
        pass
    counter += 1
#print pixel_ch1

plt.subplot(2, 1, 1)
plt.plot(events_data['timestamps'],pixel_ch1)

plt.subplot(2, 1, 2)
plt.plot(events_data['timestamps'],pixel_ch2)

plt.show()
"""


#print events_data
