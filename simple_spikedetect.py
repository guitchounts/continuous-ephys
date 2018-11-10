################ Functions to find times and peaks of spikes using the Dw method by Petrantonakis Poirazi 
import numpy as np
from bokeh.io import gridplot, output_file, show, save #, vplot
from bokeh.plotting import figure
from bokeh.models import TapTool, HoverTool
from bokeh.colors import RGB
from bokeh.models import Range1d
from matplotlib import gridspec
import matplotlib.pyplot as plt
from ephys_condition_signal import downsample

def DwEstimation(sig,w,step):
	length = len(sig)
	x = np.arange(0,w)
	y = np.arange(0,length-w+1,step)
	mat = np.zeros([len(x),len(y)],dtype=int)
	for column in range(mat.shape[1]):
		mat[:,column] = x+column
	Dw = np.sqrt(np.sum(np.diff(sig[mat],axis=1)**2,axis=0))
	return Dw


def DetectSpike(data,threshold,RegionWidth):
	cnn=0
	flag=0
	peaks = []
	times = []
	for i,sample in enumerate(data):
		if flag == 0 and data[i] > threshold:
			tempI = i
			flag = 1
		elif flag == 1 and data[i] < threshold and i-tempI > RegionWidth:
			[temppeak,na] = [np.max(data[tempI:i]),np.argmax(data[tempI:i])]
			cnn = cnn+1
			#check if the detected spike's amplitude is unreasonably large (e.g. it's licking artifact)
			if temppeak < 20*threshold: ### or! 10*threshold? 
				peaks.append(temppeak) 
				times.append(tempI-1+na)
				flag = 0
			else:
				flag = 0
				print 'Spike too large. Skipping...'
	#for j,peak in enumerate(peaks):
	#	if peak > 2000:

	return peaks,times



def ifr(spiketimes,time_vec,fs):
	# spike times shoudl be in samples; time_vec = in seconds
    ifr_vec = np.zeros([time_vec.shape[0],1])
    
    for i in range(len(spiketimes)-1):      
        ifr_vec[spiketimes[i]+1:spiketimes[i+1]] = 1 / ((spiketimes[i+1] - spiketimes[i])/fs)
    
    return ifr_vec

def run_spike_detect(data,fs=30e3,w=5,step=1,RegionWidth=15,thresh_fact=4): #### data= samples x channel (pass one channel at a time)
	threshold = thresh_fact*np.median(abs(data)/0.6745);
	D5 = DwEstimation(data,w,step)
	
	peaks,times = DetectSpike(D5,threshold,RegionWidth)

	time_vec = np.arange(0,len(data))/fs

	ifr_vec = ifr(times,time_vec,fs)
	
	return peaks,times,ifr_vec

if __name__ == "__main__":

	file = np.load('/Volumes/Mac HD/Dropbox (coxlab)/Ephys/Data/sample_trace/spikes.npz')
	data = file['data']
	time = file['time']
	fs = file['fs']
	file.close()
	print 'shape of data = ', data.shape

	data = data[0:60*fs]
	time = time[0:60*fs]

	#################### BOKEH FIGURE #########################
	fig = plt.figure(figsize=(20, 10)) 
	num_trials = 1
	gs = gridspec.GridSpec(num_trials, 1, width_ratios=[1, 1]) ## a subplot for every trial
	ax = range(num_trials)
	for trial in range(num_trials):
		print 'trial # ',trial
		peaks,times,ifr_vec = run_spike_detect(data)
		print 'peaks = ', peaks
		#ax = plt.subplot(gs[trial])
		#sns.tsplot(downsample(traces[:,trial],10),time=time,value='Voltage (uV)',color='black',linewidth=0.1)
		ax[trial] = figure(width=1000, plot_height=500)
		#s1 = figure(width=1000, plot_height=500, title='Spikes')
		ax[trial].line(time,data) ## (time is already downsampled)
		ax[trial].circle([t/30e3 for t in times],peaks,color='red') ## [t/30e3 for t in times]
		
		#ax.set_ylim([-1000,1000])
		#ax.set(y_range=Range1d(-1000, 1000))
		#axes.append[ax]
	
	p = gridplot([[s1] for s1 in ax]) #gridplot([[s1] for s1 in axes])

	#fig.savefig(save_folder + '/raw_psth_'+str(stim_name)+'.pdf')

	output_file('sample_spikes.html')
    # show the results
	show(p)



