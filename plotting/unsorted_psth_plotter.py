#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure,show
from bokeh.io import output_notebook, gridplot, output_file, show, save
import struct
import datetime
import pandas as pd
from scipy import stats,signal,ndimage
from matplotlib import gridspec
import os,sys
from numpy import matlib
import h5py

def filter(ephys,freq_range,filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='bandpass',fs=30e3):
	
    # design Elliptic filter:

    [b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace


def plot_spike_psth(spike_times,stim_data,channel_name,unsorted_save_folder):


	################################ get the TRIAL TIMES  ###################################
	#########################################################################################

	trials = range(len(stim_data))
	stim_orientation,idx,inv = np.unique(stim_data.orientations,return_index=True,return_inverse=True)
	stim_dict = dict()
	for ori in stim_orientation:	
	    stim_dict[str(ori)] = stim_data.orientations[stim_data.orientations==ori].index.values ### put indices of stim_data that correspond to this orientation, from inv
	    #### ^ stim_dict gives indices of stim_data for given orientations. 

	trial_range = 1.0 # 0.5
	#trial_time_vector = np.linspace(-trial_range,trial_range,300) ## 300=trial length, in LFP sample (LFP fs=300 samples/sec)

	

	###### Gather spike data from behavior trials:
	spikes = []

	for i in trials:
		print 'i = ', i

		trial_time = stim_data.times[i]

		print 'trial_time = ', trial_time

		temp_idx = (spike_times < trial_time+trial_range) & (spike_times > trial_time - trial_range)
    
		#print 'temp_idx = ', temp_idx
		temp_trial = spike_times[temp_idx] - trial_time


		spikes.append(temp_trial)

	

	stim_data['spikes'] = spikes

	stim_data.to_csv(unsorted_save_folder + '/stim_data_spikes.csv')

	print 'STIM_DATA = ', stim_data


	tuning_dict = dict()

	for ori in stim_dict:
    
		print stim_data.spikes[stim_dict[ori]]

		raster = figure(width=1000, height=400,y_axis_label='Trial Number',title='Raster')
		all_spike_times = []

		
		num_trials = len(stim_data.spikes[stim_dict[ori]]) #  len(trials)
		trials_by_ori = range(num_trials)		
		bin_time = 0.025 # 25 ms bins
		num_bins = np.int(trial_range*2 / bin_time)
		time_vec = np.linspace(-trial_range,trial_range,num_bins)
		tuning_time = 0.1 # sec. window in which to look for response modulation, before and after stim onset. 
		alltrials_modulation = []

		for trial in trials_by_ori:

			trial_time = stim_data.spikes[stim_dict[ori][trial]] ## spike times on this trial
			num_spikes_ontrial = len(trial_time)

			print 'trial %d has %d spikes' % (trial,num_spikes_ontrial)

			all_spike_times.append(trial_time)

			####### MAKE RASTER
			raster.segment(x0=trial_time, y0=np.repeat(trial,num_spikes_ontrial), x1=trial_time,
		                   y1=np.repeat(trial+1,num_spikes_ontrial), color="black", line_width=0.5)

			#### get histogram for each trial:
			temp_hist,edges = np.histogram(trial_time, bins=num_bins,range=[-trial_range,trial_range])

			spikes_after_stim = temp_hist[(edges<tuning_time) & (edges>=0)]
			spikes_before_stim = temp_hist[(edges<0) & (edges>-tuning_time)]

			print 'spikes_after_stim = ', spikes_after_stim
			print 'spikes_before_stim = ', spikes_before_stim

			trial_modulation = np.sum(spikes_after_stim) - np.sum(spikes_before_stim) ## total spikes in 100ms after - total spikes in 100ms before stim.
			alltrials_modulation.append(trial_modulation/tuning_time) ## divide by 0.1 to get #spikes/second

			if trial == 0:
				raw_hist = temp_hist	
			else:
				raw_hist = np.vstack([raw_hist,temp_hist])

		np.savez(unsorted_save_folder + '/raw_histogram_%s.npz' % str(ori), raw_hist=raw_hist)

		print 'length of all_spike_times = ', len(all_spike_times)
		print 'shape of overall histo = ', raw_hist.shape

		histo_fig = figure(width=1000, plot_height=500,y_axis_label='Firing Rate (Hz)',x_axis_label='Time (sec)',x_range=raster.x_range)


		
		all_spike_times = np.concatenate(all_spike_times).ravel() ## make one long vector of spike times. 

		#hist, edges = np.histogram(all_spike_times, bins=num_bins)
		
		hist = np.mean(raw_hist,axis=0)
		hist_std = np.std(raw_hist,axis=0)
		print 'shape of averaged hist = ', hist.shape

		bin_width = np.diff(edges)[0] # in seconds.

		print 'hist, edges: ', hist,edges

		print 'bin_width = ', bin_width

		print 'num_trials = ', num_trials

		print 'shape of hist, edges = ', hist.shape,edges.shape



		histo_fig.quad(top=hist/bin_width, bottom=0, left=edges[:-1], right=edges[1:], ## hist/bin_width = firing rate in Hz
		    fill_color="#036564", line_color="#036564") #033649

		#histo_fig.line(edges[:-1]+bin_width/2, hist/bin_width)		
		### get the number of spikes in the 100ms before and after stim presentation:

		

		tuning_dict[ori] = (np.mean(alltrials_modulation),stats.sem(alltrials_modulation))   ## (np.mean(hist[20:24])/np.mean(hist[16:20]))/bin_width/num_trials

		print 'tuning dict = ', tuning_dict

		print 'np.mean(alltrials_modulation) = ', np.mean(alltrials_modulation)
		print 'np.std(alltrials_modulation) = ', np.std(alltrials_modulation)


		#################################### instead of histogram, plot mean+std of numbers of spikes on each trial (binned).
		firingrate_line = figure(width=1000, plot_height=500,y_axis_label='Firing Rate (Hz)',x_axis_label='Time (sec)',x_range=raster.x_range)

		hist_smooth = ndimage.filters.gaussian_filter1d(hist/bin_width,sigma=0.5)

		firingrate_line.line(edges[:-1]+bin_width/2,hist_smooth)
		std_x = (edges[:-1]+bin_width/2).tolist()
		std_y1 = (-hist_std+hist/bin_width).tolist()
		std_y2 = (hist_std+hist/bin_width).tolist()



		print 'std_x = ', std_x
		print 'std_y1 ',std_y1
		print 'std_y2 = ',std_y2

		std_x.extend(np.flipud(std_x).tolist())
		std_y1.extend(np.flipud(std_y2).tolist())

		print 'std_plot_x = ', std_x
		print 'std_plot_y = ', std_y1



		firingrate_line.patch(std_x,std_y1,color="#99d8c9")
		#firingrate_line.patch(edges[:-1]+bin_width/2,hist_std+hist/bin_width)
		#firingrate_line.patch(edges[:-1]+bin_width/2,hist/bin_width+hist_std)

		
		all_spike_times = None 

		#histo_fig.line(win_x,[w/win_size/len(stim_dict[ori]) for w in windowd_spike_vec],color='magenta')

		output_file(unsorted_save_folder + '/spike_histogram_'+str(ori)+'.html')
		grid = gridplot([[raster], [histo_fig],[firingrate_line]]) #,[firingrate_line]

		save(grid)

	    #stim_name = 'orientation_' + ori
	    #output_file(save_folder + '/psth_'+str(stim_name)+'.html')
	    #save(raster)

		#fig.savefig((unsorted_save_folder + str(channel_name) + '.pdf'))
	
	## plot tuning curve:
	
	tuning_fig = figure(width=1000, plot_height=500,y_axis_label='Spikes/Sec After - Before Stim',x_axis_label='Orientations (deg)')

	#### plot means:
	tuning_fig.scatter(tuning_dict.keys(),[means[0] for means in tuning_dict.values()])
	#### plot stds:
	## get x and y coordinates for the lines:
	err_xs = []
	err_ys = []

	for x, y, yerr in zip(tuning_dict.keys(),[means[0] for means in tuning_dict.values()], [stds[1] for stds in tuning_dict.values()]):
		err_xs.append((x, x))
		err_ys.append((y - yerr, y + yerr))
	tuning_fig.multi_line(err_xs,err_ys)

	output_file(unsorted_save_folder + '/tuning.html')
	save(tuning_fig)

	#########################################################################################

######### ! Convert the matlab timestamp to actual time:
def matlab2datetime(matlab_datenum):
    # input matlab_datenum = e.g. 636150871373353768
    #matlab_datenum = int(matlab_datenum)
    matlab_datenum = matlab_datenum/(24*3600*1e7)+367
    
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    #print 'day = ', day
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    #print 'dayfrac = ', dayfrac
    return day + dayfrac


def get_kwik_spike_times(spikes_path):


	input_h5file = h5py.File(spikes_path,'a')


	#times = np.array(input_h5file['/event_types/TTL/events/time_samples']).astype(np.int64)


	### times are in samples here:
	spike_times = np.array(input_h5file['/channel_groups/0/spikes/time_samples']).astype(np.int64)

	input_h5file.close()    

	# before returning the times, let's convert them to seconds  

	return spike_times / 3e4

def get_spike_times(spikes_path):

	### path is the directory w/ the file name. 
	

	filename_idx = spikes_path.find('636')

	ephys_filename = spikes_path[filename_idx:filename_idx+18]

	print 'ephys_filename = ', ephys_filename

	spiketimes_file = open(spikes_path,"rb")

	print 'SpikeTimes file path = ', spikes_path

	spike_times = np.fromfile(spiketimes_file,dtype=np.uint64)
	
	spiketimes_file.close()

	print 'SpikeTimes file closed...'

	fs=30e3
	MSDNrate=1e7
	start_time = matlab2datetime(int(ephys_filename))
	print 'Ephys file start time = ', start_time

	#converted_spike_times = [thing*MSDNrate/fs + np.uint64(ephys_filename) for thing in spike_times]

	
	print 'starting to zip spike times:'
	#spike_seconds = np.array([(matlab2datetime(thing) - start_time).total_seconds() for thing in converted_spike_times]) # convert 636s to datetime objects
	#spike_seconds = np.array([(matlab2datetime(thing*MSDNrate/fs + np.uint64(ephys_filename)) - start_time).total_seconds() for thing in spike_times]) # convert 636s to datetime objects

	spike_seconds = spike_times / fs


	print 'shape of spike_seconds = ', spike_seconds.shape
	print spike_seconds[0:20]
	
	return spike_seconds

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

def windowed_histogram(trace,time,window_size,step): ## for spike histogram, trace should be a 0's and 1's vector with length = trial length (1's in places where spikes occur)
    out_x = []              ## time = vector in samples. 
    out_y = []

    fs = 3e4
    trace = np.int16(trace * fs)
    print trace[0]
    time = np.int16(time * fs)
    
    y = slidingWindow(trace,window_size,step) #window_size/5
    x = slidingWindow(time,window_size,step) ### get centers of the time
    for value in y:
        out_y.append(np.sum(value)) ## take the number of spikes in this window
        
    for t in x:
        out_x.append(np.median(t)) ## take the median time, i.e. the center of the time window
    
    return out_x,out_y
def convert2seconds_pandas(raw_times,ephys_filename):

	raw_times = pd.Series(data=raw_times)

	fs=30e3
	MSDNrate=1e7
	raw_times = (raw_times*MSDNrate/fs + np.uint64(ephys_filename))/(24*3600*1e7)+367

	days = raw_times.apply(lambda x: datetime.datetime.fromordinal(x.astype(int)))

	dayfrac = raw_times.apply(lambda y: datetime.timedelta(days=y%1) - datetime.timedelta(days = 366))



if __name__ == "__main__":


	ephys_path = sys.argv[1] #### the SpikeTimes file path, e.g. /Volumes/steffenwolff/GRat18/636152664381217973/    #####/ChGroup_3/SpikeTimes

	behavior_path = sys.argv[2] ### this is the 'oe_stim_times' pickle file
	
	stim_data = pd.read_pickle(behavior_path) # read the CSV file with stimulus times and orientations
	

	for ch in range(16):
		spikes_path = ephys_path + 'ChGroup_%d/SpikeTimes' % ch
		print 'spikes path = ', spikes_path
		channel_name = spikes_path[spikes_path.find('ChGroup_'):spikes_path.find('/SpikeTimes')] # e.g. ChGroup_3
		print 'channel_name = ', channel_name

		unsorted_save_folder = './Unsorted_PSTH/' + channel_name
		if not os.path.exists(unsorted_save_folder):
			os.makedirs(unsorted_save_folder)

		print 'Saving in folder ', unsorted_save_folder
	
	


		kwik = 0
		if kwik != 1:
			spike_times = get_spike_times(spikes_path) ### this path needs to have the 636 filename b/c it needs the file start time!
			plot_spike_psth(spike_times,stim_data,channel_name,unsorted_save_folder)

		else:
			spike_times = get_kwik_spike_times(spikes_path)

	
	
	

