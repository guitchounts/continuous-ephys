#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure,show
from bokeh.io import output_notebook, output_file, show, save #, vplot # gridplot
from bokeh.layouts import column
from bokeh import palettes
import struct
import datetime
import pandas as pd
from scipy import stats,signal,ndimage,interpolate
from matplotlib import gridspec
import os,sys
from numpy import matlib
fs=30e3

def filter(ephys,freq_range,filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='bandpass',fs=30e3):
	
    # design Elliptic filter:

    [b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace


def plot_spike_psth(cluster_data,stim_data,channel_name,sorted_save_folder):

	# spike_times,stim_data,channel_name,unsorted_save_folder
	
	colors = palettes.Spectral10
	trials = range(len(stim_data))
	stim_orientation,idx,inv = np.unique(stim_data.orientations,return_index=True,return_inverse=True)
	stim_dict = dict()
	for ori in stim_orientation:	
	    stim_dict[str(ori)] = stim_data.orientations[stim_data.orientations==ori].index.values ### put indices of stim_data that correspond to this orientation, from inv
	    #### ^ stim_dict gives indices of stim_data for given orientations. 

	trial_range = 1# 0.5
	#trial_time_vector = np.linspace(-trial_range,trial_range,300) ## 300=trial length, in LFP sample (LFP fs=300 samples/sec)

	if any('clusters' in s for s in list(cluster_data.columns.values)):
		clusters = range(cluster_data.clusters.astype(int).max()+1)
	else:
		clusters = range(len(cluster_data.spikes[0]))


	if not 'spikes' in stim_data.columns:
		print '################################ Finding Spikes ################################'
		################################ get the TRIAL TIMES  ###################################
		#########################################################################################

		

		###### Gather spike data from behavior trials:
		spikes = []

		for i in trials:
			print 'i = ', i

			trial_time = stim_data.times[i]
			#print 'trial_time = ', trial_time
			
			trial_spiketimes = []
			
			for clust in clusters:
				## collect spikes
				
				temp_idx = (cluster_data.times[cluster_data.clusters==clust] < trial_time+trial_range) & (cluster_data.times[cluster_data.clusters==clust] > trial_time - trial_range)
				#print 'type(temp_idx) = ', type(temp_idx)
				#print 'temp_idx = ', temp_idx
				clust_trial_times = cluster_data.times[temp_idx[temp_idx==True].index.values].values - trial_time	
				
				trial_spiketimes.append(clust_trial_times)

			spikes.append(trial_spiketimes)
			#print 'temp_idx = ', temp_idx
			print 'Found %s spikes on trial %d' % ([str(len(x)) for x in trial_spiketimes],i)


			#spikes.append(temp_trial)

		

		stim_data['spikes'] = spikes

		stim_data.to_pickle('stim_data_spikes_' + channel_name)

		print 'STIM_DATA = ', stim_data

	else:
		print '################################ Spikes Already in Stim Data ################################'
		spikes = stim_data['spikes']
	
	win_size = 0.100 # ms
	win_step = 0.010 # ms
		
	for clust in clusters:
		print 'Processing cluster %d' % clust
		tuning_dict = dict()
		firingrate_lines = figure(width=1000, plot_height=500,y_axis_label='Prestimulus Mean-Subtracted Firing Rate (Hz)',x_axis_label='Time (sec)')
		
		for ori_idx,ori in enumerate(stim_dict):
	    
			

			print stim_data.spikes[stim_dict[ori]]

			raster = figure(width=1000, height=400,y_axis_label='Trial Number',title='Raster')
			
			
			num_trials = len(stim_data.orientations[stim_dict[ori]]) #  len(trials) ##### ! was stim_data.spikes
			trials_by_ori = range(num_trials)		
			bin_time = 0.1 # 25 ms bins
			num_bins = np.int(trial_range*2 / bin_time)
			windowed_timevec = np.arange(0,trial_range*2*fs,1)
			time_vec = np.linspace(-trial_range,trial_range,num_bins)
			tuning_time = 0.05 # sec. window in which to look for response modulation, before and after stim onset. 
			alltrials_modulation = []
			spike_vec = np.zeros([num_trials,len(windowed_timevec)]) # shape = num trials * 3e4
			for idx,trial in enumerate(trials_by_ori):
				
				print 'stim_dict[ori] = ', stim_dict[ori]
				print 'trial = ', trial 
				print 'stim_data.spikes[stim_dict[ori][trial]] = ', stim_data.spikes[stim_dict[ori][trial]]
				print 'len(stim_data.spikes[stim_dict[ori][trial]]) = ', len(stim_data.spikes[stim_dict[ori][trial]])
				

				trial_time = stim_data.spikes[stim_dict[ori][trial]][clust] ## spike times on this trial
				## if you pre-computed the relevant spike times and used a different trial range, you'll have extra or fewer spikes. Get rid of extra ones here:
				mask = np.ones(len(trial_time),dtype=bool)
				mask[[trial_time>trial_range]] = False
				mask[[trial_time<-trial_range]] = False
				trial_time = trial_time[mask]
				## figure out what to do if old trial range is smaller than one you want now.... 
				

				#spike_vec[idx,np.int64(np.floor((trial_time+trial_range)*fs))] += 1
				
				print 'trial_time = ', trial_time
				print 'type(trial_time) = ', type(trial_time)
				print ' trial_time = ', trial_time
				num_spikes_ontrial = len(trial_time)


				print 'trial %d (time=%d) has %d spikes for cluster %d' % (trial,stim_data.times[stim_dict[ori][trial]], num_spikes_ontrial,clust)

				
				#win_x,trial_windowed_spikes = windowed_histogram(spike_vec[idx,:],windowed_timevec,win_size,win_step)
				win_x,trial_windowed_spikes = windowed_mean(trial_time,trial_range,win_size,win_step)
				#print '!!!!!!!!!!!!! len(trial_windowed_spikes = ', len(trial_windowed_spikes)

				if idx==0:

					all_windowed_spikes = trial_windowed_spikes
				else:
					all_windowed_spikes = np.vstack([all_windowed_spikes, trial_windowed_spikes])


				####### MAKE RASTER
				raster.segment(x0=trial_time, y0=np.repeat(trial,num_spikes_ontrial), x1=trial_time,
			                   y1=np.repeat(trial+1,num_spikes_ontrial), color="black", line_width=0.5)

				#### get histogram for each trial:
				print 'num_bins = ', num_bins
				print 'type(num_bins) = ', type(num_bins)
				print 'trial_time = ', trial_time
				temp_hist,edges = np.histogram(trial_time, bins=num_bins,range=[-trial_range,trial_range])


				### Let's count the spikes in the 0.05-0.1s window and compare that to spikes in the -0.05-0.0s window:
				#spikes_after_stim = temp_hist[(edges<0.1) & (edges>=0.0)]
				#spikes_before_stim = temp_hist[(edges<0) & (edges>-0.1)]

				spikes_after_stim = trial_windowed_spikes[(win_x<0.2) & (win_x>=0.0)]
				spikes_before_stim = trial_windowed_spikes[(win_x<0) & (win_x>-0.2)]
				print 'spikes_after_stim = ', spikes_after_stim
				print 'spikes_before_stim = ', spikes_before_stim

				trial_modulation = np.sum(spikes_after_stim) - np.sum(spikes_before_stim) ## total spikes in 100ms after - total spikes in 100ms before stim.
				alltrials_modulation.append(trial_modulation/tuning_time) ## divide by 0.1 to get #spikes/second

				if trial == 0:
					raw_hist = temp_hist	
				else:
					raw_hist = np.vstack([raw_hist,temp_hist])


			#print 'length of all_spike_times = ', len(all_spike_times)
			print 'shape of overall histo = ', raw_hist.shape

			histo_fig = figure(width=1000, plot_height=500,y_axis_label='Firing Rate (Hz)',x_axis_label='Time (sec)',x_range=raster.x_range)


			
			#all_spike_times = np.concatenate(all_spike_times).ravel() ## make one long vector of spike times. 

			#hist, edges = np.histogram(all_spike_times, bins=num_bins)
			
			hist = np.mean(raw_hist,axis=0)
			hist_std = stats.sem(raw_hist,axis=0) # was: np.std
			print 'shape of averaged hist = ', hist.shape

			bin_width = np.diff(edges)[0] # in seconds.

			############# interpolate the line: ############
			print 'hist.shape,edges[:-1].shape = ', hist.shape,edges[:-1].shape
			f = interpolate.interp1d(edges[:-1], hist,kind='cubic')
			interp_x = np.linspace(edges[0],edges[-2],num_bins*10)
			interp_y_mean = f(interp_x)
			print 'interp_y_mean = ', interp_y_mean

			f_std_plus = interpolate.interp1d(edges[:-1],hist+hist_std,kind='cubic')

			f_std_minus = interpolate.interp1d(edges[:-1],hist-hist_std,kind='cubic')

			interp_y_std_plus = f_std_plus(interp_x)
			interp_y_std_minus = f_std_minus(interp_x)

			print 'hist, edges: ', hist,edges

			print 'bin_width = ', bin_width

			print 'num_trials = ', num_trials

			print 'shape of hist, edges = ', hist.shape,edges.shape



			histo_fig.quad(top=hist/bin_width, bottom=0, left=edges[:-1], right=edges[1:], ## hist/bin_width = firing rate in Hz
			    fill_color="#036564", line_color="#036564") #033649

			#histo_fig.line(edges[:-1]+bin_width/2, hist/bin_width)		
			### get the number of spikes in the 100ms before and after stim presentation:

			#windowed_line_mean = [w/win_size*fs/num_trials/bin_width for w in np.mean(all_windowed_spikes,axis=0)]
			windowed_line_mean = [w/bin_width for w in np.mean(all_windowed_spikes,axis=0)]

			#windowed_line_mean = ndimage.filters.gaussian_filter1d(windowed_line_mean,sigma=15)

			windowed_line_sterr = [w/bin_width for w in stats.sem(all_windowed_spikes,axis=0)] #/win_size*fs/num_trials

			#windowed_time = [t/fs-trial_range for t in win_x]
			windowed_time = [t for t in win_x]
			histo_fig.line(windowed_time,windowed_line_mean,color='magenta')

			

			tuning_dict[ori] = (np.mean(alltrials_modulation),stats.sem(alltrials_modulation))   ## (np.mean(hist[20:24])/np.mean(hist[16:20]))/bin_width/num_trials

			print 'tuning dict = ', tuning_dict

			print 'np.mean(alltrials_modulation) = ', np.mean(alltrials_modulation)
			print 'np.std(alltrials_modulation) = ', np.std(alltrials_modulation)


			#################################### instead of histogram, plot mean+std of numbers of spikes on each trial (binned).
			firingrate_line = figure(width=1000, plot_height=500,y_axis_label='Firing Rate (Hz)',x_axis_label='Time (sec)',x_range=raster.x_range)
			
			#hist_smooth = ndimage.filters.gaussian_filter1d(hist/bin_width,sigma=0.5)

			#
			

			
			#firingrate_line.line(interp_x,interp_y_mean/bin_width)
			firingrate_line.line(windowed_time,windowed_line_mean)

			#firingrate_lines.line(interp_x,(interp_y_mean-np.mean(interp_y_mean[np.where(interp_x<=0)]))/bin_width,color=colors[ori_idx],legend=ori)
			firingrate_lines.line(windowed_time,windowed_line_mean,color=colors[ori_idx],legend=ori)
			

			#std_x = (interp_x).tolist() # edges[:-1]+bin_width/2
			#std_y1 = (interp_y_std_plus).tolist() # -hist_std+hist/bin_width
			#std_y2 = (interp_y_std_minus).tolist()
			print len(windowed_time)
			print windowed_line_mean
			print windowed_line_sterr
			std_x = windowed_time
			std_y1 = (np.array(windowed_line_mean)+np.array(windowed_line_sterr)).tolist()
			std_y2 = (np.array(windowed_line_mean)-np.array(windowed_line_sterr)).tolist()


			std_x.extend(np.flipud(std_x).tolist())
			std_y1.extend(np.flipud(std_y2).tolist())

			firingrate_line.patch(std_x,std_y1,color="#99d8c9",alpha=0.5)
			#firingrate_line.patch(edges[:-1]+bin_width/2,hist_std+hist/bin_width)
			#firingrate_line.patch(edges[:-1]+bin_width/2,hist/bin_width+hist_std)

			
			all_spike_times = None 

			#histo_fig.line(win_x,[w/win_size/len(stim_dict[ori]) for w in windowd_spike_vec],color='magenta')
			

			save_folder = sorted_save_folder + '/Cluster_' + str(clust)
			if not os.path.exists(save_folder):
				os.makedirs(save_folder)
			output_file(save_folder + '/spike_histogram_'+str(ori)+'.html')
			#grid = gridplot([[raster], [histo_fig],[firingrate_line]]) #,[firingrate_line]
			grid = column([[raster], [histo_fig],[firingrate_line]]) #,[firingrate_line]
			
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


		#grid2 = gridplot([[tuning_fig],[firingrate_lines]]) #,[firingrate_line]
		grid2 = column([[tuning_fig],[firingrate_lines]]) #,[firingrate_line]


		output_file(save_folder + '/tuning.html')
		save(grid2)

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

def windowed_mean(trial_times,trial_range,win_size,step_size):

	

	#trial_range = 1.0
	#win_size = 0.1
	#step_size = 0.01
	time = np.linspace(-trial_range,trial_range,trial_range*2/win_size+1)
	num_chunks = ((trial_range*2-win_size)/step_size)+1
	
	spike_count = []
	win_time = []

	for i in range(np.int(np.ceil(num_chunks))): 
		spike_times = trial_times[(trial_times > (i)*step_size-trial_range) & (trial_times < (i)*step_size+win_size-trial_range)]
		spike_count.append(len(spike_times))
		win_time.append(((i)*step_size-trial_range+(i)*step_size+win_size-trial_range)/2)
		

	return win_time,spike_count

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
    #trace = np.int16(trace * fs)
    print trace[0]
    #time = np.int16(time * fs)
    
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


	spikes_path = sys.argv[1] # CSV with cluster assignments and spiketimes
	behavior_path = sys.argv[2] ### this is the 'oe_stim_times.csv' file OR

	channel_name = spikes_path[spikes_path.find('ChGroup_'):spikes_path.find('/clu')] # e.g. ChGroup_3
	#channel_name = 'ChGroup_0'
	print 'channel_name = ', channel_name

	sorted_save_folder = './Sorted_PSTH/' + channel_name
	if not os.path.exists(sorted_save_folder):
	    os.makedirs(sorted_save_folder)

	print 'Saving in folder ', sorted_save_folder

	
	if spikes_path.find('csv') == -1:
		cluster_data = pd.read_pickle(spikes_path)
	else:
		cluster_data = pd.read_csv(spikes_path)
	
	stim_data = pd.read_csv(behavior_path) # read the CSV file with stimulus times and orientations

	
	
	plot_spike_psth(cluster_data,stim_data,channel_name,sorted_save_folder)


