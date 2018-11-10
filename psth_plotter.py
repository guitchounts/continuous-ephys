from get_files import get_files
import os
import h5py
import pandas as pd
import numpy as np
from ephys_condition_signal import filter, downsample, rectify,median_rejection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import gridspec
from scipy import stats
from simple_spikedetect import run_spike_detect
from bokeh.io import gridplot, output_file, show, save, vplot
from bokeh.plotting import figure
from bokeh.models import TapTool, HoverTool
from bokeh.colors import RGB
from bokeh.models import Range1d
from compiler.ast import flatten
from sliding_window import windowed_histogram

#1. collect time data from .raw.kwd files
def get_file_metadata(files):
	
	all_metadata = []
	
	for file in files:

		kwd_file = h5py.File(file,'r')
		metadata = {};
		#data['ephys'] = kwd_file['/recordings/0/data']

		metadata['sample_rate'] = kwd_file['/recordings/0/'].attrs['sample_rate'] # in samples (30e3)
		metadata['start_time'] = kwd_file['/recordings/0/'].attrs['start_time'] # in samples
		metadata['start_sample'] = kwd_file['/recordings/0/'].attrs['start_sample']
		metadata['data_len'] = kwd_file['/recordings/0/data'].shape[0]
		#data['length'] = len(data['ephys']) # this is in samples
		#stat = os.stat(raw_kwd) # os.path.getmtime(raw_kwd)
		all_metadata.append(metadata)
		kwd_file.close()
	print 'start time = '
	for meta in all_metadata:
		print meta['start_time'] #/meta['sample_rate']

	
	#global fs
	#fs = kwd_file['/recordings/0/'].attrs['sample_rate'] # in samples (30e3)
	
	return all_metadata

def kwd_indices(metadata,stim_data):  ### returns the indices of all exp files that have some stimuli in them
	# e.g. file id =  [0, 0, 0, 0, 0, 0, 0, 2, 3... 13, 14, 14, 14, 15] means stim[0] is in file 0; stim[-1] is in file 15 (after getting rid of stims that fall outside the ephys timerange). 
	# convert kwd data_len  
	global datalen_cumsum
	datalen_cumsum =  np.cumsum([meta['data_len'] for meta in metadata])
	
	file_id =[]
	stim_data_idx_tokeep = []
	for j,time in enumerate(stim_data.times):
		if time < datalen_cumsum[-1]: ## in the case that a given stim time > datalen_cumsum[-1] (i.e. it falls outside the range of all our files,)
			file_id.append(len(np.where(time > datalen_cumsum)[0]))
			#print time
			#### ALSO: get indices of stim_data that we should keep:
			stim_data_idx_tokeep.append(j)
		else:
			print 'Stim time sample %d appears to be greater than length of all files, %d. Skipping that trial.' % (time,datalen_cumsum[-1])
	#print file_id

	# convert stim times to samples:
	#stim_data.times = np.round(stim_data.times * metadata[0]['sample_rate'])
	#print stim_data.times

	#### the uniques of file_id are the files being used. use these indices to get rid of fields in ephys?
	print 'stim_data_idx_tokeep = ', stim_data_idx_tokeep
	return file_id,stim_data_idx_tokeep

def extract_chunks(stim_data,stim_files,channels,ephys):
	
	# take stim_data.times that correspond to a given experiment file, and extract time chunk from that file

	## make list of stimulus indices for each file. e.g. stim_indices[0]=[0,7] means stimulus 0-7 is contained in first file. 
	exp_file,idx = np.unique(stim_files,return_index=True)
	
	idx.sort()
	print 'stim_files =', stim_files
	print 'exp_file ==', exp_file
	print 'idx = ',idx
	
	stim_indices = []
	for i in range(len(idx)):
		
		if i+1 < len(idx):
			stim_indices.append(idx[i:i+2])
		else:
			stim_indices.append([idx[i],len(stim_files)])
	
	print 'stim_indices = ', stim_indices
	
	#### not all kwd files had stimuli in them. get indices for which files to look at, using exp_file to get 
	i = []
	for file in exp_file:
		#i.append(np.where(ephys['exp_name']==file))
		i.extend(np.where(ephys['exp_name']==file))
	print 'i before = ', i
	#i = [list(x) for x in i]
	#i = [x[0].tolist()[0] for x in i]  
	i = [x[0].tolist() for x in i]  
	i.sort()
	print 'i after = ', i 
	#### i = indices for files that have stimulis in them, out of all files 
	## e.g. i =[array([0]), array([2]), array([3])... array([14]), array([15])]
	# ephys[i] = just the files that contain stim times. 




	#tmp = ephys[i]
	#del ephys
	ephys = ephys[i]
	print 'ephys = ', ephys
	##### !!!!! use this opportunity to get rid of any fields in ephys that don't have stimuli in them. 

	for idx,item in enumerate(ephys):
		print 'EPHYS again!'
		print 'idx,item = ',idx,item


	### cycle through ephys and put in stim_indices to each row of ephys['stim_idx'] individually
	for j in range(len(exp_file)):
		ephys['stim_idx'][j] = stim_indices[j] ## where i = index, out of all files (19) where we should put things (just 15 files)
		print 'j, stim_indices[j] =====', j, ephys['stim_idx'][j]
		#ephys['stim_idx'][i] = stim_indices[i]
		#print 'exp_file :', exp_file, 'stim indices: ', stim_indices
		
		tmp_times = stim_data.times[ephys['stim_idx'][j][0]:ephys['stim_idx'][j][1]].values
		
		ephys['stim_times'][j] = tmp_times # 


	time_range = 0.5*fs
	#trial = np.zeros([time_range*2,len(stim_data)])
	
	###### GO THROUGH EACH EXPERIMENT FILE
	leftover_flag = 0
	rem_samples = 0
	leftunder_flag = 0
	curr_trial_samples = 0

	for i,file in enumerate(ephys['exp_name']):
		
		#temp_times = stim_data.times[stim_indices[i][0]:stim_indices[i][1]].values
	#	temp_times = stim_data.times[]
		kwd_file = h5py.File(file,'r')
		data = np.array(kwd_file['/recordings/0/data']) # = samples x channels
		print 'EXTRACTING CHUNKS FROM ', ephys['exp_name'][i]
		
		for ch in channels:
			data[:,ch] = filter(data[:,ch],[500, 12e3])
		


		

		if isinstance( ephys['stim_times'][i], ( int, long )) == False:
			######### check if ephys['stim_times'][i] is not an int. if it is, skip that file.

			exp_trials = np.zeros([time_range*2,len(ephys['stim_times'][i]),len(channels)]) #!!## make this shape samples x trials x channels
			

			for trial,j in enumerate(ephys['stim_times'][i]):
				## trial = index of trial in file; j = time of each stimulus in file
				print 'trial %d, stim time j = %d ' % (trial,j)
				#print 'x == ', j-time_range, j+time_range
				print 'stim times this trial: ', ephys['stim_times'][i][trial]
				#print 'stim times next trial: ', ephys['stim_times'][i][trial+1]

				### the time sample of stim in files 2+ is (e.g. 14,556,799) out of range of the file's length
				### need to ? subtract file's start time from index ?
				data_indices1 = int(j-time_range - ephys['start_time'][i])
				data_indices2 = int(j+time_range - ephys['start_time'][i])
				trialstart_time =  int(j-time_range)
				trialstop_time = int(j+time_range)
				print 'ephys[start_time][i] = ', ephys['start_time'][i]
				print 'ephys[stop_time][i] = ', ephys['stop_time'][i]
				print 'trialstart_time = ', trialstart_time
				print 'trialstop_time = ', trialstop_time

				
				## check that the trial end sample is lower the experiment's end sample. if not, get remaining samples from next experiment
				if trialstop_time <= ephys['stop_time'][i] and trialstart_time > ephys['start_time'][i]:
					print 'Trial %d is within the bounds of the experiment' % trial

					print 'data_indices2, ephys stop time = ', data_indices2,ephys['stop_time'][i]
					if len(channels) > 1:
						for ch_idx,ch in enumerate(channels):
							exp_trials[:,trial,ch_idx] = data[data_indices1:data_indices2,ch] #!!#### make exp_trials samples x trial x channels e.g. exp_trials[:,trial,channels], where channels = [1,2,3..64]
					elif len(channels) == 1:
						exp_trials[:,trial] = np.reshape(data[data_indices1:data_indices2,channels],(data[data_indices1:data_indices2,channels].shape[0],1))

					print 'exp_trials shape = ',exp_trials.shape
				
					if leftover_flag == 1: # the previous trial spilled over into this experiment. 
						if len(channels) > 1:
							for ch_idx,ch in enumerate(channels):
								exp_trials[rem_samples:,trial-1,ch_idx] = data[0:rem_samples,ch] #!!#### make exp_trials samples x trial x channels e.g. exp_trials[:,trial,channels], where channels = [1,2,3..64]
						elif len(channels) == 1:
							exp_trials[rem_samples:,trial-1] = np.reshape(data[0:rem_samples,channels],(data[0:rem_samples,channels].shape[0],1))


					if leftunder_flag == 1:
						if len(channels) > 1:
							for ch_idx,ch in enumerate(channels):
								exp_trials[curr_trial_samples:,trial,ch_idx] = data[0:time_range+curr_trial_samples,ch] #!!#### make exp_trials samples x trial x channels e.g. exp_trials[:,trial,channels], where channels = [1,2,3..64]
						elif len(channels) == 1:
							exp_trials[curr_trial_samples:,trial] = np.reshape(data[0:time_range+curr_trial_samples,channels],(data[0:time_range+curr_trial_samples,channels].shape[0],1))

					leftunder_flag=0
					curr_trial_samples=0
					leftover_flag=0
					rem_samples=0
					print 'leftover flag = ',leftover_flag
					print 'leftunder_flag  = ',leftunder_flag

				elif trial == len(ephys['stim_times'][i]) and trialstop_time >= ephys['stop_time'][i]:
					print 'Stim time %d on trial %d falls outside this experiments timerange' % (j,trial)
					print 'data_indices2, ephys stop time = ', data_indices2,ephys['stop_time'][i]
					rem_samples = data_indices2 - ephys['stop_time'][i]
					print 'rem_samples = ', rem_samples
					if len(channels) > 1:
						for ch_idx,ch in enumerate(channels):
							exp_trials[0:time_range*2-rem_samples,trial,ch_idx] = data[data_indices1:ephys['stop_time'][i],ch] #!!#### make exp_trials samples x trial x channels e.g. exp_trials[:,trial,channels], where channels = [1,2,3..64]
					elif len(channels) == 1:
						exp_trials[:,trial] = np.zeros(data[data_indices1:data_indices2,channels].shape[0],1)
						exp_trials[0:time_range*2-rem_samples,trial] = np.reshape(data[data_indices1:ephys['stop_time'][i],channels],(data[data_indices1:ephys['stop_time'][i],channels].shape[0],1))

					leftover_flag = 1
					print 'leftover flag = ',leftover_flag
					
				#elif # next trial's start time (data_indices1) < this experiment's stop time AND next trial's center time > this exp stop time
				# fill that trial's data with remaining indices of this experiment data
				#elif j = 0 and data_indices1 < 0:
					# get previous experiment file. 
				#	previous_kwd_file = h5py.File(ephys['exp_name'][i-1],'r')
				#	previous_data = np.array(previous_kwd_file['/recordings/0/data']) # = samples x channels
				#	print 'Going back to file ', ephys['exp_name'][i-1]
					
				#	for ch in channels:
				#		previous_data[:,ch] = filter(previous_data[:,ch],[500, 12e3])
					
				#	previous_kwd_file.close()

				   ####data_indices1 < ephys['start_time'][i]:
				#	print 'Stim time %d on trial %d happened before this experiments timerange' % (j,trial)
				#	print 'data_indices1, ephys start time = ', data_indices1,ephys['start_time'][i]
				elif trial == len(ephys['stim_times'][i]) and ephys['stim_times'][i+1][0]-time_range < ephys['stop_time'][i]:
					
					print 'STIM SPILLING OVER INTO NEXT EXP, j = %d, stim time i+1 [0]-time_range = %d' % (j,ephys['stim_times'][i+1][0]-time_range)
					# figure out how many samples are leftover here:
					curr_trial_samples = ephys['stop_time'][i] - ephys['stim_times'][i+1][0]-time_range
					


					if len(channels) > 1:
						exp_trials_next = np.zeros([time_range*2,len(ephys['stim_times'][i+1]),len(channels)])
						for ch_idx,ch in enumerate(channels):
							exp_trials_next[0:curr_trial_samples,0,ch_idx] = data[curr_trial_samples:,ch]
					elif len(channels) == 1:	
							exp_trials_next = np.zeros([time_range*2,len(ephys['stim_times'][i+1])])
							exp_trials_next[0:curr_trial_samples,0] = data[curr_trial_samples:,channels]
					# fill the i+1's data with the remaining chunk of this experiment:
					ephys['trial_data'][i+1] = exp_trials_next

					leftunder_flag = 1


			ephys['trial_data'][i] = exp_trials  #!!#### exp_trials is 2-D -- samples x channels?
			#tmp_data = np.vstack([tmp_data,data[ephys['stim_times'][i] channel] ])
			#print 'ephys stim times i = ', ephys['stim_times'][i]
			print 'shape exp_trials = ', exp_trials.shape
		

		elif isinstance( ephys['stim_times'][i], ( int, long )) == True:
			print 'Nothing in file %s. Skipping.' % ephys['exp_name'][i]
			pass

		kwd_file.close()
		data=None
		#print 'data shape = ', data.shape
	return ephys
		
		#print 'shape trial : ', trial.shape

		
		

		


	#for i,file in enumerate(exp_file):
	#	print 'File: ', file, '. Stim indices: ', stim_indices[i]
	#print exp_file, ' ',idx
	#for i,file in enumerate(exp_file):
		
		#print stim_data[idx[i:i+2]]  # the stim times that correspond to that file. stim_data[idx]
	#	print file,i

def plot_psth(ephys,stim_data,stim_data_idx_tokeep,channels):
	length = len(ephys['trial_data'][0])
	print 'length,fs = ', length,fs
	time = np.linspace(-0.5*length/fs,0.5*length/fs,length)
	#time = downsample(time, 10)
	#time -= time/2 ## make it -0.5 to +0.5 sec from stim onset. 
	print 'time === ', time
	print 'shape time ',time.shape
	#rand_dat = np.random.rand(len(ephys['trial_data']),100)

	#print 'shape of data = ', ephys['trial_data'].shape
	
	#for thing in ephys:
	
		#x=2
		
		#print type(thing['trial_data'])
		
	tmp = [item['trial_data'] for item in ephys]
	for item in tmp: 

		print 'shape of item in tmp = ',item.shape ##!!!! should be samples x trials x channels, e.g. (30e3,7,64)
	data = np.concatenate(tmp,axis=1)
	#data = rectify(data)
	#data = downsample(data,10)
	print 'shape data = ', data.shape
	
	###### call parts of data by the stimulus identity (i.e. grating orientation) or index within stim_data.orientations
	# get indices of same orientations:


	### !!!! stim_data still contains elements that we're getting rid of b/c they didn't coincide with proper times in the ephys. Get rid of these?

	print 'stim_data_idx_tokeep = ', stim_data_idx_tokeep
	print 'shape stim_data = ', stim_data.shape
	#stim_data.orientations = stim_data.orientations[stim_data_idx_tokeep]
	#stim_data.times = stim_data.times[stim_data_idx_tokeep]
	tmp_stim_data = stim_data.iloc[stim_data_idx_tokeep]
	print tmp_stim_data

	stim_orientation,idx,inv = np.unique(tmp_stim_data.orientations,return_index=True,return_inverse=True)
	## inv = elements in stim_data.orientations corresponding to the uniques in stim_orientation
	## e.g. stim_data.orientations = [18 162 0 18...]
	#			 stim_orientation = [0 18 36 54...]
	##      				  inv = [1 9 0 1 ...]

	#for orientation in stim_orientation: ## take individual orientations
		## from data, (shaped [30e3,128] - samples x trials), get trial indices for each orientation and plot those separately
			
		#		stim_indices.append(idx[i])
	print 'stim_orientation,idx,inv = ',stim_orientation,idx,inv

	#print stim_data.orientations[stim_data.orientations==18].values
	stim_dict = dict()
	
	psth_folder = './psth/'
	
	for ch_idx,ch in enumerate(channels):
		##! check if directory ./psth exists; if not, create it
		save_folder = psth_folder + 'channel_' + str(ch+1)
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)

			##!! create directory save_dir = ./psth/ch

		for ori in stim_orientation:	
			stim_dict[str(ori)] = tmp_stim_data.orientations[tmp_stim_data.orientations==ori].index.values ### put indices of stim_data that correspond to this orientation, from inv
		
			print 'Plotting channel %d, orientation %d' % (ch+1,ori)
			
			data_to_plot = downsample(rectify(data[:,stim_dict[str(ori)],ch_idx]),10).T

			print 'Shape of plot data = ', data_to_plot.shape
			plot_data = pd.DataFrame(data=data_to_plot,
			        	columns = downsample(time,10))
			#img = mpimg.imread('stinkbug.png')     

			fig = plt.figure(figsize=(20, 10)) 
			gs = gridspec.GridSpec(2, 1, width_ratios=[1, 1]) 

			ax1 = plt.subplot(gs[0])
			ax1 = sns.heatmap(plot_data,robust=True,xticklabels=1000,cbar=False) #plt.subplot(gs[1])
			ax1.set_title('PSTH for orientation '+str(ori))

			
			ax2 = sns.set_style("white",{'axes.linewidth' : 0.01})
			ax2 = plt.subplot(gs[1])
			#ax2 = ax1.twinx()
			ax2 = sns.tsplot(stats.trim_mean(data_to_plot,proportiontocut=0.25,axis=0),time=downsample(time,10),value='Voltage (uV)',color='black')


			#plt.show()
			#fig.savefig((save_folder + '/' + str(ori) +".pdf"))

			plot_raw_traces(data[:,stim_dict[str(ori)],ch_idx],time,ori,ch,save_folder)

def plot_raw_traces(traces,time,stim_name,ch,save_folder):
	print 'shape traces[:,0] = ', traces[:,0].shape
	print 'time shape = ', time.shape
	
	#################### BOKEH FIGURE #########################
	fig = plt.figure(figsize=(20, 10)) 
	num_trials = traces.shape[1]
	gs = gridspec.GridSpec(num_trials, 1, width_ratios=[1, 1]) ## a subplot for every trial
	ax = range(num_trials)
	all_spike_times = []
	raster = figure(width=1000, height=400,y_axis_label='Trial Number',title='Raster + Histogram Channel %d, Orientation %d' % (ch+1,stim_name))
	spike_vec = np.zeros([len(time),1])
	
	time_vec = np.linspace(0,len(time),len(time))

	for trial in range(num_trials):
		print 'trial # ',trial
		peaks,times,ifr_vec = run_spike_detect(traces[:,trial])
		all_spike_times.append(times)
		spike_vec[times] += 1
		trial_time = [t/30e3 - 0.5 for t in times]
		#ax = plt.subplot(gs[trial])
		#sns.tsplot(downsample(traces[:,trial],10),time=time,value='Voltage (uV)',color='black',linewidth=0.1)
		ax[trial] = figure(width=1000, plot_height=500)
		#s1 = figure(width=1000, plot_height=500, title='Spikes')
		ax[trial].line(time,traces[:,trial]) ## (time is already downsampled)
		ax[trial].circle(trial_time,peaks,color='red') ## convert to seconds and subtract 0.5 b/c plotting data on time from -0.5 to +0.5 seconds
		
		#ax.set_ylim([-1000,1000])
		#ax.set(y_range=Range1d(-1000, 1000))
		#axes.append[ax]
		

    
		raster.segment(x0=trial_time, y0=np.repeat(trial,len(times)), x1=trial_time,
			y1=np.repeat(trial+1,len(times)), color="black", line_width=0.5)

	p = gridplot([[s1] for s1 in ax]) #gridplot([[s1] for s1 in axes])

	#fig.savefig(save_folder + '/raw_psth_'+str(stim_name)+'.pdf')

	output_file(save_folder + '/raw_psth_'+str(stim_name)+'.html')
    # show the results
	save(p)
	############## SPIKE HISTOGRAM FIGURE ############################ 
	histo_fig = figure(width=1000, plot_height=500,y_axis_label='Firing Rate (Hz)',x_axis_label='Time (sec)',x_range=raster.x_range)
	print 'len all_spike_times = ', len(all_spike_times)
	num_bins = 50
	hist, edges = np.histogram(flatten(all_spike_times), bins=num_bins)
	bin_width = np.diff(edges)[0]/fs # in seconds.
	edges = edges/fs - 0.5 ## plot x-axis in seconds.
	histo_fig.quad(top=hist/bin_width/num_trials, bottom=0, left=edges[:-1], right=edges[1:], ## hist/bin_width = firing rate in Hz
        fill_color="#036564", line_color="#033649")


	
	#time
	# pass this to the sliding window - get sum of spikes in each 100ms window.
	win_size = 2**8
	win_step = 2**2
	win_x,windowd_spike_vec = windowed_histogram(spike_vec,time_vec,win_size,win_step)
	
	histo_fig.line([t/fs -0.5 for t in win_x],[w/win_size*fs/num_trials for w in windowd_spike_vec],color='magenta')

	output_file(save_folder + '/spike_histogram_'+str(stim_name)+'.html')
	grid = gridplot([[raster], [histo_fig]])

	save(grid)
	############# SEABORN FIGURE ###############
	#fig = plt.figure(figsize=(20, 10)) 
	#num_trials = traces.shape[1]
	#gs = gridspec.GridSpec(num_trials, 1, width_ratios=[1, 1]) ## a subplot for every trial
	#sns.set_style("white",{'axes.linewidth' : 0.01})
	#for trial in range(num_trials):
	#	print trial
	#	ax = plt.subplot(gs[trial])
	#	sns.tsplot(downsample(traces[:,trial],10),time=time,value='Voltage (uV)',color='black',linewidth=0.1)
		#s1.circle([t/30e3 for t in times],peaks,color='red')

	#	ax.set_ylim([-1000,1000])
	#sns.despine()
	#fig.savefig(save_folder + '/raw_psth_'+str(stim_name)+'.pdf')

if __name__ == "__main__":
	global fs
	fs = 30e3

	
	#data = np.random.rand(18e4,10)
	#length = len(data)
	#time = np.linspace(-0.5*length/fs,0.5*length/fs,length)
	#time = downsample(time, 10)
	#plot_raw_traces(data,time,18,12,'./psth/channel_13')





	raw_files = get_files('kwd',os.getcwd()) # collect the kwd files 
	print 'raw files = ', raw_files
	metadata = get_file_metadata(raw_files) # get their lengths + timestamps

	stim_data = pd.read_csv('oe_stim_times.csv') # read the CSV file with stimulus times and orientations
	stim_data.times = np.rint(stim_data.times * metadata[0]['sample_rate'])
	#print metadata[0]['sample_rate']
	file_id,stim_data_idx_tokeep = kwd_indices(metadata,stim_data)

	stim_files = [raw_files[file] for file in file_id]  ## the uniques of stim_files can be used to key ephys['exp_name'] - to avoid putting things in exp files that didn't have stimuli. 
	
	######### !!!!! channel numbering starts with 0 !!!!!!!!!###############
	channels =  np.arange(64) # np.array([12,13])  #

	
	# use stim_data.times and file_id to extract relevant chunks from KWD files

	################ MAKE EPHYS DATA STRUCTURE ################################
	trial_size = fs # in samples...
	dt = np.dtype([('exp_name','a26'),('exp_length',np.int64),('start_time',np.int64),('stop_time',np.int64),('stim_idx',np.int64,2),('stim_times',list),('trial_data',list)]) #(len(stim_data),trial_size)
	#### 'exp_name' = name of kwd file, e.g. 'experiment68_100.raw.kwd'
	#### 'exp_length' = # of samples in the kwd file (e.g. 9e6 for 5-min chunk at 30kHz)
	#### 'start_time' = sample number for this file's start. =0 for first file; = 1st file' exp length for second file, etc. 
	#### 'stop_time' = sample number at which this experiment ends. = start_time + exp_length
	#### 'stim_idx' = start-stop indices of behavior stimuli file. e.g.[0,7] = this kwd experiment file contains stimuli 0 through 7 
	#### 'trial_data' = [stim_idx[1]-stim_idx[0]] x length of trial (e.g. 1 sec). e.g. 7x30e3 if this file contains 7 stimuli. 
	

	ephys = np.zeros(len(raw_files),dtype=dt)
	ephys['exp_name'] = raw_files
	
	lengths = [meta['data_len'] for meta in metadata]
	lengths_cumsum = np.cumsum(lengths)
	starts = np.array([lengths_cumsum[0:-1]]) 
	starts = np.insert(starts,0,0)
	stops = starts + lengths 

	ephys['exp_length'] = lengths
	ephys['start_time'] = starts 	###  [0,  exp_len[0], exp_len[1] .... exp_len[-2]]
	ephys['stop_time'] =  lengths_cumsum		### [cumsum(exp_len)]
	
	for idx,item in enumerate(ephys):
		print 'EPHYS!'
		print 'idx,item = ',idx,item
	
	ephys = extract_chunks(stim_data,stim_files,channels,ephys)

	#ephys = {'experiment':raw_files,'lengths':[meta['data_len'] for meta in metadata]}
	plot_psth(ephys,stim_data,stim_data_idx_tokeep,channels)
	#for thing in range(len(ephys)):
	#	ephys['stim_times'][thing] = [1*thing,2,3]

	print 'shapes of trial data:'
	for item in ephys:
		if isinstance(item['stim_times'], ( int, long )) == False:
			print item['trial_data'].shape
	
	

