#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bokeh.plotting import figure,show
from bokeh.io import output_notebook, gridplot, output_file, show, save
import struct
import datetime
import pandas as pd
from scipy import stats,signal
from matplotlib import gridspec
import os,sys
sns.set_style('white')

def filter(ephys,freq_range,filt_order = 2,ripple = 0.2,attenuation = 40,filt_type='bandpass',fs=30e3):
	
    # design Elliptic filter:

    [b,a] = signal.ellip(filt_order,ripple,attenuation,[x/(fs/2) for x in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace


def plot_lfp_psth(lfp,stim_data,channel_name='1',lfp_save_folder='yes'):


	################################ get the TRIAL TIMES  ###################################
	#########################################################################################

	trials = range(len(stim_data))
	stim_orientation,idx,inv = np.unique(stim_data.orientations,return_index=True,return_inverse=True)
	stim_dict = dict()
	for ori in stim_orientation:	
	    stim_dict[str(ori)] = stim_data.orientations[stim_data.orientations==ori].index.values ### put indices of stim_data that correspond to this orientation, from inv
	    #### ^ stim_dict gives indices of stim_data for given orientations. 

	trial_range_before = 0.2
	trial_range_after = 0.8    
    
	trial_time_vector = np.linspace(-trial_range_before,trial_range_after,300) ## 300=trial length, in LFP sample (LFP fs=300 samples/sec)

	lfp_time = np.linspace(0,len(lfp)/300,len(lfp)) # time for the whole trace! 



	###### Gather LFP data from behavior trials:
	trial_lfps = []

	for i in trials:
		#print 'i = ', i
		offset = 0.1 # 100ms offset for the LFP files? according to Ashesh        
		trial_time = stim_data.times[i] - offset

		#print 'trial_time = ', trial_time

		temp_idx = (lfp_time < trial_time+trial_range_after) & (lfp_time > trial_time - trial_range_before)

		#print 'temp_idx = ', temp_idx
		temp_trial = lfp[temp_idx] #- trial_time

		#print 'temp_trial[0:10] = ', temp_trial[0:10]
	
		trial_lfps.append(temp_trial)	

	#print 'length of trial_lfps = ', len(trial_lfps)


	stim_data['lfps'] = trial_lfps


	########################### MAKE LFP trials x samples MATRIX ############################
	#########################################################################################

	#conditioned_lfp = np.zeros([len(trials),300])
	#for trial in trials:
		#print 'len of trial_lfps[trial] = ', len(trial_lfps[trial])

		#conditioned_lfp[trial,:] = signal.detrend(trial_lfps[trial])
	conditioned_lfp = np.array(trial_lfps)

	

	###################################### PLOTTING SCRIPT ##################################
	#########################################################################################
	fig = plt.figure(figsize=(10, 10),dpi=600) 
	num_figs = 10
	gs = gridspec.GridSpec(num_figs, 1, width_ratios=[1, 1])  # 10 subplots - for each orientation

	correct_sorting = ['0','18', '36', '54', '72', '90', '108', '126', '144', '162']

	ax = range(num_figs)

	for idx,ori in enumerate(stim_dict):
	       
		#print idx, correct_sorting[idx]
		ax[idx] = plt.subplot(gs[idx])
		ax[idx].set_ylabel(correct_sorting[idx])
		#ax[idx].tick_params(axis="x", which="major", length=10)
		if idx == num_figs-1:
			ax[idx].set_xlabel('Time (seconds)')            
		else:
			ax[idx].axes.xaxis.set_ticklabels([])

		for ori_trials in range(len(stim_dict[correct_sorting[idx]])):
			#print 'ori_trials = ', ori_trials
			ax[idx] = plt.plot(trial_time_vector,conditioned_lfp[ori_trials,:],alpha=0.25,lw=0.25,color='black')

		ax[idx][0].axes.yaxis.set_ticklabels([])
		ax[idx] = plt.plot(trial_time_vector,np.mean(conditioned_lfp[stim_dict[correct_sorting[idx]],:],axis=0),lw=2,color='#c53aff' )
		patch_height = ax[idx][0].axes.get_ylim()
		ax[idx][0].axes.add_patch(mpatches.Rectangle(
        (0.0,patch_height[0]),   # (x,y)
        0.2, patch_height[1],color='#bcffbc' ,alpha=.75    ))

	#ax = ax.sort()
	sns.despine(left=True,bottom=True)
	fig.savefig((lfp_save_folder + '/Ch_' + str(channel_name) + '.pdf'))

	
	
	#########################################################################################

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



		if ch == 0:
			all_lfps = lfp
		else:
			all_lfps = np.vstack([all_lfps, lfp])

	print 'shape of all_lfps = ', all_lfps.shape
	return all_lfps


if __name__ == "__main__":


	lfp_path = sys.argv[1] #### the LFP path, e.g. /Volumes/steffenwolff/GRat18/636152664381217973/LFP/

	if lfp_path[-1] != '/': ### check that the last character in the path is a '/'
		lfp_path += '/'

	behavior_path = sys.argv[2] ### this is the 'oe_stim_times.csv' file

	channel_name = range(16)   #lfp_path[lfp_path.find('Ch_'):]


	lfp = get_lfp(lfp_path + 'Ch_',channel_name)

	

	stim_data = pd.read_pickle(behavior_path) # read the CSV or pickle file with stimulus times and orientations

	lfp_save_folder = './LFP_PSTH'
	if not os.path.exists(lfp_save_folder):
	    os.makedirs(lfp_save_folder)

	for idx,ch in enumerate(channel_name):

		plot_lfp_psth(lfp[idx,:],stim_data,ch,lfp_save_folder)


