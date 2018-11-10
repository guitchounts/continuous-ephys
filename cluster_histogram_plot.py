import scipy.io as sio
from bokeh.io import gridplot, output_notebook, show, vplot
from bokeh.plotting import figure
from bokeh.models import TapTool, HoverTool
from bokeh.colors import RGB
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os,sys
from get_files import get_files

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
	ax = ax if ax is not None else plt.gca()
	if color is None:
		color = ax._get_lines.prop_cycler.next() ## ax._get_lines.color_cycle.next() -- that's deprecated 
	if np.isscalar(yerr) or len(yerr) == len(y):
		ymin = y - yerr
		ymax = y + yerr
	elif len(yerr) == 2:
		ymin, ymax = yerr
	
	ax.plot(x, y, color=color)
	ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
	

def plot_waveforms(spikes,clust_name):
	

	fig = plt.figure(figsize=(10, 10)) 
	gs = gridspec.GridSpec(2, 1, width_ratios=[1, 1]) 
	ax1 = plt.subplot(gs[0])
	


	storewindows = spikes[clust_name]['windows'][0][0][0][0] # storewindows for unsorted spikes
	window_time =    spikes[clust_name]['window_time'][0][0] # for unsorted spikes (e.g. 1:241)
	time_vec = range(len(window_time[:,0])) #  window_time[:,0] for unsorted spikes ### storewindows


	print 'type storewindows = ', type(storewindows)
	print 'shape storewindows: ', storewindows.shape
	win_avg = np.mean(storewindows,axis=1)
	win_std = np.std(storewindows,axis=1)
	
	ax1 = errorfill(time_vec,win_avg,win_std)
	
	ax1 = sns.set_style("white")
	sns.despine()
	#plt.show()
	
	print '#################### PRINTING SPIKE INFO ####################'
	print 'times: ', spikes[clust_name]['times'][0][0]
	print 'trials: ', spikes[clust_name]['trials'][0][0]


	voltmin = np.percentile(storewindows,1)-10
	voltmax = np.percentile(storewindows,99)+10
	xedges = np.arange(0.5,len(time_vec)+0.5,1)
	yedges = np.linspace(voltmin,voltmax,200)


	coordmat =  np.ravel(storewindows.T)
	coord_time = np.tile(range(len(time_vec)),storewindows.shape[1])

	density,xedges,yedges = np.histogram2d(y=coord_time,x=coordmat,bins=(yedges,xedges))
	

	ax2 = plt.subplot(gs[1],sharex=ax1) 
	ax2 = plt.imshow(density,cmap='gnuplot2',origin='lower')
	
	fig.tight_layout()

	fig.savefig("spikes.pdf")

def spike_trials(spikes,clust_name,experiment_lengths):

	all_exps = spikes[clust_name]['trials'][0][0][0][0][0]
	num_exps = max(all_exps)
	spike_times = spikes[clust_name]['times'][0][0][0][0][0]
	print 'all_exps = ', all_exps
	print 'num exps = ', num_exps
	print 'len of spike_times = ', len(spike_times)

	[uni_exps, uni_exps_indices] = np.unique(all_exps,return_index=1) # uni_exps_indices = e.g. [0  2730  4561...]
	
	
	for idx,unique_exp in enumerate(uni_exps):
		if idx>0:
			spike_times[all_exps==unique_exp] += experiment_lengths[idx]
    


	print 'spikes shape = ',spikes[clust_name]['times'][0][0][0][0].shape
	print 'uni times = ', spikes[clust_name]['times'][0][0][0][0][0,uni_exps_indices]
	print 'unique exp nums = ', uni_exps_indices
	

	return spike_times

if __name__ == "__main__":

	if len(sys.argv) < 2:
		mat_files = get_files('mat',os.getcwd())
	elif len(sys.argv) > 1:
		mat_files = [sys.argv[1]]
	else:
		print 'hwwwhat?'

	print mat_files
	
	for file in mat_files:

		spikes = sio.loadmat(file)

		clust_name = sio.whosmat(file)[0][0]
		print 'whosmat!! ', spikes[clust_name].shape
		plot_waveforms(spikes,clust_name)
		#spike_trials(spikes,clust_name)