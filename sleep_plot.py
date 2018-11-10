import numpy as np
import os
import sys
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy.matlib
from bokeh.plotting import figure,show
from bokeh.io import output_notebook
from matplotlib import gridspec
from sklearn import preprocessing




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

def concat_data(trace_files):
	# make one big array with all traces:
	power_data = []
	windowed_data = []
	t = []
	freqs = []

	for file in trace_files:
	    print 'Loading file ', file
	    
	    tmp_file = np.load(file)
	    
	    
	    power_data.append(tmp_file['power_data'][:,0:273].T) # [:,0:273]=Pxx; [:,273]=freqs
	    windowed_data.append(tmp_file['windowed_data'])
	    freqs.append(tmp_file['power_data'][:,273])
	    t.append(tmp_file['t'])
	        
	    tmp_file.close()

	windowed_data = np.array(windowed_data)
	power_data = np.array(power_data).T
	power_data = np.reshape(power_data,[power_data.shape[0],power_data.shape[1] * power_data.shape[2]])

	freqs = np.array(freqs)

	return windowed_data,power_data,freqs,t

def get_big_time(t):
	big_time_vec = t[0]
	for i,t_x in enumerate(t):
	    if i>0:
	        #print i*max(t[0])
	        tmp = t[0] + i*max(t[0])
	        big_time_vec = np.append(big_time_vec,tmp)
	        
	return big_time_vec


def get_freq_idx(freqs,desired_freq): # make desired_freq a tuple, e.g. (0,4)
    idx = []
    for counter,value in enumerate(freqs):
        if  desired_freq[0] <= value <= desired_freq[1]:
            #yield counter
            idx.append(counter)
    return idx	

def power_auc(trace,channel,freq_range):  #### e.g. power_auc(power_data,'lfp',(0,25))
	idx = get_freq_idx(freqs[0,:][channel],freq_range)
	Pxx = trace[idx,:][channel]
	Pxx_intgr = stats.zscore(np.trapz(Pxx,axis=0))
	return Pxx_intgr
    
def plot_spec(power_data,freqs,t):
	####### PLOT STUFF #########
	print("Plotting stuff...")

	idx_0_50Hz = get_freq_idx(freqs[0,:]['lfp'],(0,50))
	idx_0_25 = get_freq_idx(freqs[0,:]['lfp'],(0,25))

	Pxx_lfp_0_25 = power_data[idx_0_25,:]['lfp']
	#Pxx_lfp_0_25_intgr = stats.zscore(np.trapz(Pxx_lfp_0_25,axis=0))

	Pxx_lfp_0_25_intgr = power_auc(power_data,'lfp',(0,25))

	Pxx_emg_0_100_intgr = power_auc(power_data,'emg',(0,100))

	print 'shape of Pxx_lfp_0_25 = ', Pxx_lfp_0_25.shape
	#make Pandas dataframe out of Pxx for plotting heatmap
	big_time_vec = get_big_time(t)

	b = pd.DataFrame(data=Pxx_lfp_0_25,
	    index = freqs[0,idx_0_25]['lfp'],
	    columns = big_time_vec )


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

	#ax3 = ax1.twinx()
	#ax3 = sns.tsplot(data=Pxx_emg_0_100_intgr,value='EMG',color='blue',linewidth=0.1)
	#ax3.yaxis.set_visible(False)

	sns.despine()

	save_name = 'all_exp'
	fig.savefig((save_name+".pdf"))

if __name__ == "__main__":
	trace_files = []

	for file in os.listdir(os.getcwd()):
	    if file.endswith(".npz"):
	        trace_files.append(file)

	sort_nicely(trace_files)
	print 'NPZ files: ', trace_files

	windowed_data,power_data,freqs,t = concat_data(trace_files)


	plot_spec(power_data,freqs,t)



