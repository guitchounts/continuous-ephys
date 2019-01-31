import numpy as np
import matplotlib.pylab as plt
from scipy import stats,signal
import seaborn as sns
import sys,os
sns.set_style('ticks')
plt.rcParams['pdf.fonttype'] = 'truetype'
import pandas as pd
import json
sys.path.append("/Users/guitchounts/Documents/GitHub/mountainlab/old/WIP/python/mda/")
sys.path.append('/Users/guitchounts/Dropbox (coxlab)/Scripts/Repositories/continuous-ephys/utils')
sys.path.append('/n/home11/guitchounts/code/continuous-ephys/utils')
sys.path.append('/n/home11/guitchounts/code/mountainlab/old/WIP/python/mda/')

from utils import read_titan_rhd
from mdaio import readmda

def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):

    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace


if __name__ == "__main__":

    #tet4_firings = readmda('/Volumes/coxfs01/guitchounts/MountainSortCluster-master/TetrodeData/Tetrode4/data/firings2-4.mda')

    ### starting in a dir like GRat54/636xxxxx/0/
    #tetrode_num = 1

    waveform_stats = {}

    for tetrode_num in range(1,17):
        waveform_stats[tetrode_num] = {}

        tet_raw = readmda('./TetrodeData/Tetrode%d/data/rawT%d.mda' %(tetrode_num,tetrode_num))
        ## ^^ can also load from original RHD file. 

        #### this stuff will actually be on /n/regal (or wherever you put it)
        clust_dir = './Tetrode%d/mountainlab/prvbucket/_mountainprocess/' % tetrode_num

        firings_files =  [out_file for out_file in os.listdir(clust_dir) if out_file.startswith('output_firings_out')]
        output_firings_out = firings_files[np.argmax([os.path.getsize(clust_dir + file) for file in firings_files])]
        cluster_times_ids = readmda(clust_dir + output_firings_out)


        metrics_files =  [out_file for out_file in os.listdir(clust_dir) if out_file.startswith('output_metrics_out')]
        output_metrics_out = metrics_files[np.argmax([os.path.getsize(clust_dir + file) for file in metrics_files])]
        metrics_path = (clust_dir + output_metrics_out)

        with open(metrics_path) as json_data_file:
            metrics = json.load(json_data_file)
        # rhd_path = '/Volumes/coxfs01/guitchounts/ephys/GRat27/636520606779599145.rhd'
        # rhd = read_titan_rhd.ReadTitanRHD(rhd_path)
        # length = 1e6
        # ephys_timestamps,acc,ephys = rhd.read_file(length)

        fs = 3e4

        tet_filt = filter(tet_raw.T,[800,8e3],fs=fs).T

        


        ### Gather the avg waveforms from each cluster:

        ## for plotting purposes, want a mat with the extracted waveforms:
        ## e.g. 4x 32 x 100000 waveforms. use cluster_times_ids mat to separate cluster IDs. 
        ## in tet_filt, find the indexes of spiketimes and extract -16 until +16 timepoints for each. 

        spike_width = 64 # number of samples to take, centered at spike peak. #(16,16)  

        waveforms = [] ## list of arrays. length = number of clusters
        isis = []
        for clu in np.unique(cluster_times_ids[2,:]):  #[9.,11.]: #
            


            tmp_waveforms = np.empty([4,np.where(cluster_times_ids[2,:] == clu)[0].shape[0],spike_width]) ### 4xnum_spikesx32
            
            peaks = cluster_times_ids[1,np.where(cluster_times_ids[2,:]==clu)[0]].astype('int') ## index of tet4 where peak happens
            
            clu_times = cluster_times_ids[1,np.where(cluster_times_ids[2,:]==clu)[0]]
            #clu_times = xx[1,np.where(xx[2,:]==clu)[0]]
            isis.append(np.diff(clu_times)/fs*1e3) ## in milliseconds
            
            tmp_waveforms = tet_filt[:,[np.arange(peak-spike_width/2,peak+spike_width/2) for peak in peaks]]
            waveforms.append(tmp_waveforms)
            
            


        #### NO figure
        clusts = np.unique(cluster_times_ids[2,:]).astype('int')
        
        ###f,axarr = plt.subplots(len(clusts),3,dpi=600,sharey='col',gridspec_kw={'hspace':2,'wspace':0,'width_ratios':[1,2,4]})
                                                                           #'width_ratios':[1,2]}) # , figsize=(2,6)
        #print('clusts = ',clusts)                                                                  
        for clu_idx,clu in enumerate(clusts):
            
            waveform_stats[tetrode_num][clu] = {'widths' : [], 'heights' : [], 'slopes' : [], 'metrics' : []  }

            for ch in range(4):
                
                #clust_times = np.where(xx[2,:].astype('int') == 9)[0]
                #axarr[clu_idx].plot(range(0+32*ch,32+32*ch), np.mean(waveforms[clu_idx][ch,:,:],axis=0))
                #print('clu,waveforms[clu_idx].shape = ',clu,waveforms[clu_idx].shape)
                y = np.mean(waveforms[clu_idx][ch,:,:],axis=0)
                err = np.std(waveforms[clu_idx][ch,:,:],axis=0)
                x = range(0+spike_width*ch,spike_width+spike_width*ch)
                #axarr[clu_idx,0].plot(x,y,c='k',lw=.25)
                #axarr[clu_idx,0].fill_between(x, y-err, y+err,alpha=.25,color='k',linewidth=0)
                #axarr[clu_idx,0].set_title('%d Spikes in Cluster %d' % (waveforms[clu_idx].shape[1],clu),fontdict={'fontsize' : 6})
                #axarr[clu_idx,1].hist(isis[clu_idx],bins=200)
                

                ### get waveform stats (width, peak:trough, slope at the end, firing rates )
                min_wv = np.argmin(y)
                max_wv = min_wv + np.argmax(y.flatten()[min_wv:])  ### [min_wv:min_wv+spike_width/2])
                
                peak = spike_width/2

                if y[peak] < 0:

                    max_post_peak = np.argmax(y.flatten()[peak:])
                
                elif y[peak] > 0:

                    max_post_peak = np.argmin(y.flatten()[peak:])

                #width = (max_wv - min_wv) / fs * 1e3
                #height = abs(y.flatten()[max_wv]) / abs(y.flatten()[min_wv])
                width = (max_post_peak - peak) / fs * 1e3 ## in ms
                height = abs(y.flatten()[peak]) / abs(y.flatten()[max_post_peak])


                slope = np.mean(np.gradient( y[-spike_width/4 : ]  )) ## take the mean gradient of the end of the spike waveform... 


                waveform_stats[tetrode_num][clu]['widths'].append(width)
                waveform_stats[tetrode_num][clu]['heights'].append(height)
                waveform_stats[tetrode_num][clu]['slopes'].append(slope)

                #clu_isi = np.diff(clu_times) ## convert from seconds to ms

                #axarr[clu_idx,1].hist(isis[clu_idx],bins=np.logspace(-1,2,100),range=[0.1,100],histtype='stepfilled')
                #n, bins, patches = 
                #plt.setp(patches, 'facecolor', 'magenta', 'alpha', 0.5)
                #axarr[clu_idx,1].set_xscale('log')
                #axarr[clu_idx,1].set_xlim([0.1, 100])
                
                if len(isis[clu_idx]) > 1:
                    violations = np.float(len(np.where(isis[clu_idx]<=1.0)[0])) / np.float(len(isis[clu_idx])) * 100
                else:
                    violations = 0

                # axarr[clu_idx,1].set_title('Violations: %.2f%% <1ms' % (violations),fontdict={'fontsize' : 4})
                
                # axarr[clu_idx,0].axes.xaxis.set_ticklabels([])
                # axarr[clu_idx,0].axes.xaxis.set_ticks([])
                # axarr[clu_idx,1].axes.xaxis.set_ticklabels([])
                # axarr[clu_idx,1].axes.xaxis.set_ticks([])
                # axarr[clu_idx,1].axes.yaxis.set_ticklabels([])
                # axarr[clu_idx,1].axes.yaxis.set_ticks([])

            waveform_stats[tetrode_num][clu]['metrics'] = metrics['clusters'][clu_idx] #['firing_rate']

            # axarr[clu_idx,2].text(0,0,metrics['clusters'][clu_idx],wrap=True,fontdict={'fontsize' : 4})
            # axarr[clu_idx,2].axes.yaxis.set_ticklabels([])
            # axarr[clu_idx,2].axes.yaxis.set_ticks([])
            # axarr[clu_idx,2].axes.xaxis.set_ticklabels([])
            # axarr[clu_idx,2].axes.xaxis.set_ticks([])
            # axarr[-1,1].set_xlabel('ISI (ms)')
            # axarr[-1,1].axes.xaxis.set_ticklabels([0.1,1,10,100])
            # axarr[-1,1].axes.xaxis.set_ticks([0.1,1,10,100])
                #axarr[i].plot(np.mean(tet4_filt[i,[np.arange(j-32,j+32) for j in clust_times]],axis=0))
                #axarr[i].plot(16,peaks[i,clust_times],'x')
        #plt.tight_layout()
        #sns.despine(left=True,bottom=True)

        save_path = './Tetrode%d' % tetrode_num

        #f.savefig(save_path + '/clusters.pdf')

        # save cluster results with spike times:
        
        # d = dict(times = cluster_times_ids[1,:]/3e4,clusters=cluster_times_ids[2,:]) ### save times in seconds and cluster assignments
        # cluster_assignments = pd.DataFrame.from_dict(d)
        # cluster_assignments.to_csv(save_path + '/cluster_assignments.csv')
        print('Saving Tetrode %d !!!!' % tetrode_num)

        waveform_stats_frame = pd.DataFrame.from_dict(waveform_stats)
        waveform_stats_frame.to_csv(save_path + '/waveform_stats.csv')


