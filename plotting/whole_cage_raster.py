#!/usr/bin/env python
from bokeh.plotting import figure,show
from bokeh.io import output_file, gridplot, output_file, show, save, vplot
import numpy as np
import sys,os


def plot_raster(spikes_path,ttl_path,save_dir):

    fs = 3e4

    ## get spike times

    #spikes_path = './ChGroup_0/SpikeTimes'

    spiketimes_file = open(spikes_path,"rb")

    spike_times = np.fromfile(spiketimes_file,dtype=np.int64)

    spiketimes_file.close()


    ## get TTL file:

    #ttl_file = open('./TTLChanges/Ch_2', "rb")
    ttl_file = open(ttl_path, "rb")

    ttl = np.fromfile(ttl_file, dtype=np.uint64)

    ttl_file.close() 

    ttl_on_times = ttl[0::2]

    trials = range(len(ttl_on_times))
    trial_range = 7500 # samples. i.e. 250 ms


    raster = figure(width=1000, height=400,y_axis_label='Trial Number',title='Raster')

    for trial in trials:
        #print ttl_on_times[trial] - 7500, ttl_on_times[trial] + 7500*3
        trial_time = ttl_on_times[trial]
        #print trial_time
        temp_idx = (spike_times < trial_time+trial_range*3) & (spike_times > trial_time - trial_range)
        trial_spike_times = spike_times[temp_idx] - trial_time
        
        num_spikes_ontrial = len(trial_spike_times)
        
        raster.segment(x0=trial_spike_times / 3e4, y0=np.repeat(trial,num_spikes_ontrial), x1=trial_spike_times /3e4,
                               y1=np.repeat(trial+1,num_spikes_ontrial), color="black", line_width=0.5)




    channel_name = spikes_path[spikes_path.find('ChGroup_'):spikes_path.find('/SpikeTimes')] # e.g. ChGroup_3
    print 'channel_name = ', channel_name

    output_file(save_dir + '/spike_raster_'+str(channel_name)+'.html')

    save(raster)

if __name__ == "__main__":


    spikes_path = sys.argv[1]

    ttl_path = sys.argv[2]

    
    flashing_light = './flashing_light_PSTH/' 
    if not os.path.exists(flashing_light):
        os.makedirs(flashing_light)

    print 'Saving in folder ', flashing_light

    plot_raster(spikes_path,ttl_path,flashing_light)



