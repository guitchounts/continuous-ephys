#!/usr/bin/env python

import os
import h5py
import numpy as np
import re

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

def get_files():
    input_file_path = os.getcwd()
    raw_files = []
    for file in os.listdir(input_file_path):
        if file.endswith(".kwik"):
            raw_files.append(file)

    
    return sort_nicely(raw_files)

def get_kwik_pcdata(kwik_file):
    
    
    input_h5file = h5py.File(kwik_file,'a')
    channels = np.array(input_h5file['/event_types/TTL/events/user_data/event_channels'])
    directions = np.array(input_h5file['/event_types/TTL/events/user_data/eventID'])
    times = np.array(input_h5file['/event_types/TTL/events/time_samples']).astype(np.int64)
    # event_types - appears to be unused. skipping. (not sure what to look for in the KWIK file for this...)

    input_h5file.close() ###### <- added w/o testing 04/05/17 gg

    ##### TO DO: check that the KWIK file is NOT empty. If it is, skip it. 

    oe_channels=[0,1]
    relevant = [ind for (ind,ch) in enumerate(channels) if ch in oe_channels]
    channels = channels[relevant]
    directions = directions[relevant]
    times = times[relevant]

    relevant = relevant[0::2] # take every other element - for some reason they're recorded twice.
    channels = channels[0::2]
    directions = directions[0::2].astype(np.int16) # convert to int16 b/c otherwise later subtraction gives 255 instead of -1!
    times = times[0::2]

    # renormalize the channel numbers to 0,1, etc...
    for i, ch in enumerate(oe_channels):
        channels[np.where(channels == ch)] = i

    # renormalize direction to 1, -1
    for i, ch in enumerate(oe_channels):
        directions[np.where(directions == 0)] = -1


    # get the file's first and last timestamps:
    
    first,last = [times[0],times[-1:]]
    

    uni_array,uni_idx = np.unique(times,return_index=True)
    
    #relevant_channels_directions_times = relevant_channels_directions_times[:,uni_idx]
    print 'times length = ', len(times), 'unique len: ', len(times[uni_idx])

    #return relevant[uni_idx], channels[uni_idx], directions[uni_idx], times[uni_idx]
    return relevant, channels, directions, times

def concatenate(raw_files):
    relevant_channels_directions_times = np.empty((4,0), int)
    #print 'shape of relevant_channels_directions_times is ', relevant_channels_directions_times.shape
    #print 'relevant_channels_directions_times is ', relevant_channels_directions_times
    experiment_length = []
    for file in raw_files:
        try:
            temp = get_kwik_pcdata(file)
            temp = np.array(temp)
            relevant_channels_directions_times = np.hstack((relevant_channels_directions_times,temp))
            
            experiment_length.append(relevant_channels_directions_times.shape[1])  
        except:
            pass
        #print 'temp is ', temp

        #print 'shape of temp is ', temp.shape
    
    
    return relevant_channels_directions_times,experiment_length #np.array(rcdt)


if __name__ == "__main__":

    # 1. collect all relevant kwik files:
    raw_files = get_files()
    print 'raw files is ', raw_files
    # 2. send those files to extract relevant info and concatenate:
    #all_times,all_channels,all_directions = concatenate(raw_files)
    relevant,channels,directions,times = concatenate(raw_files)
    print 'shape of relevant_channels_directions_times is ', relevant.shape,channels.shape,directions.shape,times.shape
        
