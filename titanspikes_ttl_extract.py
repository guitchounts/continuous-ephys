#!/usr/bin/env python

import numpy as np
import datetime

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



# 1. read in the appropriate TTL files.

def read_ttl_file(file_path,channel):

	file_path = file_path + str(channel)

	# get ephys file name. This will be the 18-digit string starting with 636:
	filename_idx = file_path.find('636')

	ephysfilename = file_path[filename_idx:filename_idx+18]
	#ephysfilename = '636150871373353768'
	

	file = open(file_path, "rb")
	
	ttl = np.fromfile(file, dtype=np.uint64)

	file.close() 

	fs=30e3
	MSDNrate=1e7

	
	ttl_seconds = [thing*MSDNrate/fs + np.uint64(ephysfilename) for thing in ttl]

	print 'ttl[0:10 =', ttl[0:10]

	# get seconds.microseconds from start of file. 
	
	start_time = matlab2datetime(int(ephysfilename))

	ttl_seconds = [(matlab2datetime(thing) - start_time).total_seconds() for thing in ttl_seconds] # convert 636s to datetime objects
	# ttl_seconds = seconds.microseconds
	
	print 'Ephys file start time = ', start_time
	print 'Length of ttl_seconds = ', len(ttl_seconds)


	print 'type(ttl_seconds) ', type(ttl_seconds)
	print 'ttl_seconds[0] ', type(ttl_seconds[0])


	print 'ttl_seconds[0:10] = ', ttl_seconds[0:10]

	return ttl_seconds

# 2. get channels, directions, times. 

def get_TTL_info(file_path,ch0=0,ch1=1):

	#times:
	ttl_times_ch0 = read_ttl_file(file_path,ch0)
	ttl_times_ch1 = read_ttl_file(file_path,ch1)


	#directions:
	# first sample assume is +1, the second -1:
	directions_ch0 = np.ones(len(ttl_times_ch0),dtype='int')
	directions_ch1 = np.ones(len(ttl_times_ch1),dtype='int')
	directions_ch0[1::2] = -1
	directions_ch1[1::2] = -1

	channels_0 = np.zeros(len(ttl_times_ch0),dtype='int')
	channels_1 = np.ones(len(ttl_times_ch1),dtype='int')


	# TIMES, CHANNELS, DIRECTIONS
	times_ch_dirs_0 = np.vstack([ttl_times_ch0,channels_0,directions_ch0])
	times_ch_dirs_1 = np.vstack([ttl_times_ch1,channels_1,directions_ch1])

	times_ch_dirs = np.hstack([times_ch_dirs_0,times_ch_dirs_1])

	idx = np.argsort(times_ch_dirs)

	times_ch_dirs = times_ch_dirs[:,idx[0,:]]
	
	#print 'times_ch_dirs shape = ', times_ch_dirs.shape()

	# return channels, directions, times. Stack the two channels on top of each other. 
	return times_ch_dirs

def convert2seconds(ttlins,file_path):

	print 'Starting conversion to seconds'

	filename_idx = file_path.find('636')
	ephysfilename = file_path[filename_idx:filename_idx+18]

	fs=30e3
	MSDNrate=1e7
	
	#ttl_seconds = [thing*MSDNrate/fs + np.uint64(ephysfilename) for thing in ttlins]

	print 'ttl[0:10 =', ttlins[0:10]

	# get seconds.microseconds from start of file. 
	
	start_time = matlab2datetime(int(ephysfilename))

	#ttl_seconds = [(matlab2datetime(thing) - start_time).total_seconds() for thing in ttl_seconds] # convert 636s to datetime objects
	# ttl_seconds = seconds.microseconds
	
	ttl_seconds = ttlins / fs



	print 'Ephys file start time = ', start_time
	print 'Length of ttl_seconds = ', len(ttl_seconds)


	print 'type(ttl_seconds) ', type(ttl_seconds)
	print 'ttl_seconds[0] ', type(ttl_seconds[0])


	print 'ttl_seconds[0:10] = ', ttl_seconds[0:10]

	return ttl_seconds

def read_raw_ttl(file_path,swap_12_codes =0,limit=-1):
	# these are the 'TTLIns' files. They contain the bit codes already! 
	
	print 'TTL path = ', file_path
	# get ephys file name. This will be the 18-digit string starting with 636:
	#filename_idx = file_path.find('636')

	#ephysfilename = file_path[filename_idx:filename_idx+18]
	#ephysfilename = '636150871373353768'
	

	file = open(file_path, "rb")
	
	ttlins = np.fromfile(file, dtype=np.uint16)

	ttlins = ttlins[0:int(limit)] ## only take the first 1e7 codes 
	
	file.close() 

	# ttlins are sampled at 30kHz, so we just need to extract the changes from them

	# get diffs
	diffs = np.diff(ttlins)

	# find elements of diffs that are nonzero 
	nonzero_diffs = np.where(diffs)
	
	times = nonzero_diffs[0][:-1]/3e4 + 0.1000 ## the [0] b/c this is a tuple for some reason. 
	# adding 100ms offset! 

	codes = ttlins[nonzero_diffs[0][1:]]

	# codes = codes[np.where(codes<4)[0]]
	# times = times[np.where(codes<4)[0]]

	if swap_12_codes == 0 :
		zeros=np.where(codes==0)
		ones=np.where(codes==1)
		twos=np.where(codes==2)
		threes=np.where(codes==3)
		zeros=np.where(codes==0)
	elif swap_12_codes == 1 :
		zeros=np.where(codes==2) #1 ##### for some reason the codes in the TTLIns file are 2s,6s, 10s, and 14s instead of 0-3.... (at least in one file)
		ones=np.where(codes==6) # 10
		twos=np.where(codes==10) # 6
		threes=np.where(codes==14) # 14
		



	codes[ones]=1
	codes[twos]=2
	codes[zeros]=0
	codes[threes]=3

	print ' TTLIns codes[0:10] = ', codes[0:10]
	print 'shape of times = ', times.shape
	print 'shape of codes = ', codes.shape

	return np.vstack([times,codes]).T

if __name__ == "__main__":

	times_ch_dirs = get_TTL_info()

	print 'Shape of times_ch_dirs = ', times_ch_dirs.shape
	print 'Times', times_ch_dirs[0,0:10]
	print 'Channels', times_ch_dirs[1,0:10]
	print 'Dirs', times_ch_dirs[2,0:10]




