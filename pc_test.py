import sys, os
import h5py
from utils import pixelclock, timebase
import open_ephys
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from get_bitcode_simple import get_bitcode_simple
import itertools
import concatenateKWIKfiles
import pandas as pd


if __name__ == "__main__":

	oe_codes = np.zeros([10,2])
	mw_codes = np.zeros([10,2])

	# times:
	oe_codes[:,0] = range(10)
	mw_codes[:,0] = range(10)
	mw_codes[:,0]= [time+100 for time in mw_codes[:,0]]

	# codes: 
	oe_codes[:,1] = [1, 3, 1, 2, 3, 0, 2, 0, 3, 2]
	mw_codes[:,1] = [1, 3, 1, 2, 3, 0, 2, 0, 3, 2]

	

	matches = pixelclock.match_codes(
	        [evt[0] for evt in oe_codes], # oe times
	        [evt[1] for evt in oe_codes], # oe codes
	        [evt[0] for evt in mw_codes], # mw times
	        [evt[1] for evt in mw_codes], # mw codes
	        minMatch = 3,
	        maxErr = 0) 

	print 'matches = ', matches
	# condition the data to plot square pulses:
	tmp_mw_codes = [evt[1] for evt in mw_codes]
	tmp_mw_codetimes = [evt[0] for evt in mw_codes]
	plot_mw_codes = np.array(list(itertools.chain(*zip(tmp_mw_codes,tmp_mw_codes[:-1])))) 
	plot_mw_codetimes = np.array(list(itertools.chain(*zip(tmp_mw_codetimes,tmp_mw_codetimes[1:])))) 

	tmp_oe_codes = [evt[1] for evt in oe_codes]
	tmp_oe_codetimes = [evt[0] for evt in oe_codes]
	plot_oe_codes = np.array(list(itertools.chain(*zip(tmp_oe_codes,tmp_oe_codes[:-1])))) 
	plot_oe_codetimes = np.array(list(itertools.chain(*zip(tmp_oe_codetimes,tmp_oe_codetimes[1:])))) 

	# plot the 4 code values over time:
	ax1 = plt.subplot(2, 1, 1)
	plt.plot(plot_mw_codetimes,plot_mw_codes)
	plt.scatter([mat[1] for mat in matches],np.ones(len(matches)),s=200, c=range(len(matches)))
	#plt.plot([mat[1] for mat in matches],numpy.matlib.repmat(1.5,len(matches),1), '-ro')
	plt.ylabel('MW Codes')

	ax2 = plt.subplot(2, 1, 2)
	plt.plot(plot_oe_codetimes,plot_oe_codes)
	plt.scatter([mat[0] for mat in matches],np.ones(len(matches)),s=200, c=range(len(matches)))
	#plt.plot([mat[0] for mat in matches],numpy.matlib.repmat(1.5,len(matches),1), '-ro')
	plt.ylabel('OE Codes')

	plt.show()


	tb = timebase.TimeBase(matches)

	oe_code_conv2mw_time = []
	for oe in plot_oe_codetimes:
		oe_code_conv2mw_time.append(tb.audio_to_mworks(oe))

	# plot the 4 code values over time:
	f,(ax1,ax2) = plt.subplots(2,1,sharex=True)
	#ax1 = plt.subplot(2, 1, 1)
	ax1.set_title('Plotting OE codes in MW time')
	ax1.plot(plot_mw_codetimes,plot_mw_codes)
	#ax1.scatter([mat[1] for mat in matches],np.ones(len(matches)),s=200, c=range(len(matches)))
	#plt.plot([mat[1] for mat in matches],numpy.matlib.repmat(1.5,len(matches),1), '-ro')
	plt.ylabel('MW Codes')

	#ax2 = plt.subplot(2, 1, 2)
	ax2.plot(oe_code_conv2mw_time,plot_oe_codes)
	#ax2.scatter([mat[0] for mat in matches],np.ones(len(matches)),s=200, c=range(len(matches)))
	#plt.plot([mat[0] for mat in matches],numpy.matlib.repmat(1.5,len(matches),1), '-ro')
	plt.ylabel('OE Codes in MW time')


	
	plt.show()

