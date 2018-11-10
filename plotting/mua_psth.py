import numpy as np
from sp import multirate


def default_params(winsigma=.0025,noise='none',car_exclude= [],savedir=pwd,min_f=1,max_f=10e3;hist_colors='jet',mua_colors='hot',figtitle='',freq_range=[500 4.5e3],proc_fs=10e3,downsampling=2,channels=EPHYS.labels,hampel=6,attenuation=40,ripple=.2,car_trim=40,fs = 30e3):

	# some stuff


################################################################

#################### SIGNAL CONDITIONING ################################

downfact = fs / proc_fs;

if downfact%1>0:
	print 'Downsample factor must be integer'
	# should raise an error...

def ephys_denoise_signal(data,labels,channels,noise,car_exclude,car_trim):
	#some stuff
	return

proc_data = ephys_denoise_signal(EPHYS.data,EPHYS.labels,channels,noise,car_exclude, car_trim);

print 'Anti-alias filtering and downsampling';

def ephys_condition_signal(data,freq_range):
	filt_order = 2;

proc_data = ephys_condition_signal(proc_data,[300, proc_fs/2]);

print 'Downsampling to ' + str(proc_fs);
proc_data = proc_data[0::downfact]; # figure out best way to do this!
# why does proc_data get passed to ephys_condition_signal again? line 149. Make it a float32 (single) also




[nsamples, ntrials,nchannels] = proc_data.shape # ? is proc_data an np array?
TIME = [1:nsamples]/ proc_fs;

# "if downsampling:"
print 'Downsampling by factor of ' + str(downfact);
MUA = {};
MUA['t'] = TIME[0::downsampling];
MUA['image'] = np.zeros(shape=(ntrials,len(MUA['t']),len(channels)),dtype='float32')

for i in range(nchannels): MUA['image'][:,:,i] = proc_data[0::downsampling,:,i]; # this was transposed in .m - why?

MUA['channels'] = channels;
MUA['trials'] = trials;

################################################################################

print 'Generating figures....';
print 'Saving to directory ' + str(savedir);

[path, name,ext] = os.path.split(savedir) # what is ext? need to get rid of it
# or just get pwd and make mua folder there


