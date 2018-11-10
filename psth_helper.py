import pymworks

input = 'grat10_ephys_160202_1622.mwk/grat10_ephys_160202_1622.mwk'

file = pymworks.open_file(input)
events = file.get_events('#announceStimulus')


# find all unique stimulus names so we know what we're dealing with:
stimuli_names = set([ev.value['name'] for ev in events if hasattr(ev.value, '__iter__')])

# take the name and times of a particular stimulus:
def get_stim_times(stim_name):
    stim_times = [ev.time for ev in events if hasattr(ev.value, '__iter__') and ev.value['name'] == stim_name]
    
    kwik_times = [tb.mworks_to_audio(x/1e6) for x in stim_times] #mworks times are in microseconds - convert to seconds
    
    return kwik_times

# make dictionary of stimulus name and its times:
stimuli_kwik_times = {stimulus_name : get_stim_times(stimulus_name) for stimulus_name in stimuli_names}    

# give kwik times to..... .raw.kwd file....
	# 1. determine which .raw.kwd file(s) we'll need - compare stimulus times with .kwik files' stamps.
	stimuli_kwik_times['0']

# collect the raw.kwd files:
def get_kwd_files():
    input_file_path = os.getcwd()
    kwd_files = []
    for file in os.listdir(input_file_path):
        if file.endswith(".raw.kwd"):
            kwd_files.append(file)

    return kwd_files

kwd_files = get_kwd_files();


# find their start and stop times:
def get_data(file):
    kwd_file = h5py.File(file,'a')
    metadata = {};
    #data['ephys'] = kwd_file['/recordings/0/data']
    metadata['sample_rate'] = kwd_file['/recordings/0/'].attrs['sample_rate'] # in samples (30e3)
    metadata['start_time'] = kwd_file['/recordings/0/'].attrs['start_time'] # in samples
    metadata['start_sample'] = kwd_file['/recordings/0/'].attrs['start_sample']
    #data['length'] = len(data['ephys']) # this is in samples
    #stat = os.stat(raw_kwd) # os.path.getmtime(raw_kwd)
    return metadata

# compile meta data from all raw files:
all_metadata = [];
for file in kwd_files:
    all_metadata.append(get_data(file))

# for a given stim time, check which raw.kwd file it's in:
def check_which_kwd_file(stim_time):
	for i, item in enumerate(alldata):
	    if stim_time > alldata[i]['start_time']:
	        print 'stim is in file ' + str(i)
	        x = i
	    else:
	        pass
	return x # x is the index of the file in raw_files.


# collect the indices of all stimuli:
stim_files = [];
for stim_time in input_stim: # input_stim is the input list of stimuli times. for e.g. stimulus '0', use stimuli_kwik_times['0']
    stim_files.append(check_time(stim_time))

print stim_files # this is the output list of file indices for each stimulus. 

# next, take the stimulus time and file index, and extract a relevant chunk of ephys data:
relevant_data = [];
for file in stim_files:
    kwd_file = h5py.File(kwd_files[file],'a')
    relevant_data.append(kwd_file['/recordings/0/data'])
    print file





