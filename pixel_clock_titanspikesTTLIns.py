# This is will be our scratch pad
import sys, os,re
import h5py
from utils import pixelclock, timebase
#import open_ephys
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
#from get_bitcode_simple import get_bitcode_simple
import itertools
#import concatenateKWIKfiles
import pandas as pd
from bokeh.io import output_file, show #, vplot # grid_plot
from bokeh.plotting import figure
from bokeh.models import TapTool, HoverTool
from bokeh.colors import RGB
from itertools import groupby
import  titanspikes_ttl_extract


# mworks
try:
    sys.path.append('/Library/Application Support/MWorks/Scripting/Python')
    sys.path.append('/Users/guitchounts1/Library/Application Support/MWorks/Scripting/Python')
    
    import mworks.data as mw
except Exception as e:
    print("Please install mworks...")
    print(e)


def isiterable(o):
    return hasattr(o, '__iter__')

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
    oe_path = []
    mwk_path = []
    for file in os.listdir(input_file_path):
        if file.endswith(".mwk"):
            mwk_path.append(file)
        elif file.endswith(".kwik"):
            oe_path.append(file)
    sort_nicely(oe_path)
    return mwk_path, oe_path

def highpass_codetimes(code_times,fs,thresh_samples = 0.2):
    code_times = np.array(code_times)
    
    thresh = thresh_samples * fs # seconds * samples/sec = samples
    dels = []
    diffs = [j-i for i, j in zip(code_times[:-1,0], code_times[1:,0])]
    
    print('Running lowpass on codetimes')

    for idx,item in enumerate(diffs):
        if item > thresh:
            print('time greater than thresh = ', item)
                    
            dels.append(idx)

    code_times = code_times.tolist()

    for index in sorted(dels, reverse=True):
        del code_times[index] # erase the indices for the times [idx,0], codes [idx,1], and channels [idx,2]

    
    return code_times


def lowpass_codetimes(code_times,fs,thresh_samples = 0.2):
    code_times = np.array(code_times)
    
    thresh = thresh_samples * fs # seconds * samples/sec = samples
    dels = []
    diffs = [j-i for i, j in zip(code_times[:-1,0], code_times[1:,0])]
    
    print('Running lowpass on codetimes')

    for idx,item in enumerate(diffs):
        if item < thresh:
            #print 'time less than thresh = ', item
                    
            dels.append(idx)

    code_times = code_times.tolist()

    for index in sorted(dels, reverse=True):
        del code_times[index] # erase the indices for the times [idx,0], codes [idx,1], and channels [idx,2]

    
    return code_times

def del_duplicate_codes(codes):
    codes = np.array(codes)
    code_times = [i[1] for i in codes] # get the actual code 

    diffs = np.diff(code_times) # find the diffs in the codes
    # where the diffs = 0, the code repeats. Find those indices and erase from codes:
    dels = np.where(diffs==0)
    codes = codes.tolist()
    for index in sorted(dels[0], reverse=True): #### dels[0] b/c dels = tuple. e.g. "(array([1, 5]),)"
        del codes[index] # erase the indices for the times [idx,0], codes [idx,1], and channels [idx,2]

    return codes

def fit_line(oe,mw):  #### oe = x; mw = y variable. 
    # fit line:
    A = np.vstack([oe[0:1000], np.ones(len(oe[0:1000]))]).T ## ! might be dangerous to hardcode the 1000 here but using 
    # length = min(len(oe),len(mw)) ==== 13000 in the case for one exp for grat17 == bad result (likely because the matches suck beyond the very beginning)
    m,c = np.linalg.lstsq(A,mw[0:1000])[0]
    print('m,c = ', m,c)

    return m,c


def mw_to_oe_time(mw_time,m,c):  ### y=mx+c. ( mw=m(oe)+c |||||| oe = (mw-c)/m )
    #for m_time in mw_time:
    #    oe = (m_time - c)/m
    return (mw_time - c)/m

def oe_to_mw_time(oe_time,m,c):
    return m*(oe_time)+c


def sync_pixel_clock(mwk_path, oe_path, oe_channels=[0, 1]):

    # 1. read in ephys binary data and timestamps

    
    ###[times,channels,directions] = titanspikes_ttl_extract.get_TTL_info('./636152664381217973/TTLChanges/Ch_')
    
    # times are in seconds.microseconds
    ephys_fs = 1

    experiment_length=[]


    print('Experiment length = ', experiment_length)

    
    #oe_codes, latencies = pixelclock.events_to_codes(np.vstack((times, channels, directions)).T, len(oe_channels), 0.01)
    # the pixel clock should change once per frame, or at ~16ms max. This is 16ms * 30samples/ms = 480 samples. If a code is shorter than that, it's probably a fluke.
    # if oe_code times are in 636... format, use 10e4 as min code length
    #if in seconds.microseconds, min code time = 0.01

    oe_codes = titanspikes_ttl_extract.read_raw_ttl(oe_path)  # './636151800793559606/TTLIns'
    # oe_codes[0,:] = times
    # oe_codes[1,:] = codes

    print('Number of ephys codes = ', len(oe_codes))

   
    # !! assuming there's just one mworks file, take the first element in the list mwk_path:
    mwk_path = os.path.abspath(mwk_path[0])

    mwk = mw.MWKFile(mwk_path)
    mwk.open()



    # Start by getting the pixel clock / bit code data
    stimulus_announces = mwk.get_events(codes=['#announceStimulus'])

    # bit_codes is a list of (time, code) tuples
    mw_codes = [(e.time, e.value['bit_code']) for e in stimulus_announces if isiterable(e.value) and 'bit_code' in e.value]

    print('Number of mworks codes = ', len(mw_codes))
    ## for mw_codes and oe_codes - if one code persists for too long a time (>thresh), get rid of it (keep only the fast-changing codes that come from the grating stimulus):

    oe_codes = lowpass_codetimes(oe_codes,fs=1,thresh_samples = 0.01) #0.2
    mw_codes = highpass_codetimes(mw_codes,fs=1e6,thresh_samples = 1)



    print('Number of oe codes after lowpass = ', len(oe_codes))
    print('Number of mworks codes after lowpass = ', len(mw_codes))



    #oe_codes = del_duplicate_codes(oe_codes)
    #mw_codes = del_duplicate_codes(mw_codes)


    #### special skipping first few codes (which are bad,mkay) to get better matches- 8/3/16 for grat17:
    #mw_codes = mw_codes[1:]
    #oe_codes = oe_codes[425:]

    # 3. get pixel clock matches
    matches = []
    win_size = 40
    print('win max is ',int(len(oe_codes)/win_size))
    for win in range(0,int(len(oe_codes)/win_size),50): #range(int(round(len(oe_codes)/win_size)))
        print('win = ', win)
        if win*win_size+win_size < len(oe_codes):
            tmp_match = pixelclock.match_codes(
                [evt[0] for evt in oe_codes[win*win_size:(win+1)*win_size]], # oe times
                [evt[1] for evt in oe_codes[win*win_size:(win+1)*win_size]], # oe codes
                [evt[0] for evt in mw_codes], # mw times
                [evt[1] for evt in mw_codes], # mw codes
                minMatch = 20,
                maxErr = 0) 
            print('temp matches = ', tmp_match)
            matches.extend(tmp_match)
        else:
            #print '!!win = ', win
            tmp_match = pixelclock.match_codes(
                    [evt[0] for evt in oe_codes[win*win_size:-1]], # oe times
                    [evt[1] for evt in oe_codes[win*win_size:-1]], # oe codes
                    [evt[0] for evt in mw_codes], # mw times
                    [evt[1] for evt in mw_codes], # mw codes
                    minMatch = 9,
                    maxErr = 0)
                    
            matches.extend(tmp_match)
    
    
    print('matches = ', matches)
    #print 'type = ', type(matches)
  

    mw_times = [item[0] for item in mw_codes] #[e.time for e in stimulus_announces if isiterable(e.value)]
    oe_times = [item[0] for item in oe_codes]

       

    # condition the data to plot square pulses:
    tmp_mw_codes = [evt[1] for evt in mw_codes]
    tmp_mw_codetimes = [evt[0] for evt in mw_codes]
    plot_mw_codes = np.array(list(itertools.chain(*zip(tmp_mw_codes,tmp_mw_codes[:-1])))) 
    plot_mw_codetimes = np.array(list(itertools.chain(*zip(tmp_mw_codetimes,tmp_mw_codetimes[1:])))) 

    tmp_oe_codes = [evt[1] for evt in oe_codes]
    tmp_oe_codetimes = [evt[0] for evt in oe_codes]
    plot_oe_codes = np.array(list(itertools.chain(*zip(tmp_oe_codes,tmp_oe_codes[:-1])))) 
    plot_oe_codetimes = np.array(list(itertools.chain(*zip(tmp_oe_codetimes,tmp_oe_codetimes[1:])))) 



    # Bokeh:

    ## make save directory:
    save_folder = './pixelclock/' 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)



    ####### FIGURE 1 ###########

    colors = []
    #col = np.matlib.repmat(rgb,10,1)

    for i in range(len(matches)):
        r = np.random.randint(255)
        g = np.random.randint(255)
        b = np.random.randint(255)
        
        colors.append(RGB(r,g,b)) 
    match_idx = [idx for idx,match in enumerate(matches)]    
    #TOOLS = [HoverTool(),'box_zoom','reset','box_select']
    TOOLS="pan,wheel_zoom,box_zoom,reset,hover,previewsave"
    s1 = figure(width=1000, plot_height=500, title='MWorks and OpenEhys Pixel Clock Codes') # ,tools = TOOLS
    s1.line(plot_mw_codetimes/1e6,plot_mw_codes)
    mw_match_circles = [mat[1]/1e6 for mat in matches]
    mw_match_circles_samples = [mat[1] for mat in matches]

    s1.circle(mw_match_circles,match_idx,color=colors,size=20)
    
    #s1.circle(mw_match_circles,np.ones(len(matches)),color=colors,size=20)
    s1.yaxis.axis_label = 'MW Codes'

    #tap = s1.select(dict(type=TapTool))
    

    s2 = figure(width=1000, plot_height=500, title=None) #,tools = TOOLS
    s2.line(plot_oe_codetimes/ephys_fs,plot_oe_codes)
    oe_match_circles = [mat[0]/ephys_fs for mat in matches]
    oe_match_circles_samples = [mat[0] for mat in matches]
    
    
    s2.circle(oe_match_circles,match_idx,color=colors,size=20) # ,tags = match_idx
    #s2.circle(oe_match_circles,np.ones(len(matches)),color=colors,size=20) # ,tags = match_idx
    s2.xaxis.axis_label = 'Time (sec)'
    s2.yaxis.axis_label = 'OE Codes'


    p = gridplot([[s1], [s2]])
    output_file(save_folder + "pc_codes_match.html")
    # show the results
    show(p)
    
    #plt.savefig('pc_code_matches.pdf')
    #plt.show()


    #m,c = fit_line(tmp_oe_codetimes,tmp_mw_codetimes) #oe,mw 
    m,c = fit_line(oe_match_circles_samples,mw_match_circles_samples)
    # tb object lets you go back and forth between oe and mw timezones
    tb = timebase.TimeBase(matches,tmp_oe_codetimes,tmp_mw_codetimes)
    

    ## to test quality of match, plot OE codes in MW time

    print('len plot_oe_codetimes = ', len(plot_oe_codetimes))



    


    #### want: take MW time (e.g. stim time) and get oe time:
    mw2oe_time = []
    for mw_time in plot_mw_codetimes:
        oe_tmp = mw_to_oe_time(mw_time,m,c) ### take MW time and convert to OE time 
        #oe_tmp = tb.mw_to_oe_time(mw_time)
        mw2oe_time.append(oe_tmp)

    mw2oe_time = np.array(mw2oe_time)

    oe2mw_time = []
    for oe_time in plot_oe_codetimes:
        mw_tmp = oe_to_mw_time(oe_time,m,c)  ### take OE and convert to MW time!
        #mw_tmp = tb.oe_to_mw_time(oe_time)
        oe2mw_time.append(mw_tmp)

    oe2mw_time = np.array(oe2mw_time)


    ####### FIGURE 2 ###########


    ####### PLOT the codes on the same time axis: e.g. everything on MW.
    
    

    tmp_oeMWconv_codetimes = [tb.audio_to_mworks(evt[0]/ephys_fs)* 1e6 for evt in oe_codes]
    plot_oeMWconv_codetimes = np.array(list(itertools.chain(*zip(tmp_oeMWconv_codetimes,tmp_oeMWconv_codetimes[1:])))) 

    

    s1 = figure(width=1000, plot_height=500, title='OE Codes Plotted in MW Time')
    s1.line(plot_mw_codetimes/1e6,plot_mw_codes)
    
    #match_circles = [mat[1] for mat in matches]
    #s1.circle(match_circles,np.ones(len(matches)),color=colors,size=20)

    s1.yaxis.axis_label = 'MW Codes in MW Time'

    s2 = figure(width=1000, plot_height=500, title=None,x_range=s1.x_range,y_range=s1.y_range)
    s2.line(oe2mw_time/1e6,plot_oe_codes)
    s2.xaxis.axis_label = 'Time (sec)'
    s2.yaxis.axis_label = 'OE Codes in MW Time'
    
    p = gridplot([[s1], [s2]])
    output_file(save_folder + "oe_codes_in_MWtime.html")
    # show the results
    show(p)
    
    ####### FIGURE 3 ###########

    s1 = figure(width=1000, plot_height=500, title='MW Codes Plotted in OE Time')
    s1.line(mw2oe_time/ephys_fs,plot_mw_codes)
    
    #match_circles = [mat[1] for mat in matches]
    #s1.circle(match_circles,np.ones(len(matches)),color=colors,size=20)

    s1.yaxis.axis_label = 'MW Codes in OE Time'

    s2 = figure(width=1000, plot_height=500, title=None,x_range=s1.x_range,y_range=s1.y_range)
    s2.line(plot_oe_codetimes/ephys_fs,plot_oe_codes)
    s2.xaxis.axis_label = 'Time (sec)'
    s2.yaxis.axis_label = 'OE Codes in OE Time'
    
    p = gridplot([[s1], [s2]])
    output_file(save_folder + "mw_codes_in_OEtime.html")
    # show the results
    show(p)
    

    ####### FIGURE 4 ###########
    ####### PLOT the LINE fit:
    
    pp = figure(width=1000, plot_height=500, title='Line Fit for MW and OE Time')
    
    #pp.line(oe2mw_time/1e6,plot_oe_codes)
    pp.line(plot_oe_codetimes,m*plot_oe_codetimes+c,color='red')
    #pp.circle(plot_oe_codetimes[0:len(matches)],plot_mw_codetimes[0:len(matches)])
    pp.circle(oe_match_circles_samples,mw_match_circles_samples)
    pp.xaxis.axis_label = 'oe codetimes'
    pp.yaxis.axis_label = 'mw codetimes'
    output_file(save_folder + "pc_line_fit.html")
    show(pp)


    print("number of MW events:")
    print(len(mw_times))

    print("number of OE events:")
    print(len(oe_times))

    print("number of matches: " + str(len(matches)))
    
    linefit = dict(m=[m],c=[c])
    linefit_pd = pd.DataFrame.from_dict(linefit)
    linefit_pd.to_csv('linefit.csv')
   

    


    return matches,mwk,m,c,experiment_length

#------------------------------------------------------------------------------------- 

# TO DO: get rid of these:



# input = 'grat10_ephys_160202_1622.mwk/grat10_ephys_160202_1622.mwk'
# file = pymworks.open_file(input)
# events = file.get_events('#announceStimulus')



# # find all unique stimulus names so we know what we're dealing with:
# stimuli_names = set([ev.value['name'] for ev in events if hasattr(ev.value, '__iter__')])

# # take the name and times of a particular stimulus:
# def get_stim_times(stim_name):
#     stim_times = [ev.time for ev in events if hasattr(ev.value, '__iter__') and ev.value['name'] == stim_name]
    
def get_mw_stim_times(mwk):
    events = mwk.get_events(codes =['#announceStimulus'])
    stimuli_names = set([ev.value['name'] for ev in events if hasattr(ev.value, '__iter__')])

    # get times of grating stim:
    grating_stim_times_mw = [ev.time for ev in events if hasattr(ev.value, '__iter__') and ev.value['name'] == 'grating']    
    gratings = [ev.value for ev in events if hasattr(ev.value, '__iter__') and ev.value['name'] == 'grating']

    # pixel clock refreshes @ 60Hz. Get the time stamps that are separated by longer than 1/60 seconds
    diffs = np.diff(grating_stim_times_mw)
    stim_transition_idx = [i for i,v in enumerate(diffs) if v>17000] # take anything longer than this
    stim_transition_idx = [z+1 for z in stim_transition_idx] # add 1 b/c the transition is on the next index...
    stim_transition_idx.insert(0,0) # don't forget to add the first orientation! 
    stim_transition_times = [grating_stim_times_mw[x] for x in stim_transition_idx] # convert to seconds?

    #stim_orientations = [gratings[x]['rotation'] for x in stim_transition_idx]

    ################################## GET JUST THE SUCCESSFUL NOSEPOKE TRIALS ##########################################
    
    success_trials = mwk.get_events(codes=['nosepoke_success'])


    successtrial_stimtime = []
    ###  loop over success trials. for each, subtract all stim_transition_times. Take whichever one is smallest and positive:
    for idx,trial in enumerate(success_trials):
    
    
        trial_diffs = [trial.time-x for x in stim_transition_times]
        
        closest_time = [thing for thing in trial_diffs if thing > 0]
        if len(closest_time) > 0:
                   
            successtrial_stimtime.append(trial.time - np.min(closest_time))
        ### get the index of that closest time  in the stim_transition_times array and take that transition_time. 
    
    # then, get the indeces of the events that had successful trials:
    successtrial_idx = [np.where(grating_stim_times_mw==x)[0][0] for x in successtrial_stimtime]
    successtrial_orientations = [gratings[x]['rotation'] for x in successtrial_idx]


    ## alternative way to get stim times, using itertools groupby:
    #x = np.array([grating['rotation'] for grating in gratings])
    #grouped_L = [(k, sum(1 for i in g)) for k,g in groupby(x)] ## ignore the sum - this just counts the # indices each orientation repeats (seems to be 421 times (? 421/60Hz = 7 seconds per stim?))
        # grouped_L = list of tuples; first element in tuple is the orientation; second is the num of repeats. 
        # this method fails when one orientation is [geniunely] repeated (b/c it looks for changes in the rotation)
    
    mwk.close()

    #return stim_transition_times,stim_orientations
    return successtrial_stimtime,successtrial_orientations
    

#     kwik_times = [tb.mworks_to_audio(x/1e6) for x in stim_times] #mworks times are in microseconds - convert to seconds
    
#     return kwik_times

# # make dictionary of stimulus name and its times:
# stimuli_kwik_times = {stimulus_name : get_stim_times(stimulus_name) for stimulus_name in stimuli_names}    

# # give kwik times to..... .raw.kwd file....
# # 1. determine which .raw.kwd file(s) we'll need - compare stimulus times with .kwik files' stamps.

# # collect the raw.kwd files:
# def get_kwd_files():
#     input_file_path = os.getcwd()
#     kwd_files = []
#     for file in os.listdir(input_file_path):
#         if file.endswith(".raw.kwd"):
#             kwd_files.append(file)

#     return kwd_files

# kwd_files = get_kwd_files();


# # find their start and stop times:
# def get_data(file):
#     kwd_file = h5py.File(file,'a')
#     metadata = {};
#     #data['ephys'] = kwd_file['/recordings/0/data']
#     metadata['sample_rate'] = kwd_file['/recordings/0/'].attrs['sample_rate'] # in samples (ephys_fs)
#     metadata['start_time'] = kwd_file['/recordings/0/'].attrs['start_time'] # in samples
#     metadata['start_sample'] = kwd_file['/recordings/0/'].attrs['start_sample']
#     #data['length'] = len(data['ephys']) # this is in samples
#     #stat = os.stat(raw_kwd) # os.path.getmtime(raw_kwd)
#     return metadata

# # compile metadata from all raw files:
# all_metadata = [];
# for file in kwd_files:
#     all_metadata.append(get_data(file))

# # for a given stim time, check which raw.kwd file it's in:
# def check_which_kwd_file(stim_time):
#     for i, item in enumerate(alldata):
#         if stim_time > alldata[i]['start_time']:
#             print 'stim is in file ' + str(i)
#             x = i
#         else:
#             pass
#     return x # x is the index of the file in raw_files.


# # collect the indices of all stimuli:
# stim_files = [];
# for stim_time in input_stim: # input_stim is the input list of stimuli times. for e.g. stimulus '0', use stimuli_kwik_times['0']
#     stim_files.append(check_time(stim_time))

# print stim_files # this is the output list of file indices for each stimulus. 

# # next, take the stimulus time and file index, and extract a relevant chunk of ephys data:
# relevant_data = [];
# for file in stim_files:
#     kwd_file = h5py.File(kwd_files[file],'a')
#     relevant_data.append(kwd_file['/recordings/0/data'])
#     print file








if __name__ == "__main__":

    #mwk_file = sys.argv[1]
    

    # 1. get KWIK and MWK files in current directory
    mwk_path,oe_path = get_files()
    
    oe_path = sys.argv[1]

    print('mwk path is ', mwk_path)
    
    print('oe path is ', oe_path)



    matches,mwk,m,c,experiment_length = sync_pixel_clock(mwk_path, oe_path, oe_channels=[0,1])
    
   


    
    ########## get times of grating orientation: ###################
    mw_stim_transition_times,stim_orientations = get_mw_stim_times(mwk);

    ephys_fs = 1
    oe_stim_transition_times = []
    for mw_time in mw_stim_transition_times:
        oe_tmp = mw_to_oe_time(mw_time,m,c) ### take MW time and convert to OE time 
        #oe_tmp = tb.mw_to_oe_time(mw_time)
        oe_stim_transition_times.append(oe_tmp/ephys_fs) 

    oe_stim_transition_times = np.array(oe_stim_transition_times)

    ## get experiment number for oe times? 

    print('oe_stim_transition_times = ', oe_stim_transition_times)

    #d = {'times':oe_stim_transition_times,'orientations':stim_orientations} #,'experiment_lengths':experiment_length
    d = dict(times = oe_stim_transition_times, orientations = stim_orientations)
    stim_info = pd.DataFrame.from_dict(d)
    #stim_info.transpose()
    stim_info.to_pickle('oe_stim_times')
   


















