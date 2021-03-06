# This is will be our scratch pad
import sys, os,re
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
from bokeh.io import gridplot, output_file, show #,  vplot
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import TapTool, HoverTool
from bokeh.colors import RGB
from itertools import groupby
import  titanspikes_ttl_extract


# mworks
try:
    sys.path.append('/Library/Application Support/MWorks/Scripting/Python')
    import mworks.data as mw
except Exception as e:
    print("Please install mworks...")
    print e


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

def lowpass_codetimes(code_times,fs,thresh_samples = 0.2):
    code_times = np.array(code_times)
    
    thresh = thresh_samples * fs # seconds * samples/sec = samples
    print 'Lowpass threshold = ', thresh
    dels = []
    diffs = [j-i for i, j in zip(code_times[:-1,0], code_times[1:,0])]

    for idx,item in enumerate(diffs):
        if item < thresh:
            print 'time less than thresh = ', item
            
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
    
    num_codes = min(len(oe),len(mw))

    # fit line:
    A = np.vstack([oe[0:1000], np.ones(len(oe[0:1000]))]).T ## ! might be dangerous to hardcode the 1000 here but using 
    # length = min(len(oe),len(mw)) ==== 13000 in the case for one exp for grat17 == bad result (likely because the matches suck beyond the very beginning)
    m,c = np.linalg.lstsq(A,mw[0:1000])[0]
    print 'm,c = ', m,c

    return m,c


def mw_to_oe_time(mw_time,m,c):  ### y=mx+c. ( mw=m(oe)+c |||||| oe = (mw-c)/m )
    #for m_time in mw_time:
    #    oe = (m_time - c)/m
    return (mw_time - c)/m

def oe_to_mw_time(oe_time,m,c):
    return m*(oe_time)+c


def sync_pixel_clock(vid_path, oe_path,head_data, oe_channels=[0, 1]):

    
    # make path to the Ch_0 and Ch_1 files:
    if oe_path[-1] == '/':
        ttls_path = oe_path + 'Ch_'
    else:
        ttls_path = oe_path + '/Ch_'

    [times,channels,directions] = titanspikes_ttl_extract.get_TTL_info(ttls_path,oe_channels[0],oe_channels[1])
    

    # times are in seconds.microseconds
    ephys_fs = 1

    experiment_length=[]


    print 'Experiment length = ', experiment_length    


    # duplicate every element of times and directions

    # rel_times_0 = times[channels==0]
    # rel_times_1 = times[channels==1]
    # rel_directions_0 = directions[channels==0]
    # rel_directions_1 = directions[channels==1]

    # plot_times_0 = np.array(list(itertools.chain(*zip(rel_times_0,rel_times_0[1:]))))
    # plot_times_1 = np.array(list(itertools.chain(*zip(rel_times_1,rel_times_1[1:]))))
    # plot_directions_0 = np.array(list(itertools.chain(*zip(rel_directions_0,rel_directions_0[:-1]))))
    # plot_directions_1 = np.array(list(itertools.chain(*zip(rel_directions_1,rel_directions_1[:-1]))))

    

    #oe_codes, latencies = pixelclock.events_to_codes(np.vstack((times, channels, directions)).T, len(oe_channels), 0.01,swap_12_codes =1,swap_03_codes=0)
   
    # the pixel clock should change once per frame, or at ~16ms max. This is 16ms * 30samples/ms = 480 samples. If a code is shorter than that, it's probably a fluke.
    # if oe_code times are in 636... format, use 10e4 as min code length
    #if in seconds.microseconds, min code time = 0.01

    #oe_codes = titanspikes_ttl_extract.read_raw_ttl(os.getcwd() + '/TTLIns') 


    #print 'Number of ephys codes = ', len(oe_codes)

   
    
    
    ############################ get Video's LED data (pre-extracted): 

    video_data = np.load(vid_path)

    vid_channels = video_data['channels']

    vid_directions = video_data['directions']

    vid_times = video_data['times']


    

    #head_data.time  = head_data.time / 1e3 ## convert from ms to sec

    ch1_diffs = np.diff(head_data.bit1) 
    ch2_diffs = np.diff(head_data.bit2)

    ch1_nonzero_diffs = np.nonzero(ch1_diffs)[0] ## [0] b/c where returns a stupid tuple
    ch2_nonzero_diffs = np.nonzero(ch2_diffs)[0]

    ch1_directions = ch1_diffs[ch1_nonzero_diffs]

    ch1_times = head_data.iloc[ch1_nonzero_diffs].converted_times.values

    ch1_channels = np.zeros(ch1_directions.shape[0])

    ch2_directions = ch2_diffs[ch2_nonzero_diffs]

    ch2_times = head_data.iloc[ch2_nonzero_diffs].converted_times.values

    ch2_channels = np.ones(ch2_directions.shape[0])

    ard_channels = np.hstack([ch1_channels,ch2_channels])
    ard_directions = np.hstack([ch1_directions,ch2_directions])
    ard_times = np.hstack([ch1_times,ch2_times])

    print 'ch1_times = ', ch1_times[0:10]
    print 'ch2_times = ', ch2_times[0:10]


    ard_codes,ard_latencies = pixelclock.events_to_codes(np.vstack((ard_times, ard_channels, ard_directions)).T, 2, 0.01,swap_12_codes = 1,swap_03_codes=1)
    
    vid_codes,vid_latencies = pixelclock.events_to_codes(np.vstack((vid_times, vid_channels, vid_directions)).T, 2, 0.01,swap_12_codes = 0,swap_03_codes=0)
    

    ### hack: instead of changing downstream variable names, just rename ard codes "oe_codes":
    #oe_codes = None
    #oe_codes = ard_codes

    ##### alternative method:
    # codes = []
    # times = []
    # for idx in range(head_data.shape[0]):
    #     tmp_sum = head_data.bit1.values[idx] +  head_data.bit2.values[idx]
    #     #print tmp_sum
    #     if tmp_sum == 1:
    #         if head_data.bit1.values[idx] == 1:
    #             tmp_code = 2
    #         elif head_data.bit2.values[idx] == 1:
    #             tmp_code = 1
    #     elif tmp_sum == 0:
    #         tmp_code = 0
    #     else:
    #         tmp_code = 3
    #     codes.append(tmp_code)
    #     times.append(head_data.time[idx])
    # # (0,0) = 0. code = 0
    # # (0,1) = 1 code = 1
    # # (1,0) = 1 code = 2
    # # (1,1) = 2 code = 3
    # codes = np.array(codes)
    # times = np.array(times)
    # uni_codes = codes[np.where(np.diff(codes))[0]]
    # uni_times = times[np.where(np.diff(codes))[0]]

    # ard_codes = zip(uni_times,uni_codes)

  
    print 'Number of Arduino codes = ', len(ard_codes) 
    print 'Number of Video codes = ', len(vid_codes) 
  
   #### special skipping first few codes (which are bad,mkay) to get better matches- 8/3/16 for grat17:
    #vid_codes = vid_codes[3:] #10096
    #oe_codes = oe_codes[2:]
    ard_codes = ard_codes[1:]

    vid_codes = lowpass_codetimes(vid_codes,30,1)

    print 'some arduino codes = ', ard_codes[0:10]

    # 3. get pixel clock matches
    def run_match(codes1,codes2):
        print '############################### PREPARING TO MATCH CODES #######################################'
        matches = []
        win_size = 400
        print 'win max is ',int(len(codes1)/win_size)
        for win in range(0,int(len(codes1)/win_size),5): #range(int(round(len(oe_codes)/win_size)))
        #for idx,win in enumerate(range(int(len(oe_codes)/win_size))):
            print 'win = ', win
            if win*win_size+win_size < len(codes1):
                ## don't go thru all arduino codes, but start at the previous time. (i.e. move forward!)
                if len(matches) > 1:
                    ard_idx = np.where([thing[0] == matches[-1][1] for thing in codes2])[0][0]
                    print 'ard idx and last match time = ', ard_idx,matches[-1][1]
                else:
                    ard_idx = 0

                tmp_match = pixelclock.match_codes(
                    [evt[0] for evt in codes1[win*win_size:(win+1)*win_size]], # oe times
                    [evt[1] for evt in codes1[win*win_size:(win+1)*win_size]], # oe codes
                    
                    [evt[0] for evt in codes2[ard_idx:]], # mw times
                    [evt[1] for evt in codes2[ard_idx:]], # mw codes
                    minMatch = 15,
                    maxErr = 0) 
                print 'temp matches = ', tmp_match
                matches.extend(tmp_match)
            else:
                print '!!win = ', win
                tmp_match = pixelclock.match_codes(
                        [evt[0] for evt in codes1[win*win_size:-1]], # oe times
                        [evt[1] for evt in codes1[win*win_size:-1]], # oe codes
                        [evt[0] for evt in codes2], # mw times
                        [evt[1] for evt in codes2], # mw codes
                        minMatch = 9,   
                        maxErr = 0)
                        
                matches.extend(tmp_match)

            return matches
    print 'len(vid_codes), len(ard_codes) = ',len(vid_codes), len(ard_codes) 
    matches = run_match(vid_codes,ard_codes)
    
    print 'matches = ', matches
    print 'type = ', type(matches)
   
    #ard_times = [item[0] for item in ard_codes] #[e.time for e in stimulus_announces if isiterable(e.value)]
    #oe_times = [item[0] for item in oe_codes]

       

    # condition the data to plot square pulses:
    tmp_ard_codes = [evt[1] for evt in ard_codes]
    tmp_mw_codetimes = [evt[0] for evt in ard_codes]
    plot_ard_codes = np.array(list(itertools.chain(*zip(tmp_ard_codes,tmp_ard_codes[:-1])))) 
    plot_mw_codetimes = np.array(list(itertools.chain(*zip(tmp_mw_codetimes,tmp_mw_codetimes[1:])))) 

    tmp_oe_codes = [evt[1] for evt in vid_codes]
    tmp_oe_codetimes = [evt[0] for evt in vid_codes]
    plot_oe_codes = np.array(list(itertools.chain(*zip(tmp_oe_codes,tmp_oe_codes[:-1])))) 
    plot_oe_codetimes = np.array(list(itertools.chain(*zip(tmp_oe_codetimes,tmp_oe_codetimes[1:])))) 



    # Bokeh:

    ############################################################### FIGURE 1 #####################################################

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
    s1.line(plot_mw_codetimes,plot_ard_codes)
    mw_match_circles = [mat[1] for mat in matches]
    mw_match_circles_samples = [mat[1] for mat in matches]

    s1.circle(mw_match_circles,match_idx,color=colors,size=20)
    
    #s1.circle(mw_match_circles,np.ones(len(matches)),color=colors,size=20)
    s1.yaxis.axis_label = 'Arduino Codes'

    #tap = s1.select(dict(type=TapTool))
    

    s2 = figure(width=1000, plot_height=500, title=None) #,tools = TOOLS
    s2.line(plot_oe_codetimes/ephys_fs,plot_oe_codes)
    oe_match_circles = [mat[0]/ephys_fs for mat in matches]
    oe_match_circles_samples = [mat[0] for mat in matches]
    
    
    s2.circle(oe_match_circles,match_idx,color=colors,size=20) # ,tags = match_idx
    #s2.circle(oe_match_circles,np.ones(len(matches)),color=colors,size=20) # ,tags = match_idx
    s2.xaxis.axis_label = 'Time (sec)'
    s2.yaxis.axis_label = 'Video Codes'

    


    p = gridplot([[s1], [s2]])
    output_file("pc_codes_match.html")
    # show the results
    show(p)
    
    #plt.savefig('pc_code_matches.pdf')
    #plt.show()


    #m,c = fit_line(tmp_oe_codetimes,tmp_mw_codetimes) #oe,mw 
    m,c = fit_line(oe_match_circles_samples,mw_match_circles_samples)
    # tb object lets you go back and forth between oe and mw timezones
    tb = timebase.TimeBase(matches,tmp_oe_codetimes,tmp_mw_codetimes)
    

    ## to test quality of match, plot OE codes in MW time

    print 'len plot_oe_codetimes = ', len(plot_oe_codetimes)



    


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
    
    

    tmp_oeMWconv_codetimes = [tb.audio_to_mworks(evt[0]/ephys_fs)  for evt in vid_codes]
    plot_oeMWconv_codetimes = np.array(list(itertools.chain(*zip(tmp_oeMWconv_codetimes,tmp_oeMWconv_codetimes[1:])))) 

    

    s1 = figure(width=1000, plot_height=500, title='Video Codes Plotted in Arduino Time')
    s1.line(plot_mw_codetimes,plot_ard_codes)
    
    #match_circles = [mat[1] for mat in matches]
    #s1.circle(match_circles,np.ones(len(matches)),color=colors,size=20)

    s1.yaxis.axis_label = 'Arduino Codes in Arduino Time'

    s2 = figure(width=1000, plot_height=500, title=None,x_range=s1.x_range,y_range=s1.y_range)
    s2.line(oe2mw_time,plot_oe_codes)
    s2.xaxis.axis_label = 'Time (sec)'
    s2.yaxis.axis_label = 'Video Codes in Arduino Time'
    
    p = gridplot([[s1], [s2]])
    output_file("vid_codes_in_Arduino_time.html")
    # show the results
    show(p)
    
    ####### FIGURE 3 ###########

    s1 = figure(width=1000, plot_height=500, title='Arduino Codes Plotted in Video Time')
    s1.line(mw2oe_time/ephys_fs,plot_ard_codes)
    
    #match_circles = [mat[1] for mat in matches]
    #s1.circle(match_circles,np.ones(len(matches)),color=colors,size=20)

    s1.yaxis.axis_label = 'Arduino Codes in Video Time'

    s2 = figure(width=1000, plot_height=500, title=None,x_range=s1.x_range,y_range=s1.y_range)
    s2.line(plot_oe_codetimes/ephys_fs,plot_oe_codes)
    s2.xaxis.axis_label = 'Time (sec)'
    s2.yaxis.axis_label = 'Video Codes in Video Time'
    
    p = gridplot([[s1], [s2]])
    output_file("ard_codes_in_Video_time.html")
    # show the results
    show(p)
    

    ####### FIGURE 4 ###########
    ####### PLOT the LINE fit:
    
    pp = figure(width=1000, plot_height=500, title='Line Fit for MW and OE Time')
    
    #pp.line(oe2mw_time/1e6,plot_oe_codes)
    pp.line(plot_oe_codetimes,m*plot_oe_codetimes+c,color='red')
    #pp.circle(plot_oe_codetimes[0:len(matches)],plot_mw_codetimes[0:len(matches)])
    pp.circle(oe_match_circles_samples,mw_match_circles_samples)
    pp.xaxis.axis_label = 'video codetimes'
    pp.yaxis.axis_label = 'arduino codetimes'
    output_file("pc_line_fit.html")
    show(pp)


    #print "number of Arduino events:"
    #print len(ard_times)

    #print "number of OE events:"
    #print len(oe_times)

    print "number of matches: " + str(len(matches))
    

    


    return matches,m,c,experiment_length

#------------------------------------------------------------------------------------- 




if __name__ == "__main__":

    #mwk_file = sys.argv[1]
    #oe_path = sys.argv[2]

    # 1. get KWIK and MWK files in current directory
    #ard_path,oe_path = get_files()

    
   
    oe_path = sys.argv[1]
    video_path = sys.argv[2]
    head_path = sys.argv[3]

    print 'video_path path is ', video_path

    print 'oe path is ', oe_path

    #names = ['time','bit1','bit2','ox','oy','oz','ax','ay','az']
    #head_data = pd.read_csv(ard_path,names=names)

    head_data = pd.read_csv(head_path)
    #head_data = head_data[10:]

    matches,m,c,experiment_length = sync_pixel_clock(video_path,oe_path, head_data, oe_channels=[2,3])
    

    #converted_times = []
    # for head_time in head_data.time:
    #     oe_tmp = mw_to_oe_time(head_time,m,c)
    #     converted_times.append(oe_tmp)

    #converted_times = np.array(converted_times)

    video_converted_times = mw_to_oe_time(head_data.converted_times,m,c)

    xxx = oe_to_mw_time(head_data.converted_times,m,c)


    print 'head_data.time.iloc[0:10] = ', head_data.converted_times.iloc[10:20]

    print 'mw_to_oe_time 40 = ', mw_to_oe_time(40,m,c)

    print 'oe_to_mw_time 40 = ', oe_to_mw_time(40,m,c)


    head_data['video_converted_times'] = video_converted_times

    head_data.to_csv('video_head_data.csv')

    #head_data.to_pickle('head_data')

















