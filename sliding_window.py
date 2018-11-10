import numpy as np



def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    print nrows
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))



def windowed_firing_rate(spike_times,win,step,bins,spike_fs=3e4):

    #begin_time = 0
    #end_time = ChGroup_11_good.times.values[-1]


    #spike_time_vec = np.arange(begin_time,end_time,1/spike_fs)



    #bins  = strided_app(spike_time_vec,win*spike_fs,step*spike_fs) ### these are all entries of time vector in each window

    ## take just the start and stop times of each window:
    edges = bins[:,[0,-1]] ### e.g. [0,1],[0.1,1.1] etc. 





    firing_rate_vec = np.empty(bins.shape[0])
    ## loop through edges and take number of spikes in each edge:
    for idx,edge in enumerate(edges):
        # e.g. [0,1] seconds:
        #start = int(edge[0] * spike_fs)
        #stop = int(edge[1] * spike_fs)

        num_spikes_in_window = len(spike_times[(spike_times < edge[1]) & (spike_times > edge[0])])
       


        firing_rate_vec[idx] = num_spikes_in_window / win


    return firing_rate_vec,np.mean(edges,axis=1) ### the mean is the time stamps 