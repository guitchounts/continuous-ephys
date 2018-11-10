import numpy as np

def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]

def windowed_histogram(trace,time,window_size,step): ## for spike histogram, trace should be a 0's and 1's vector with length = trial length (1's in places where spikes occur)
    out_x = []              ## time = vector in samples. 
    out_y = []
    y = slidingWindow(trace,window_size,step) #window_size/5
    x = slidingWindow(time,window_size,step) ### get centers of the time
    for value in y:
        out_y.append(np.sum(value)) ## take the number of spikes in this window
        
    for t in x:
        out_x.append(np.median(t)) ## take the median time, i.e. the center of the time window
    
    return out_x,out_y