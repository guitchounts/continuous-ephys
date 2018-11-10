#!/usr/bin/env python

import copy, logging

import numpy as np
from scipy.stats import linregress

# new methods are much faster for longer matches sequences (us compared to ms!)

class TimeBase(object):
    """
    Timebase object used to convert times from mworks to audio and back
    """
    def __init__(self, matches,oe_times,mw_times, cull = False, fitline = True):
        """
        Parameters
        ----------
        matches : list of tuples
            Matching time stamps for audio and mw clocks where
            matches[:,0] are audio times
            matches[:,1] are mw times

            oe_times = times for OE pixel clock codes
            mw_times = times for MW pixel clock codes 
        cull : bool
            Cull offsets (remove large changes in offset)
        """
        self.matches = np.array(copy.deepcopy(matches))
        self.matches = self.matches[self.matches[:,0].argsort(),:] # sort array by first column
        
        # convert matches to seconds
        self.matches[:,0] = self.matches[:,0] / 30e3 # samples / (30e3 samples / sec)
        self.matches[:,1] = self.matches[:,1] / 1e6 # microseconds / (1e6 microseconds / sec)
    
        self.oe_times = oe_times
        self.mw_times = mw_times
        
        # offsets are a - b so
        #   a - offset = b
        #   b + offset = a
        self.offsets = self.matches[:,0] - self.matches[:,1]

        print 'offsets = ', self.offsets 
        
        if cull: self.cull_offsets()

        if fitline: self.gg_fit_line()
    
    def gg_fit_line(self):  #### oe = x; mw = y variable. 
        # fit line:

        ## 
        num_codes = min(len(self.oe_times),len(self.mw_times))

        A = np.vstack([self.oe_times[0:num_codes], np.ones(num_codes)]).T ## ! might be dangerous to hardcode the 1000 here but using 
        # length = min(len(oe),len(mw)) ==== 13000 in the case for one exp for grat17 == bad result (likely because the matches suck beyond the very beginning)
        print 'A.shape = ', A.shape
        #print 'len(self.mw_times[0:len(self.oe_times)+1]) = ', len(self.mw_times[0:len(self.oe_times)+1])
        print 'len(self.mw_times[0:num_codes] )  = ', len(self.mw_times[0:num_codes])
        m,c = np.linalg.lstsq(A,self.mw_times[0:num_codes])[0]
        print 'm,c = ', m,c

        self.m = m
        self.c = c
        #return m,c


    def mw_to_oe_time(self,mw_time):  ### y=mx+c. ( mw=m(oe)+c |||||| oe = (mw-c)/m )
        #for m_time in mw_time:
        #    oe = (m_time - c)/m

        return (mw_time - self.c)/self.m

    def oe_to_mw_time(self,oe_time):
        return self.m*(oe_time)+self.c



    def cull_offsets(self, thresh = 0.03):
        """
        Remove offsets which differ from the previous offset by thresh seconds
        
        Parameters
        ----------
        thresh : float
            If the difference between the previous and current offsets is >= thresh
            than remove the current offset
        """
        deltaOffsets = self.offsets[1:] - self.offsets[:-1]
        goodIndices = np.where(abs(deltaOffsets) < thresh)[0]+1
        self.offsets = self.offsets[goodIndices]
        self.matches = self.matches[goodIndices]
    
    def fit_line(self):
        """
        Fit regression line to offsets and use that for all time matching
        """
        x = self.matches[:,0]
        y = self.matches[:,0] - self.matches[:,1]
        slope, offset, _, _, _ = linregress(x, y)
        self.offsets = x * slope + offset
    
    def audio_to_mworks(self, audio):
        """
        Convert an audio time (in seconds) to mworks time (in seconds)
        
        Parameters
        ----------
        audio : float
            Audio time (in seconds)
        """
        
        

        closest = np.where(self.matches[:,0] >= audio)[0]
        if len(closest) == 0:
            #logging.warning("audio_time_to_mw matched to last offset")
            return audio - self.offsets[-1]
        
        
        return audio - self.offsets[closest[0]]
    
    def mworks_to_audio(self, mw):
        """
        Convert a mworks time (in seconds) to audio time (in seconds)
        
        Parameters
        ----------
        mw : float
            MWorks time (in seconds)
        """
        if getattr(mw, '__iter__', False):
            print("getattr(mw, '__iter__', False)")
            audio_times = [None]*len(mw)
            cursor = 0
            for (t,m) in enumerate(mw):                
                matched_time = None
                
                while cursor < self.matches.shape[0]:
                    
                    current_match_time = self.matches[cursor,1]
                    
                    if current_match_time >= m:
                        matched_time = m + self.offsets[cursor]
                        break
                    cursor += 1
                
                if matched_time is None:
                    #logging.warning("mw_time_to_audio matched to last offset")
                    audio_times[t] = m + self.offsets[-1]
                else:
                    audio_times[t] = matched_time
            return audio_times
        else:
            print 'len self.matches[:,1] ', len(self.matches[:,1])
            print 'len mw ', len(mw)
            closest = np.where(self.matches[:,1] >= mw)[0]
            if len(closest) == 0:
                #logging.warning("mw_time_to_audio matched to last offset")
                return mw + self.offsets[-1]
            return mw + self.offsets[closest[0]]
    
    def old_mw_time_to_audio(self, mw_time, mw_offset = 0):
        """
        Depreciated method, use: mworks_to_audio
        """
        mw_t = mw_time + mw_offset
        # print mw_t
        for (i, evt_match) in enumerate(self.matches):
            # print evt_match
            # if mw_t > evt_match[1]:
            if evt_match[1] >= mw_t:
                # simple "one point" matching for now
                return mw_t + self.offsets[i]# + self.audio_offset
        
        logging.warning("mw_time_to_audio matched to last offset")
        return mw_t + self.offsets[-1]# + self.audio_offset

    def old_audio_time_to_mw(self, audio_time, audio_offset = 0):
        """
        Depreciated method, use: audio_to_mworks
        """
        a_t = audio_time + audio_offset# - self.audio_offset
        
        for (i, evt_match) in enumerate(self.matches):
            # if a_t > evt_match[0]:
            if evt_match[0] >= a_t:
                return a_t - self.offsets[i]
        
        logging.warning("audio_time_to_mw matched to last offset")
        return a_t - self.offsets[-1]

def test_timebase():
    audio = np.linspace(0., 100., 1000)
    mw = audio + 1000. #np.linspace(1000., 1100., 10)
    matches = np.transpose(np.vstack((np.transpose(audio),np.transpose(mw))))
    
    tb = TimeBase(matches)
    
    # exact matches won't happen due to floating point issues
    for match in matches:
        a, m = match
        d = abs(tb.mworks_to_audio(tb.audio_to_mworks(a)) - a)
        assert d < 1E-9, "Audio->MW->Audio failed: audio: %.6f mw: %.6f Err: %.6f" % (a, m, d)
        d = abs(tb.audio_to_mworks(tb.mworks_to_audio(m)) - m)
        assert d < 1E-9, "MW->Audio->MW failed: mw: %.6f audio: %.6f Err: %.6f" % (m, a, d)
        d = abs(a - tb.mworks_to_audio(m))
        assert d < 1E-9, "MW->Audio failed: mw: %.6f, audio: %.6f Err: %.6f" % (m, a, d)
        d = abs(m - tb.audio_to_mworks(a))
        assert d < 1E-9, "Audio->MW failed: audio: %.6f mw: %.6f Err: %.6f" % (a, m, d)

def test_timebase_batch():
    audio = np.linspace(0., 100., 10000)
    mw = audio + 1000. #np.linspace(1000., 1100., 10)
    matches = np.transpose(np.vstack((np.transpose(audio),np.transpose(mw))))
    
    tb = TimeBase(matches)
    
    import time
    tic = time.time()
    batch_at = tb.mworks_to_audio(matches[:,1])
    batch_time = time.time()-tic
    
    tic = time.time()
    unbatch_at = [tb.mworks_to_audio(m) for m in matches[:,1]]
    unbatch_time = time.time()-tic
    
    print("Batch time = %f, unbatch time = %f" % (batch_time, unbatch_time))
    
    assert(np.allclose(np.array(batch_at), np.array(unbatch_at)))

if __name__ == "__main__":
    test_timebase_batch()
