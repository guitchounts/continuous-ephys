import os
import multiprocessing
import datetime
import pymworks
import matplotlib.pyplot as plt
import numpy as np

def get_session_statistics(animal_name, session_filename):
    '''
    Returns a time-ordered list of dicts, where each dict is info about a trial.
    e.g. [{"trial_num": 1,
           "behavior_outcome": "failure",
           "stm_pos_x": 7.5,
           },
          {"trial_num": 2,
           "behavior_outcome": "success",
           "stm_pos_x": -7.5
           }]
    NOTE: trial_num: 1 corresponds to the FIRST trial in the session,
    and trials occur when Announce_TrialStart and Announce_TrialEnd
    events have success, failure, or ignore events between them with
    value = 1.

    :param animal_name: name of the animal string
    :param session_filename: filename for the session (string)
    '''

    #TODO: unfuck this: hard coded paths not ideal for code reuse
    path = 'input/' + 'phase1/' + animal_name + '/' + session_filename
    
    # events_we_want = ['#stimDisplayUpdate','Announce_TrialStart', 'Announce_TrialEnd','success', 'failure', 'ignore']

    df = pymworks.open_file(path)

    # Start by getting the pixel clock / bit code data
    stimulus_announces = df.get_events['#announceStimulus']

    # bit_codes is a list of (time, code) tuples
    bit_codes = [(e.time, e.value['bit_code']) for e in stimuli if 'bit_code' in e.value]

    # ok, now let's grab the pixel clock events out of the open-ephys file

    


    
    result = []
    index = 0
    temp_events = []
    last_announce = None
    trial_num = 0
    while index < len(events):
        if events[index].name == 'Announce_TrialStart':
            temp_events = []
            last_announce = 'Announce_TrialStart'

        elif events[index].name == 'Announce_TrialEnd': 
            if last_announce == 'Announce_TrialStart':
                trial_result = {}
                for ev in temp_events:
                    
                    ### ANIMAL RESPONSE ###
                    if ev.name == 'success' and ev.value == 1:
                        trial_result['behavior_outcome'] = 'success'
                        trial_result['time_trial_response'] = ev.time
                    elif ev.name == 'failure' and ev.value == 1:
                        trial_result['behavior_outcome'] = 'failure'
                        trial_result['time_trial_response'] = ev.time
                    elif ev.name == 'ignore' and ev.value == 1:
                        trial_result['behavior_outcome'] = 'ignore'
                        trial_result['time_trial_response'] = ev.time

                    ### SCREEN INFORMATION ###
                    #elif ev.name == '#stimDisplayUpdate':
                     # print ev.value[0]['name']
                    elif ev.name == '#stimDisplayUpdate' and len(ev.value) == 1: # this won't actually happen
                        pass
                    elif ev.name == '#stimDisplayUpdate' and len(ev.value) == 2: # if blank screen with pixel clock on.
                        trial_result['bit_code'] = ev.value[1]['bit_code'] # this is the actual bit code on this screen
                        trial_result['stim_name'] = ev.value[0]['name'] # this should be BlankScreen
                      #  print 'stim: ', trial_result['stim_name'], ' and bit code: ', trial_result['bit_code']
                    elif ev.name == 'get_#stimDisplayUpdate' and len(ev.value) == 3: 
                      
                      # two options here: 1)blank screen, stimulus, and pixel clock
                      # or 2) blank, pixel, grayscreen (error trial)

                      # if [2] element of ev.value is bit code, do this:
                      if ev.value[2]['name'] == 'pixel clock':
                        trial_result['time_trial_start'] = ev.time
                        trial_result['stim_name'] = ev.value[1]['name']
                        trial_result['bit_code'] = ev.value[2]['bit_code']
                        #print "bit code as ev.value[2]['name']"    

                      # elif [2] value of ev.value is gray screen (i.e. this is an error trial),
                      elif ev.value[2]['name'] == 'BlankScreenGray':
                        # bit code is under ev.value[1] 
                        trial_result['bit_code'] = ev.value[1]['bit_code']
                        trial_result['stim_name'] = ev.value[2]['name']
                        #print "gray screen as ev.value[2]['name']"

                      else:
                        #print ev.value[2]['name']
                        pass
                      
                    

                   ### STIMULUS PROPERTIES ###
                   # elif ev.name == 'stm_size':
                   #     trial_result['stm_size'] = ev.value
                   # elif ev.name == 'stm_rotation':
                   #     trial_result['stm_rotation'] = ev.value
                   # elif ev.name == 'stm_rotation_in_depth':
                   #     trial_result['stm_rotation_in_depth'] = ev.value
                   # elif ev.name == 'stm_pos_x':
                    #    if trial_result['stm_pos_x'] == -0.0:     #weird MWorks protocol thing where some of the 0.0s come out negative, so just correcting that
                     #       trial_result['stm_pos_x'] = abs(ev.value)
                      #  else:
                       #     trial_result['stm_pos_x'] = ev.value
                   # elif ev.name == 'stm_pos_y':
                   #     trial_result['stm_pos_y'] = ev.value
                    else:
                        pass
                        
                    print 'bit code was: ', trial_result['bit_code'], 'and stim was: ', trial_result['stim_name']


                if 'behavior_outcome' in trial_result:
                    trial_num += 1
                    trial_result['trial_num'] = trial_num
                    result.append(trial_result)

            last_announce = 'Announce_TrialEnd'

        else:
            temp_events.append(events[index])
        index += 1
    #FYI, testing showed some good filtering of weird events here...
    #blah = df.get_events(["success", "failure", "ignore"])
    #print "EVENTS EQUAL? ", len(result) == len(blah) - 6, session_filename
    #subtract 6 because session initialization emits 2 behavior outcomes per
    #outcome type
    #print len(result), len(blah) - 6
    #lines above unequal in 6/77 sessions for AB3&7 because of random behavior
    #outcome events firing in rapid succession. They happen within a couple
    #microseconds of one another so filtering these out is probably good
    return result
    print result

if __name__ == '__main__':
    
    animal_name = 'test'
    session_filename = 'test_150501.mwk'
    get_session_statistics(animal_name,session_filename)