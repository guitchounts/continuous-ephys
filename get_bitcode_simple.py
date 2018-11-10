import os
import multiprocessing
import datetime
import pymworks
import matplotlib.pyplot as plt
import numpy as np

def get_bitcode_simple(animal_name, session_filename):
    

    
    path = '/Volumes/Mac HD/Dropbox (coxlab)/Behavior/3-port-analysis-master/' + 'input/' + 'phase1/' + animal_name + '/' + session_filename
    
    display_events= ['#stimDisplayUpdate']

    df = pymworks.open_file(path)
    events = df.get_events(display_events)
    
    
    
    temp_events = []
    last_announce = None
    trial_num = 0

    result = {'bit_code':[],'stim_name':[],'time':[]}

    index = 0
    while index < len(events):
    #    result['bit_code'].append()
        trial_result = {'bit_code':[],'stim_name':[],'time':[]}
    
        if events[index].name == '#stimDisplayUpdate' and len(events[index].value) == 1: # this won't actually happen
            pass
        elif events[index].name == '#stimDisplayUpdate' and len(events[index].value) == 2:



            trial_result['bit_code'] = events[index].value[1]['bit_code'] # this is the actual bit code on this screen
            trial_result['stim_name'] = events[index].value[0]['name'] # this should be BlankScreen
            trial_result['time'] =  events[index].time
            


        elif events[index].name == '#stimDisplayUpdate' and len(events[index].value) == 3:

            if events[index].value[2]['name'] == 'pixel clock':
                    
                    trial_result['stim_name'] = events[index].value[1]['name']
                    trial_result['bit_code'] = events[index].value[2]['bit_code']
                    trial_result['time'] =  events[index].time
                    

                  # elif [2] value of ev.value is gray screen (i.e. this is an error trial),
            elif events[index].value[2]['name'] == 'BlankScreenGray':
                    # bit code is under ev.value[1] 
                    trial_result['bit_code'] = events[index].value[1]['bit_code']
                    trial_result['stim_name'] = events[index].value[2]['name']
                    trial_result['time'] =  events[index].time
                    

            else:
                #print ev.value[2]['name']
                pass

        result['bit_code'].append(trial_result['bit_code'])
        result['stim_name'].append(trial_result['stim_name'])
        result['time'].append(trial_result['time'])

        index += 1

    #print result

    return result

if __name__ == '__main__':
    
    animal_name = 'test'
    session_filename = 'test_150501.mwk'
    get_bitcode_simple(animal_name,session_filename)

