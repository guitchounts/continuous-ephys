import os
import re

####### File sorting functions: ########
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
    sorted_l = l.sort(key=alphanum_key)
    return sorted_l

def get_raw_files(file_key,path=os.listdir(os.getcwd())):
    raw_files = []
    for file in path:
        #if file.endswith(file_extention):
        #    raw_files.append(file)
        if file.find(file_key) != -1:
            raw_files.append(file)
            
    sort_nicely(raw_files)   
    return raw_files