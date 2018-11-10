import sys
import os
import re

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

def get_files(file_type,input_file_path):
    # type = ".kwik" or ".kwd"    
    raw_files = []
    for file in os.listdir(input_file_path):
        if file.endswith(file_type):
            raw_files.append(file)

    raw_files.sort(key=alphanum_key)
    sort_nicely(raw_files)
    return raw_files
    

if __name__ == "__main__":

    file_type = sys.argv[1]
    input_file_path = os.getcwd()
    sorted_files = get_files(file_type,input_file_path);

    print sorted_files