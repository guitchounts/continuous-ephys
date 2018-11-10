import numpy as np
import h5py
import sys

def convert(kwd_file_name):

	input_h5file = h5py.File(kwd_file_name,'a')
	ephys = input_h5file['/recordings/0/data'][:] # the [:] means ephys is a numpy array rather than an H5py data structure

	input_h5file.close()

	exp_name = kwd_file[0:kwd_file.find('.raw.kwd')]

	np.save(exp_name,ephys) ## actually saves as .npy


if __name__ == "__main__":

	kwd_file = sys.argv[1] # e.g. 'experiment1_100.raw.kwd'

	convert(kwd_file)