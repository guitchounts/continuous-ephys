import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import pandas as pd
import sys

def get_impedance(dir):

	tree_before = ET.parse(dir)
	root_before = tree_before.getroot()

	impedances = []
	for channel in root_before:
	    impedances.append(float(channel.attrib['magnitude']))

	return np.asarray(impedances)

if __name__ == "__main__":
	# get "before" file:
	#tree_before = ET.parse('./Carbon Fiber Arrays/160501.1_G64optetrode/before_plating.xml')
	

	### get number all conditions:
	num_conditions = len(sys.argv)-1


	condition_names = [thing[0:thing.find('/imp')] for thing in sys.argv[1:]]
	print condition_names


	trees = []
	roots = []
	impedances = {condition : [] for condition in condition_names}


	for i in range(num_conditions):
		trees.append(ET.parse(sys.argv[i+1]))
		roots.append(trees[i].getroot())

		for channel in roots[i]:
			impedances[condition_names[i]].append(float(channel.attrib['magnitude']))
	 
		if len(impedances[condition_names[i]]) == 32:
			impedances[condition_names[i]] = impedances[condition_names[i]][8:24]

	num_electrodes = len(impedances[condition_names[i]])
	print num_electrodes
	#print impedances
	# print len(impedances)
	# for condition in condition_names:
	# 	print condition, np.median(impedances[condition])
	medians = [np.median(impedances[condition])/1e6 for condition in condition_names]
	b = pd.DataFrame(impedances)
	#print b.shape
	#print b

	fig = plt.figure(dpi=600)
	ax = plt.subplot(111)
	fig.suptitle(['%.2f MOhm' % median for median in medians])


	sns.boxplot(data=b,order=condition_names,palette="Set2",ax=ax) #### !!! [::-1] reverses the order of the keys and values - this plots the conditions in the right order
	sns.stripplot(data=b, color=".25",order=condition_names,alpha=0.25,ax=ax)


	for i in range(num_electrodes):
	#	print 'range(num_conditions) === ', range(num_conditions)
	#	print 'impedances.values() === ', impedances.values()
		ax.plot( range(num_conditions), b.values[i,:], c='k',alpha=0.25) #### !!! reversing the order of the conditions

	ax.set_ylabel('Impedance (Ohms)')
	ax.set_yscale('log')
	#ax.set_ylim([])
	ax.set_yticks([1e4,1e6,1e8])
	ax.set_ylim([1e4,1e8])
	#ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax.tick_params(axis='y',which='major',length=10,width=1)
	ax.tick_params(axis='y',which='minor',length=5,width=.5)

	sns.despine(bottom=True)


	#ax.add_artist(legend1) # add l1 as a separate artist to the axes
	#ax.set_yscale('log')
	#fig = ax.get_figure()


	end_str = sys.argv[2].find('/imp')
	#save_name = str(sys.argv[2][2:end_str]) + '.pdf'
	save_name = 'impedances.pdf'
	fig.savefig(save_name)