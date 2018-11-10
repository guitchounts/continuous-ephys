import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import pandas as pd
import sys

if __name__ == "__main__":
	# get "before" file:
	#tree_before = ET.parse('./Carbon Fiber Arrays/160501.1_G64optetrode/before_plating.xml')
	#tree_before = ET.parse(sys.argv[1])
	#root_before = tree_before.getroot()

	#impedances_before = []
	#for channel in root_before:
	#    impedances_before.append(float(channel.attrib['magnitude']))
	    
	# get "after" file:
	#tree_after =ET.parse('./Carbon Fiber Arrays/160501.1_G64optetrode/acid_pedotCNT.xml')
	tree_after = ET.parse(sys.argv[1])
	root_after = tree_after.getroot()

	impedances_after = []
	for channel in root_after:
	    impedances_after.append(float(channel.attrib['magnitude']))
	#print impedances_before 

	if len(impedances_after) == 32:
		impedances_after = impedances_after[8:24]
		#impedances_before = impedances_before[8:24]
		print 'length 32!'

	# plot:

	b = pd.DataFrame(data={
	    'fire-sharpened': impedances_after})

	ax = sns.boxplot(data=b,order=['fire-sharpened'],palette="Set2")
	ax = sns.stripplot(data=b, color=".25",order=['fire-sharpened'],alpha=0.25)
	#ax.set_ylim([0,2e7])

	for i in range(len(impedances_after)):
	    plt.plot( [0],impedances_after[i], c='k',alpha=0.25)

	sns.despine(offset=10, trim=True)
	
	#before_mean = np.mean(impedances_before)/1e6
	#before_median = np.median(impedances_before)/1e6
	after_mean = np.mean(impedances_after)/1e6
	after_median = np.median(impedances_after)/1e6

	legend1 = ax.legend([after_mean,after_median],loc=2)
	#legend2 = ax.legend([after_mean,after_median],loc=1)
	#legend2 = ax.legend(['before mean: %f M-Ohm' % after_mean, 'before median: %f M-Ohm' % after_median ],loc=4)


	#l1 = legend([ax], ["Label 1"], loc=1)
	#l2 = legend([ax], ["Label 2"], loc=4) # this removes l1 from the axes.
	ax.add_artist(legend1) # add l1 as a separate artist to the axes
	ax.set_yscale('log')
	fig = ax.get_figure()
	#fig.ylim([0,10000000])
	#axes = fig.gca()
	#axes.set_ylim([0,1e7])
	end_str = sys.argv[1].find('/imp')
	save_name = str(sys.argv[1][2:end_str]) + '.pdf'
	print save_name
	fig.savefig(save_name)
	#sns.violinplot(data=b,split=True, inner="stick", palette="Set3");

