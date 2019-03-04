#!/usr/bin/env python
import gzip
import sys
import numpy as np
import networkx as nx
import random
import math
import pandas as pd
import scipy as sp
from scipy import stats
from numba import jit

# could be omitted
from collections import deque
import time

def updateSysPopForCommonCompetingMemes(d, new_d):
	for k in d.keys():
		if k in new_d:
			d[k].append(new_d[k])
		elif k in d:
			del d[k]
	return d

@jit
def entropy(vec):
	"""Returns the entropy in the elements of the input vector. """
	temp = np.unique(np.asarray(vec), return_counts=True)[1]
	prob_vec = temp/float(sum(temp))
	return prob_vec.dot(-np.log10(prob_vec))
#========== PARAMETERS ========== 

n = 1000 # nodes
m = 10
p = 1.0 # probability to choose existing meme
nn = 100000 # timesteps 10,000,000
memory_size = 10
increment = 10000 # interval to write
first_write = 100 # step of first write 
write_step = first_write # write step tracker
track_memes_after = nn/10 # only save memes after this step
max_memes_track = 10 # number of memes to track after track_memes_after
track_competing_memes = 0 # number of memes to track
meme_fitness_thresh = 1. # threshold for competing memes
#================================ 

print "******** PARAMS *********"
print 'Network: BA: n:{0}, m:{1}, memory:{2}'.format(n, m, memory_size)
print 'Introduction rate: {0}'.format(1.0-p)
print 'Total time steps in this simulation: {0}'.format(nn)
print 'Track memes after {0} steps'.format(track_memes_after)
print 'Max memes to track: {0}'.format(max_memes_track)
print 'No. of competing memes to track: {0}'.format(track_competing_memes)
print 'Fitness threshold for competing memes: {0}'.format(meme_fitness_thresh)
print 'Writing proceeds at an increment of {0} steps starting at {1}'.format(increment, first_write)
print "*************************"



prefix = 'BA_n' + str(n) + '_m' + str(m) + '_mu' + str(p) + '_t' + str(nn) + '_'+ 'memory_size' + str(memory_size) + '_'
prefix=prefix

G = nx.barabasi_albert_graph(n,m)
# adj = G.adjacency_list() # adjacency list
adj = {}
for ky in range(0,n):
   adj[ky]=list(G.neighbors(ky))

# network meme matrix: initialization
nodememory = sp.random.rand(n, memory_size)
print nodememory,'\n'

for i01 in range(0,n):
	nodememory[i01]=sorted(nodememory[i01],reverse=True)
print nodememory,'\n'

### espacamento logaritmico
nume=100
log_space1=np.logspace(0.6, np.log10(nn), num=nume)
log_space1 = list(set([int(i) for i in log_space1]))

nume=1000
log_space2=np.logspace(0.6, np.log10(nn), num=nume)
log_space2 = list(set([int(i) for i in log_space2]))

log_space=list(set(log_space1+log_space2))
log_space.append(float(1))
log_space.append(float(2))
log_space = list(set([int(i) for i in log_space]))
log_space=sorted(log_space, key=int)

nume=len(log_space)
cont_log_space=0
write_step=int(log_space[cont_log_space])
rug1={}
rug2={}
v1=0
v2=0
for ix in range(0,len(log_space)):
    rug1[int(log_space[ix])]=0
    rug2[int(log_space[ix])]=0

# meme tracking: k:v => k:meme, v:dict={start, end, count, nodeset}
# 'simulatenous_popularity': maximum number of nodes affected by this meme across time, when selected
# 'max_popularity': number of nodes that have this meme in their memory
# 'system_popularity': number of memory locations held by this meme across the entire n/w
memes = {}
for meme in np.unique(nodememory):
	temp = len(np.unique(np.where(np.any(nodememory == meme,1))[0]))
	memes[meme] = {'start':1, 'end':1, \
				   'simulatenous_popularity':0, \
				   'max_popularity': temp,\
				   'lastnodeaffected':-1}

# System stats
steps = []
rug= []
rug_M= []
unique_memecount = [] # Number of unique memes at each timestep	
system_entropy = [] # Entropy in memes overall
user_entropy = np.apply_along_axis(entropy, 1, nodememory)
avg_user_entropy = [] # entropy in memes per user, averaged across the n/w
competing_memes = []
common_competing_memes = {} # Common competing memes since saturation. 

d = 10
progress = deque(list(np.linspace(1, d, d)*nn/d) + [nn-1])
starttime = time.time()

meme_count_after_saturation = 0
max_meme_size = 0

# meme_salvar=open('memes.csv','w')
# begin simulation
print "Please wait while the simulation completes.."
t1 = time.time()
for counter in xrange(nn):
	counter = counter + 1
	if(counter >= progress[0]):
		print "{0}% done. Elapsed time: {1}".format(int(100 * progress[0]/nn), time.time()-t1)
		progress.popleft()
		print "Number of common competing memes: {0}".format(len(common_competing_memes))
		
	select_one_node = random.randint(0,n-1)

	probability_new_idea = random.uniform(0,1) # mu
	if probability_new_idea <= p:
		meme_chosen = random.uniform(0,1) # new meme
		xaffectednodes =  adj[select_one_node]+[select_one_node]
	else:
		meme_probability = random.uniform(0,1)
		temp = np.cumsum(nodememory[select_one_node, ])/sum(nodememory[select_one_node, ])
		meme_idx = np.where(temp >= meme_probability)[0][0]
		meme_chosen = nodememory[select_one_node, meme_idx] # selected meme
		aux_xaffectednodes = adj[select_one_node]
		aux_nodes=[]
		for nodes_affec in aux_xaffectednodes:
			if meme_chosen not in nodememory[nodes_affec]:
				aux_nodes.append(nodes_affec)
		xaffectednodes=aux_nodes
	# print >>meme_salvar,meme_chosen
	# affect the neighbors
	affectednodes=xaffectednodes
	temp = nodememory[affectednodes, :memory_size-1].copy()
	lastmemes = set(nodememory[affectednodes, memory_size-1])
	nodememory[affectednodes, 0] = meme_chosen
	nodememory[affectednodes, 1:] = temp
	nodememory[affectednodes]=-np.sort(-nodememory[affectednodes])


	
	# remove memes not found anywhere in the n/w
	for lm in lastmemes: 
		if (not np.any(nodememory == lm)) and (lm != meme_chosen) and (lm in memes):
			memes[lm]['end'] = counter
			if memes[lm]['start'] < track_memes_after:
				del memes[lm]
	if len(memes) > max_meme_size:
		max_meme_size = len(memes)

	# update meme set
	max_pop = len(set(np.where(np.any(nodememory == meme_chosen, 1))[0]))
	if meme_chosen in memes:
		if memes[meme_chosen]['simulatenous_popularity'] < len(affectednodes):
			memes[meme_chosen]['simulatenous_popularity'] = len(affectednodes)
		if memes[meme_chosen]['max_popularity'] < max_pop:
			memes[meme_chosen]['max_popularity'] = max_pop
		memes[meme_chosen]['lastnodeaffected'] = select_one_node
	else:
		if meme_count_after_saturation < max_memes_track:
			if counter >= track_memes_after:
				meme_count_after_saturation += 1
			memes[meme_chosen] = {'start':counter, 'end':1, \
								'simulatenous_popularity':len(affectednodes), \
								'max_popularity':max_pop, \
								'lastnodeaffected':select_one_node}
		if meme_count_after_saturation + 1 == max_memes_track:
			print "Maximum no. of requested memes achieved at timestep: ", counter


	aux001=len(np.unique(nodememory))
	v1=v1+aux001
	aux2=v1/(counter)

	v2=v2+np.mean(nodememory)
	aux3=v2/(counter)


	
	# system entropy
#	if ( counter <= first_write ) or (counter > first_write and counter == write_step):
	if  int(counter) == int(write_step):
		if counter == first_write:
			print "Time for first write:", time.time() - t1
		steps.append(counter)
		rug1[(counter)]=rug1[(counter)]+aux2
		rug2[(counter)]=rug2[(counter)]+aux3
		
		rug.append(rug1[(counter)])
		rug_M.append(rug2[(counter)])
		unique_memecount.append(aux001) # update meme count for this step		
		system_entropy.append(entropy(nodememory.ravel()))
		# avg. user entropy
		#user_entropy[affectednodes] = np.apply_along_axis(entropy, 1, nodememory[affectednodes,])
		#avg_user_entropy.append(np.mean(user_entropy))
		cont_log_space=cont_log_space+1
		if cont_log_space<nume:
	
			print cont_log_space,log_space[cont_log_space],counter,"cont_log_space"
			write_step=int(log_space[cont_log_space])
		

	# compute system popularity
	# meme_count_after_saturation = len({m for m, v in memes.iteritems() if v['start'] >= track_memes_after})
	if (counter >= track_memes_after) and (meme_count_after_saturation < max_memes_track):
		# find max X competing memes
		mms, syspop = np.unique(nodememory[nodememory >= meme_fitness_thresh], return_counts=True)
		idx = syspop.argsort()[::-1][:track_competing_memes] # reverse sorted indices
		system_popularity = {mms[i]:syspop[i] for i in idx}
		# competing_memes.append(system_popularity) # this will be HUGE eventually.
		if counter == track_memes_after:
			common_competing_memes = {k:[v] for k,v in system_popularity.iteritems()}
			print "Common competing memes at saturation:", len(common_competing_memes)
		else:
			common_competing_memes = updateSysPopForCommonCompetingMemes(common_competing_memes, system_popularity)
# 			if meme_count_after_saturation + 1 == max_memes_track:
# 				print "** Max memes achieved. Early termination at {0}th step.".format(counter)
# 				break
		
# simulation complete at this point
endtime = time.time()
print "Done! Total time in seconds: ", (endtime - starttime)	

# ==============================================================
# save system info per timestep: #uniquememes, avg. user entropy & avg. system entropy
# ==============================================================
print "\nSaving systems statistics.."
# sysdata = pd.DataFrame(np.asarray([steps,rug, unique_memecount, avg_user_entropy, system_entropy]).transpose(), columns=['Idx', 'Rug', 'UniqueMemes', 'AvgUserEntropy', 'SystemEntropy'])
sysdata = pd.DataFrame(np.asarray([steps, rug_M, rug, unique_memecount, system_entropy]).transpose(), columns=['Idx', 'Fit_Media', 'Rug', 'UniqueMemes', 'SystemEntropy'])
sysdata.to_csv(prefix + "system.csv", index=False)
print "System stats saved!"

# ==============================================================
# meme statistics
# ==============================================================
print "\nLen of memes set before final update: ", len(memes)
for m in memes.keys():
	if memes[m]['start'] < track_memes_after:
		# print m, "-->", memes[m]
		del memes[m]
print len(memes), "out of", max_memes_track, "requested memes were born after step", track_memes_after
print "Size of all memes (in MBs):", sys.getsizeof(memes)/float(1024**2)
print "Saving meme statistics.."
fitness = []
starttime = []
endtime = []
simultaneous_popularity = []
max_popularity = []
for meme, val in memes.iteritems():
	fitness.append(meme)
	starttime.append(val['start'])
	endtime.append(val['end'])
	simultaneous_popularity.append(val['simulatenous_popularity'])
	max_popularity.append(val['max_popularity'])	
memedata = pd.DataFrame(np.asarray([fitness, starttime, endtime, simultaneous_popularity, max_popularity]).transpose(), columns=['Fitness', 'Starttime', 'Endtime', 'SimultaneousPopularity', 'MaxPopularity'])
memedata.to_csv(prefix + "meme.csv", index=False)
print "Max meme size:", max_meme_size
print "Meme stats saved!"


# ==============================================================
# save COMMON competing memes information
# ==============================================================


input_memes=gzip.open(prefix+"input_memes_data.csv.gz","w")

for item in range(0,n):
 	xx=str(list(nodememory[item, :memory_size]))
 	xx=xx.replace("[","").replace("]","")
 	xx=str(xx).replace(", "," ")
 	xx=str(xx)
 	input_memes.write(str(item)+" "+str(xx)+" \n")


input_network=gzip.open(prefix+"input_network.csv.gz","w")

for node in range(0,n):

 	yy=str(sorted(list(G.neighbors(node))))
 	yy=yy.replace("[","").replace("]","")
 	yy=str(yy).replace(", "," ")
 	yy=str(yy)
 	input_network.write(str(node)+" "+str(yy)+" \n")




print "\n\nDone!!\n"



