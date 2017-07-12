"""
create a file containing only (datasetname, score, timestamp)
from scores.txt and ***timestamp.txt
"""

import os
import datetime, time
import sys
import traceback
import matplotlib.pyplot as plt
from math import log


def get_time_budget(datasetname, time_budget_file):
	with open(time_budget_file) as f:
		content = f.readlines()

	time_budget = {}
	for c in content:
		time_budget[c.rstrip('\n').split(': ')[0]]=int(c.rstrip('\n').split(': ')[-1])

	return time_budget[datasetname]



def make_timestamp_file(result_folder, timestamp_filename):
	success = True
	try:
		os.system("stat -c '%n %y %s' * > ../%s"%timestamp_filename)
	except Exception:
		print(traceback.format_exc())
		success = False
	return success

def make_name_score_time(score_file, timestamp_file):
	name_time = dict()
	name_score = dict()
	for line in open(timestamp_file): 
	# line looks like: tania_test_00016.predict 2017-03-08 20:17:33.991599000 +0100
		datasetname, date, thetime, _, _ = line.split()
		datasetname = datasetname.split('.')[0].upper()
		y, m, d = date.split('-')
		y, m, d = int(y), int(m), int(d)
		h, mn, sf = thetime.split(':')
		h, mn = int(h),int(mn)
		s, ms = sf.split('.')
		s = int(s)
		ms = int(float('0.'+ms)*60)
		datetimeobject = datetime.datetime(y,m,d,h,mn,s,ms)
		timestamp_insecond = time.mktime(datetimeobject.timetuple())
		name_time[datasetname] = timestamp_insecond
		# print 'original line: ', line
		# print 'datasetname: ', datasetname
		# print 'y,m,d,h,mn,s,ms: ', y,m,d,h,mn,s,ms
		# print 'timestamp_insecond: ', timestamp_insecond
		# print 
		# print 
		# print 
	i = 0
	for line in open(score_file):
		# print i
		i+=1
		try: 	
			_, datasetname2, _, score = line.split()
			datasetname2 = datasetname2.split(':')[0].split('(')[1].split(')')[0]
			name_score[datasetname2.upper()] = float(score)
			# print 'ok'
		except ValueError:
			if 'Duration' in line.split()[0]:
				pass
		except Exception:
			print(traceback.format_exc())
			pass
	return name_time, name_score



def plot_learning_curve(datasetname, timestamp_file, score_file, color, winner, timebudget_control=False, log_time=False):

	name_time, name_score = make_name_score_time(score_file, timestamp_file)
	name_sequence = sorted([name for name in name_time if name.startswith(datasetname.upper()+'_TEST')], \
	 	key = lambda x: int(x.split('_')[-1]))
	# print ('name_sequence', name_sequence)
	pre_time_sequence = [name_time[n] for n in name_sequence]
	time_sequence = [pre_time_sequence[i]-pre_time_sequence[0] \
		for i in range(len(pre_time_sequence))] # make time_sequance starting from 0
	print ('time_sequence', time_sequence)
	score_sequence = [name_score[n] for n in name_sequence]

	if timebudget_control:
		timebudget_for_this_dataset = get_time_budget(datasetname, 'time_budget_all_datasets.txt')
		time_sequence_within_timebudget = [t for t in time_sequence if t<timebudget_for_this_dataset]

		print ('time_sequence_within_timebudget', time_sequence_within_timebudget)
		print ('last prediction before within timebudget', name_sequence[len(time_sequence_within_timebudget)-1])

	if log_time:
		log_time = []
		
		for t in time_sequence:
			if t == 0.0:
				t += 0.00001
			log_time.append(log(t))
		if timebudget_control:
			log_time_within_timebudget = []
			for t in time_sequence_within_timebudget:
				if t == 0.0:
					t += 0.00001
				log_time_within_timebudget.append(log(t))
			# print log_time_within_timebudget
		# print log_time
		

		plt.plot(log_time, score_sequence, '.-', markersize=8, color=color) #label=datasetname+' by '+winner,
		if timebudget_control:
			plt.plot(log_time_within_timebudget, score_sequence[:len(time_sequence_within_timebudget)], '.-', markersize=8, color='yellow') #label=datasetname+' by '+winner+' within timebudget', 

	else:
		plt.plot(time_sequence, score_sequence, '.-', markersize=8, color=color) #label=datasetname+' by '+winner,
		if timebudget_control:
			plt.plot(time_sequence_within_timebudget, score_sequence[:len(time_sequence_within_timebudget)], '.-', markersize=8, color='yellow') #label=datasetname+' by '+winner+' within timebudget', 

	# plt.plot(log_time, score_sequence, '*-', label=datasetname)




if __name__ == '__main__':

	input_dir_abhishek = '/Users/lishengsun/Documents/AutoML_2016/Final_report/Upload_to_git/learning_curves/temp_scores/abhishek'
	timestamp_file_abhishek = os.path.join(input_dir_abhishek, 'results_abhishek_timestamp.txt')
	score_file_abhishek = os.path.join(input_dir_abhishek, 'scores.txt')

	input_dir_autosklearn = '/Users/lishengsun/Documents/AutoML_2016/Final_report/Upload_to_git/learning_curves/temp_scores/aad_freiburg'
	timestamp_file_autosklearn = os.path.join(input_dir_autosklearn, 'results_aad_freiburg_timestamp.txt')
	score_file_autosklearn = os.path.join(input_dir_autosklearn, 'scores.txt')
	name_time, name_score = make_name_score_time(score_file_autosklearn, timestamp_file_autosklearn)

	input_dir_autosklearn_sameloss = '/Users/lishengsun/Documents/AutoML_2016/Final_report/Upload_to_git/learning_curves/temp_scores/aad_freiburg'
	timestamp_file_autosklearn = os.path.join(input_dir_autosklearn, 'results_aad_freiburg_timestamp.txt')
	score_file_autosklearn_sameloss = os.path.join(input_dir_autosklearn, 'same_loss/vanilla_autosklearn_sameLoss_17-06-29-10-56/scores.txt')

	datasets0 = ['adult', 'cadata', 'digits', 'dorothea', 'newsgroups']
	datasets1 = ['christine', 'jasmine', 'philippine', 'sylvine', 'madeline']
	datasets2 = ['albert', 'dilbert', 'fabert', 'robert', 'volkert']
	datasets3 = ['alexis', 'dionis', 'grigoris', 'jannis', 'wallis']
	datasets4 = ['evita', 'flora', 'helena', 'tania', 'yolanda']
	datasets5 = ['arturo', 'carlo', 'marco', 'pablo', 'waldo']
	datasets = ['adult', 'cadata', 'digits', 'dorothea', 'newsgroups', \
		'christine', 'jasmine', 'madeline', 'philippine', 'sylvine',  \
		'albert', 'dilbert', 'fabert', 'robert', 'volkert', \
		'alexis', 'dionis', 'grigoris', 'jannis', 'wallis', \
		'evita', 'flora', 'helena', 'tania', 'yolanda', \
		'arturo', 'carlo', 'marco', 'pablo', 'waldo']
	large_y_datasets = ['tania', 'carlo']
	plt.figure(1)
	for datasetname in datasets:
		try:
			plot_learning_curve(datasetname, timestamp_file_abhishek, score_file_abhishek, 'blue', 'abhishek', log_time=True)
			plot_learning_curve(datasetname, timestamp_file_autosklearn, score_file_autosklearn, 'green', 'aad_freiburg', timebudget_control=True, log_time=True)
			# plot_learning_curve(datasetname, timestamp_file_autosklearn, score_file_autosklearn_sameloss, 'green', 'aad_freiburg_sameLoss', timebudget_control=True)

		except Exception:
			
			print(traceback.format_exc())
			pass

		# plt.xlabel('log(time in seconds)')
		plt.ylim([-0.1,1])
		plt.xlabel('log (time in seconds)', fontsize=18)
		plt.ylabel('scores', fontsize=18)
		plt.title('Learning curve of %s\n (Performance as a function of time)'%datasetname, fontsize=18)
		# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
		# plt.axis([40, 160, 0, 0.03])
		# plt.grid(True)
		# plt.legend(loc='best')
		plt.savefig('LearningCurvePng/'+datasetname+'_learningCurve')
		plt.show()
		




