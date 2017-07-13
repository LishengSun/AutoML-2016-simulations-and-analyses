"""
author: Lisheng Sun

Error bars computed by boostrapping.

Change: 
	- input_dir
	- output_dir 
	- prediction_files: line94 to the directory where scores you want to compute errorbars upon are stores.

"We need some notion of errorbars.

Compute error bars and fill out (separate) tables with error bars.

For uniformity of method, compute error bars with boostrap using 100 repeat.
For each repeat compute the score by resampling without replacement a subset of the samples,
then average the results and compute the stdev, which will be the error bar.

By looking at the results, we'll decide how to represent error bars."

"""
from libscores import *
import numpy as np
import traceback
import datetime

def errorbars(predict_file, solution_file, metric, task):

	solution = read_array(solution_file) # array
	prediction = read_array(predict_file) # array

	tot_num_of_samples, _ = solution.shape
	if(solution.shape!=prediction.shape): 
		raise ValueError("Bad prediction shape {}".format(prediction.shape))

	scores = []
	for i in range(100):
		#sample 10% of examples, without replacement
		idx_i = np.random.choice(tot_num_of_samples, int(tot_num_of_samples*0.1), replace=False)
		solution_i = solution[idx_i, :]
		prediction_i = prediction[idx_i, :]
		# compute the score of this subset
		if metric =='r2_metric' or metric =='a_metric': 
			# Remove NaN and Inf for regression
			solution_i = sanitize_array (solution_i) 
			prediction_i = sanitize_array (prediction_i)  
			score_i = eval(metric + '(solution_i, prediction_i, "' + task + '")')

		else:
            # Compute version that is normalized (for classification scores). This does nothing if all values are already in [0, 1]
			[csolution_i, cprediction_i] = normalize_array (solution_i, prediction_i)
			score_i = eval(metric + '(csolution_i, cprediction_i, "' + task + '")')

		scores.append(score_i)
	stdev = np.std(scores)
	mean = np.mean(scores)
	return mean, stdev

		

if __name__ == "__main__":

	running_on_lri_server = False

	same_loss = False

	the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")


	if running_on_lri_server:
	    root_dir = '/users/ao/lsun/Simulations/'
	else:
	    root_dir = '/Users/lishengsun/Documents/AutoML_2016/Final_report/Upload_to_git/'
	
	input_dir = root_dir + "simulation_results/" 
	output_dir = root_dir + "scores_of_simulation_results/winner_abhishek/"#_%s/"%the_date 

	errorbar_file = open(os.path.join(output_dir, 'errorbars.txt'), 'wb')
	# Get all the solution files from the solution directory
	solution_names = sorted(ls(os.path.join(input_dir, 'ref', '*.solution')))



	for solution_file in solution_names:
	    	# Extract the dataset name from the file name
		basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.')-1]
    		# Load the info file and get the task and metric
		info_file = ls(os.path.join(input_dir, 'ref', basename[0:3] + '*_public.info'))[0]
		info = get_info (info_file)    
        	score_name = info['task'][0:-15] + info['metric'][0:-7].upper() 
                try:
       			 # Get the last prediction from the res subdirectory (must end with '.predict')
       			
			predict_files = [ls(os.path.join(input_dir, 'res/abhishek', basename + '*.predict'))[-1]]
			if (predict_files == []): 
				raise IOError('Missing prediction file {}'.format(basename))
			for predict_file in predict_files:
				predict_name = predict_file[-predict_file[::-1].index(filesep):-predict_file[::-1].index('.')-1]
				if same_loss:
					if info['task'] == 'regression':
						mean, stdev = errorbars(predict_file, solution_file, 'r2_metric', info['task'])
					else:
						mean, stdev = errorbars(predict_file, solution_file, 'bac_metric', info['task'])
				else:
					mean, stdev = errorbars(predict_file, solution_file, info['metric'], info['task'])

				print ("==============="+predict_name+"===================")
				print ("mean: %f"%mean)
				print ("error bar (std): %f"%stdev)

				errorbar_file.write('%s\t%f\n'%(predict_name, stdev))
				# score_file.write("set%d" % set_num + " (" + predict_name.capitalize() + "): score: %0.12f\n" % score)
		
		except Exception:
			print (traceback.format_exc())
			pass





