#%%
import numpy as np
import joblib
import matplotlib.pyplot as plt
#%%

inFile = 'tmp_potatos_output_3.pkl'
data = joblib.load(inFile)

#%%

for subj in data.keys():
	print(subj)
	opt_list = data[subj]['info']
	infos = data[subj]['settings']

	fig, axs = plt.subplots(2, 6, figsize=(16, 8), constrained_layout=True)
	
	for n, (opt, info) in enumerate(zip(opt_list, infos)):
		bad_idx = opt['bad_idx']
		ax = axs[0,n]
		z_scores = [opt['z_scores'][bad_i] for bad_i in bad_idx]
		# print(z_scores.shape)
		ax.hist(z_scores, bins=20, range = (0,8), edgecolor='black', density=True)
		ax.set_title(info.values())
		ax.set_xlabel("z_score")
		ax.set_ylabel("Frequency")
		ax.set_xlim([0, 8])	
		# ax.grid(True)

		ax = axs[1,n]
		p_scores = [opt['clean_probability'][bad_i] for bad_i in bad_idx]
		# print(p_scores.shape)
		ax.hist(p_scores, bins=100, range = (0,0.01), edgecolor='black', density=True)
		ax.set_title(info.values())
		ax.set_xlabel("p_score")
		ax.set_ylabel("Frequency")
		# ax.set_xlim([0, 0.01])	
		ax.grid(True)

	# for ax in axs.flat[len(opt_list):]:
	# 	ax.set_visible(False)
		
	plt.show()





	#%%

inFile = '/home/reabt/experiments/ncc/MRI/code/tmp_potatos_output.pkl'
data2 = joblib.load(inFile)

for subj in data2.keys():
	print(subj)
	opt_list = data2[subj]['info']
	infos = data2[subj]['settings']

	fig, axs = plt.subplots(2, 8, figsize=(16, 8), constrained_layout=True)
	
	for n, (opt, info) in enumerate(zip(opt_list, infos)):
		bad_idx = opt['bad_idx']
		ax = axs[0,n]
		z_scores = [opt['z_scores'][bad_i] for bad_i in bad_idx]
		# print(z_scores.shape)
		ax.hist(z_scores, bins=50, edgecolor='black', density=True)
		ax.set_title(info.values())
		ax.set_xlabel("z_score")
		ax.set_ylabel("Frequency")
		ax.set_xlim([-3, 8])	
		# ax.grid(True)

		ax = axs[1,n]
		p_scores = [opt['clean_probability'][bad_i] for bad_i in bad_idx]
		# print(p_scores.shape)
		n_p = len(p_scores)
		p_weights = [100.0 / n_p] * n_p if n_p else []
		ax.hist(p_scores, bins=100, range=(0,0.01), edgecolor='black', weights=p_weights)
		ax.set_title(info.values())
		ax.set_xlabel("p_score")
		ax.set_ylabel("Percentage")
		# ax.set_xlim([0, 0.01])	
		ax.grid(True)

	# for ax in axs.flat[len(opt_list):]:
	# 	ax.set_visible(False)
		
	plt.show()