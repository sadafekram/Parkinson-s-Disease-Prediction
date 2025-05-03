import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from scipy.spatial import distance_matrix
import random
from subs.utils import *
from subs.models import *


data = pd.read_csv("parkinsons_updrs.csv")
describe_DataFrame(data)

data_subj_ori = pd.DataFrame()
subjects = pd.unique(data['subject#'])

for subj in subjects:
    temp_subj = data[data['subject#']==subj]
    temp_subj_copy = temp_subj.copy()
    temp_subj_copy.test_time = temp_subj_copy.test_time.astype(int)
    temp_subj_copy['GP']=temp_subj_copy['test_time']
    temp_subj_grouped = temp_subj_copy.groupby('GP').mean()
    data_subj_ori = pd.concat([data_subj_ori,temp_subj_grouped], axis=0, ignore_index=True)
    
print("The Dataset Shape After the Mean is ", data_subj_ori.shape)

num_ins, num_att = data_subj_ori.shape 
data_norm = (data_subj_ori - data_subj_ori.mean()) / data_subj_ori.std()
data_cov = data_norm.cov()

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(7, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(abs(data_cov), cmap=cmap, vmin=0, vmax=1, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 1})
fig.savefig('./UPDRS_CORR_COEFF.png', dpi=400)

seed_eval = input("Do you want to run for 301769 seed? TYPE YES or NO! In negative case, it will consider 20 random seeds!")

if seed_eval=="YES":

	seed = 301769
	train_test_ratio = 0.5
	X_train, X_test, y_train, y_test, mean_X, std_X, mean_Y, std_Y, regressors = data_preprocess(data_subj_ori, train_test_ratio, random_seed=seed)

	cont = input("Continue To LLS?")

	LLS = LeastLinearSquares(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
	LLS.run()
	LLS.plot_weights(regressors)
	LLS.plot_histogram()
	LLS.plot_regplot()
	LLS.stats()

	cont = input("Continue To GSD?")

	GSD = LinearRegression(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
	GSD.compiler(optimizer="SteepestDescent", mode="Global", epochs=1000)
	GSD.run()
	GSD.plot_weights(regressors)
	GSD.plot_histogram()
	GSD.plot_regplot()
	GSD.stats()

	cont = input("Continue To LSD?")

	intention = int(input("Do you want to enter your desired batch_size or find the optimum value automatically? \nType your desired number or enter 0 to proceed.\n"))
			
	if intention>4:
		batch_size = intention
		print(f"Batch Size Has Been Set to {batch_size} By Your Decision!")
	elif intention==0:
		confirm = input("WARNING!!! THIS PROCESS CAN TAKE ABOUT 30-60 MINS DEPEND ON YOUR HARDWARE!!!\nAre you sure? YES or NO?\n")
		if confirm=='YES':
			filter_n = input("Find best N, based on 'MSE' or 'R^2'?")
			all_n_data = pd.DataFrame()
			for i in range(X_test.shape[0]):
				print("N:", i)
				LSD = LinearRegression(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
				LSD.compiler(optimizer="SteepestDescent", mode="Local", epochs=200, batch_size=i)
				LSD.run()
				temp = LSD.stats()
				all_n_data = pd.concat([all_n_data, temp])

			all_n_data = all_n_data.reset_index().drop(columns=['index'])

			if filter_n=='MSE':
				filter_n_arg = all_n_data[filter_n].idxmin()
			else:
				filter_n_arg = all_n_data[filter_n].idxmax()

			plt.figure(figsize=(12,5))
			plt.subplot(121)
			plt.plot(all_n_data.index[0:100], all_n_data['MSE'].iloc[0:100], color='tab:blue', linestyle='-', linewidth=2)
			plt.vlines(x=all_n_data['MSE'].idxmin(), ymin=0, ymax=12, colors='green', ls=':', lw=2, label=f"Train Batch Size = {all_n_data['MSE'].idxmin()}")
			plt.xlabel(r'Train Batch Size', fontsize=10)
			plt.ylabel(r'$Mean-Squared-Error$', fontsize=10)
			plt.legend()
			plt.grid()
			plt.subplot(122)
			plt.plot(all_n_data.index[0:100], all_n_data['R^2'].iloc[0:100], color='tab:orange', linestyle='-', linewidth=2)
			plt.vlines(x=all_n_data['R^2'].idxmax(), ymin=0.9, ymax=1, colors='green', ls=':', lw=2, label=f"Train Batch Size = {all_n_data['R^2'].idxmax()}")
			plt.xlabel(r'Train Batch Size', fontsize=10)
			plt.ylabel(r'$R^2 Score$', fontsize=10)
			plt.legend()
			plt.grid()
			plt.savefig('./find_n.png', dpi=400)
			plt.show()
			batch_size = filter_n_arg
		else:
			batch_size = 10
			print("Batch Size Has Been Set to 10 Automatically!")
	else:
		raise ValueError("Please Enter a Valid Input > 4")

	LSD = LinearRegression(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
	LSD.compiler(optimizer="SteepestDescent", mode="Local", epochs=1000, batch_size=batch_size)
	LSD.run()
	LSD.plot_histogram()
	LSD.plot_regplot()
	LSD.stats()

else:

	all_LLS_tr_stats = pd.DataFrame()
	all_LLS_te_stats = pd.DataFrame()
	all_GSD_tr_stats = pd.DataFrame()
	all_GSD_te_stats = pd.DataFrame()
	all_LSD_te_stats = pd.DataFrame()

	random_int_seeds = random.sample(range(1, 100), 20)
	for seed in random_int_seeds:
		train_test_ratio = 0.5
		X_train, X_test, y_train, y_test, mean_X, std_X, mean_Y, std_Y, regressors = data_preprocess(data_subj_ori, train_test_ratio, random_seed=seed)
		LLS = LeastLinearSquares(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
		GSD = LinearRegression(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
		GSD.compiler(optimizer="SteepestDescent", mode="Global", epochs=1000)
		LSD = LinearRegression(X_train, y_train, X_test, y_test, mean_X, std_X, mean_Y, std_Y, random_seed=seed)
		LSD.compiler(optimizer="SteepestDescent", mode="Local", epochs=1000, batch_size=10)
		LLS.run()
		GSD.run()
		LSD.run()
		temp_LLS = LLS.stats()
		temp_GSD = GSD.stats()
		temp_LSD = LSD.stats()
		all_LLS_tr_stats = pd.concat([all_LLS_tr_stats, temp_LLS.loc['Training'].to_frame().T]).reset_index().drop(columns=['index'])
		all_LLS_te_stats = pd.concat([all_LLS_te_stats, temp_LLS.loc['test'].to_frame().T]).reset_index().drop(columns=['index'])
		all_GSD_tr_stats = pd.concat([all_GSD_tr_stats, temp_GSD.loc['Training'].to_frame().T]).reset_index().drop(columns=['index'])
		all_GSD_te_stats = pd.concat([all_GSD_te_stats, temp_GSD.loc['test'].to_frame().T]).reset_index().drop(columns=['index'])
		all_LSD_te_stats = pd.concat([all_LSD_te_stats, temp_LSD.loc['Test'].to_frame().T]).reset_index().drop(columns=['index'])
		
	print("20 Random Seeds:\n", random_int_seeds)

	print("LLS Training Mean for 20 Seeds:\n", all_LLS_tr_stats.mean())
	print("LLS Test Mean for 20 Seeds:", all_LLS_te_stats.mean())
	print("GSD Training Mean for 20 Seeds:", all_GSD_tr_stats.mean())
	print("GSD Test Mean for 20 Seeds:", all_GSD_te_stats.mean())
	print("LSD Test Mean for 20 Seeds:", all_LSD_te_stats.mean())