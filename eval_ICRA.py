
%matplotlib inline
import sys
import time
import os
from os.path import expanduser
from collections import OrderedDict

import numpy as np
import matplotlib.pylab as plt

from LCTM import utils
from LCTM import models
from LCTM import datasets
from LCTM import metrics

# Directories and filename
base_dir = expanduser("~/Data/")
# base_dir = "/home/colin/data/"
save_dir = expanduser("~/Data/Results/")

# ------------------------------------------------------------------
# If running from command line get split index number
# Otherwise choose one of the splits manually
try:
	dataset = ["50Salads", "JIGSAWS"][int(sys.argv[1])]
	idx_task = int(sys.argv[2])
	try:
		eval_idx = int(sys.argv[3])
	except:
		eval_idx = -1
	save = [False, True][1]
except:
	dataset = ["50Salads", "JIGSAWS"][1]
	eval_idx = -1
	idx_task = 1
	save = [False, True][1]

print("Setup: {}, split:{}, features:{}".format(dataset, idx_task, eval_idx))

# Feature types
features = ["low", "mid", "high", "eval"][eval_idx] if dataset=="50Salads" else ""
features = "WACV" if dataset=="JIGSAWS" else "accel/"+features

if dataset == "JIGSAWS": data = datasets.JIGSAWS(base_dir, features)
elif dataset == "50Salads": data = datasets.Salads(base_dir, features)
else: print("Dataset not correctly specified")
experiment_name = features + "_" + str(int(time.time()))


# Model parameters
if dataset == "JIGSAWS": primitive_len = 100
elif dataset == "50Salads": primitive_len = 200
else: print("Dataset not correctly specified")

sample_rate = 5
skip = primitive_len // sample_rate
conv_len = primitive_len // sample_rate

# Keep stats for each model
avgs = OrderedDict()
metrics_ = ["frame", "filt", "seg", "clf"]
metrics_ += ["edit_fr", "edit_fi", "edit_s"]
# metrics_ += ["edit", "mp_precision", "mp_recall"]
for k in metrics_:
	avgs[k] = []

# Run for all splits in the evaluation setup
for idx_task in range(1, data.n_splits+1):	
	
	# Load Data
	X_train, y_train, X_test, y_test = data.load_split(idx_task)
	X_train, y_train = utils.subsample(X_train, y_train, sample_rate)
	X_test, y_test = utils.subsample(X_test, y_test, sample_rate)

	# ------------Model & Evaluation---------------------------
	# Define and train model
	# model = models.LatentChainModel(n_latent=1, skip=skip, debug=True)
	model = models.LatentConvModel(n_latent=1, conv_len=conv_len, skip=skip, debug=True)
	# model = models.SegmentalModel(debug=True)
	model.fit(X_train, y_train, n_iter=300, learning_rate=.1, pretrain=True)

	# Evaluate using framewise inference
	model.inference_type = "framewise"
	P_test = model.predict(X_test)
	avgs["frame"] += [metrics.accuracy(P_test, y_test)]
	if save: utils.save_predictions(save_dir, P_test, y_test, idx_task, experiment_name="frame")

	# Evaluate using filtered inference
	model.inference_type = "filtered"
	model.filter_len = skip//2
	P_test_filt = model.predict(X_test)
	avgs["filt"] += [metrics.accuracy(P_test_filt, y_test)]
	if save: utils.save_predictions(save_dir, P_test_filt, y_test, idx_task, experiment_name="filt")

	# Evaluate using segmental inference
	model.inference_type = "segmental"
	model.max_segs = utils.max_seg_count(y_train)
	P_test_seg = model.predict(X_test)
	avgs["seg"] += [metrics.accuracy(P_test_seg, y_test)]
	if save: utils.save_predictions(save_dir, P_test_seg, y_test, idx_task, experiment_name="seg")

	# ------------Other metrics---------------------------
	P_other = P_test_seg
	# Evaluate with known segmentation
	avgs["clf"] += [metrics.classification_accuracy(P_other, y_test)]

	# Edit score
	# avgs["edit"] += [metrics.edit_score(P_other, y_test)]
	avgs["edit_fr"] += [metrics.edit_score(P_test, y_test)]
	avgs["edit_fi"] += [metrics.edit_score(P_test_filt, y_test)]
	avgs["edit_s"] += [metrics.edit_score(P_test_seg, y_test)]

	# TODO: overlap score

	# Midpoint precision/recall
	# avgs["mp_precision"] += [metrics.midpoint_precision(P_other, y_test)]
	# avgs["mp_recall"] += [metrics.midpoint_recall(P_other, y_test)]	

	txt = "#{}: ".format(idx_task)
	txt += ", ".join(["{}:{:.3}%".format(k, v[-1]) for k,v in avgs.items()])
	txt += "\n"
	print(txt)

# -------- Compute statistics -----------
print("-----------------")
print("%: " + "\t ".join(avgs.keys()))
for i in range(len(avgs["frame"])):
	txt = "{}: ".format(i+1)
	for k in avgs:
		txt += "{:.3}%\t".format(avgs[k][i])
	print(txt)
print("-----------------")
means = [np.mean(avgs[k]) for k in avgs]
print("Avg: " + "".join(["{:.4}% \t".format(m) for m in means]))
print("-----------------")


