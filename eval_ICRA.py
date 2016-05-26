
%matplotlib inline
import sys
import time
import os
from os.path import expanduser
from collections import OrderedDict

import numpy as np
import matplotlib.pylab as plt

# Directories and filename
base_dir = os.path.expanduser("~/data/")
save_dir = os.path.expanduser("~/data/Results/")
LCTM_dir = os.path.expanduser("~/libs/LCTM/")
sys.path.append(LCTM_dir)

from LCTM import utils
from LCTM import models
from LCTM import datasets
from LCTM import metrics
from LCTM.utils import imshow_

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
features = "PVG" if dataset=="JIGSAWS" else "accel/"+features

if dataset == "JIGSAWS": data = datasets.JIGSAWS(base_dir)
elif dataset == "50Salads": data = datasets.Salads(base_dir)
else: print("Dataset not correctly specified")
experiment_name = features + "_" + str(int(time.time()))

# Model parameters
n_latent = 3
sample_rate = 5
use_prior = True

if dataset == "JIGSAWS": primitive_len = 100#100
elif dataset == "50Salads": primitive_len = 200
else: print("Dataset not correctly specified")
conv_len = primitive_len // sample_rate

# ----- Keep stats for each model -----
avgs = OrderedDict()
# metrics_ = ["frame", "filt", "seg", "clf"]
# metrics_ += ["edit_fr", "edit_fi", "edit_s"]
metrics_ = ["acc_fr", "acc_fi", "acc_seg"]
metrics_ += ["edit_fr", "edit_fi", "edit_seg"]
metrics_ += ["overlap_fr", "overlap_fi", "overlap_seg"]
# metrics_ += ["edit", "mp_precision", "mp_recall"]
for k in metrics_:
	avgs[k] = []
# ----------------------------------------

# Run for all splits in the evaluation setup
for idx_task in range(1, data.n_splits+1):	
# idx_task=1
# if 1:

	# Load Data
	X_train, y_train, X_test, y_test = data.load_split(features, idx_task, sample_rate=sample_rate)

	y_train = [y[0] for y in y_train]
	y_test = [y[0] for y in y_test]
	n_train = len(X_train)
	n_test = len(X_test)
	y_all = utils.remap_labels(np.hstack([y_train, y_test]))
	y_train, y_test = y_all[:n_train], y_all[-n_test:]

	# Compute STD over training data (to retain physical positions/velocities)
	# Put remove per-trial means to deal with different coordinate systems
	X_std = np.hstack(X_train).std(1)[:,None]
	X_train = [(x-x.mean(1)[:,None])/X_std for x in X_train]
	X_test = [(x-x.mean(1)[:,None])/X_std for x in X_test]


	# ------------Model & Inference---------------------------
	# Define and train model
	model = models.LatentConvModel(n_latent=n_latent, conv_len=conv_len, skip=conv_len, prior=use_prior, debug=True)
	model.fit(X_train, y_train, n_iter=200, learning_rate=.1, pretrain=True)

	# Evaluate using framewise inference
	model.inference_type = "framewise"
	P_test = model.predict(X_test)
	if save: utils.save_predictions(save_dir, P_test, y_test, idx_task, experiment_name="frame")

	# Evaluate using filtered inference
	model.inference_type = "filtered"
	model.filter_len = max(conv_len//2, 1)
	P_test_filt = model.predict(X_test)
	if save: utils.save_predictions(save_dir, P_test_filt, y_test, idx_task, experiment_name="filt")

	# Evaluate using segmental inference
	model.inference_type = "segmental"
	# model.max_segs = utils.max_seg_count(y_train)
	model.max_segs = 25
	P_test_seg = model.predict(X_test)
	if save: utils.save_predictions(save_dir, P_test_seg, y_test, idx_task, experiment_name="seg")

	# ------------Other metrics---------------------------
	P_other = P_test_seg
	# Evaluate with known segmentation
	# avgs["clf"] += [metrics.classification_accuracy(P_other, y_test)]

	avgs["acc_fr"] += [metrics.accuracy(P_test, y_test)]
	avgs["acc_fi"] += [metrics.accuracy(P_test_filt, y_test)]
	avgs["acc_seg"] += [metrics.accuracy(P_test_seg, y_test)]

	# Edit score
	avgs["edit_fr"] += [metrics.edit_score(P_test, y_test)]
	avgs["edit_fi"] += [metrics.edit_score(P_test_filt, y_test)]
	avgs["edit_seg"] += [metrics.edit_score(P_test_seg, y_test)]

	# Edit score
	avgs["overlap_fr"] += [metrics.overlap_score(P_test, y_test)]
	avgs["overlap_fi"] += [metrics.overlap_score(P_test_filt, y_test)]
	avgs["overlap_seg"] += [metrics.overlap_score(P_test_seg, y_test)]

	txt = "#{}: ".format(idx_task)
	txt += ", ".join(["{}:{:.3}%".format(k, v[-1]) for k,v in avgs.items()])
	txt += "\n"
	print(txt)

# -------- Compute statistics -----------
print("-----------------")
print("%: " + "\t ".join(avgs.keys()))
for i in range(len(list(avgs.values())[0])):
	txt = "{}: ".format(i+1)
	for k in avgs:
		txt += "{:.3}%\t".format(avgs[k][i])
	print(txt)
print("-----------------")
means = [np.mean(avgs[k]) for k in avgs]
print("Avg: " + "".join(["{:.4}% \t".format(m) for m in means]))
print("-----------------")


