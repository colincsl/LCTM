%matplotlib inline

import sys, os
from os.path import expanduser
import numpy as np
from scipy import io as sio
import sklearn.metrics as sm
import matplotlib.pylab as plt
import theano
from keras.utils import np_utils

# From CVPR16 folder
# from models import *

os.chdir(expanduser("~/libs/LCTM/"))
from LCTM import metrics, models, datasets
from LCTM.dtw import DTW
from LCTM.infer import segmental_inference
from LCTM.utils import subsample, match_lengths, mask_data, unmask, save_predictions, imshow_

# Directories and filename
# base_dir = expanduser("~/data/")
# save_dir = expanduser("~/data/Results/")
base_dir = expanduser("~/Data/")
save_dir = expanduser("~/Data/Results/")


# ------------------------------------------------------------------
# If running from command line get split index number
# Otherwise choose one of the splits manually
try:
    dataset = ["50Salads", "JIGSAWS", "EndoVis", "GTEA", "MPII"][int(sys.argv[1])]
    idx_task = int(sys.argv[2])
    try: eval_idx = int(sys.argv[3])
    except: eval_idx = -1
    save = [False, True][0]
except:
    dataset = ["50Salads", "JIGSAWS", "EndoVis", "GTEA", "MPII"][2]
    eval_idx = -1
    idx_task = 1
    # idx_task = 'youtube'
    save = [False, True][0]
print("tCNN: ({}, {}, {})".format(dataset, eval_idx, idx_task))

video = [False, True][1]
if video:
    feature_set = "_"+["low", "mid", "high", "eval"][eval_idx] if dataset=="50Salads" else ""
    # features = "cnn_rgb_"+"Thurs12_sigmoid"+feature_set
    # features = "cnn_rgb_"+"Monday_sigmoid"+feature_set
    # features_flow = "cnn_flow_"+"Thurs12_sigmoid"+feature_set
    # features = "cnn_rgb_"+"Tues_sig_3x3"+feature_set
    # features = "cnn_rgb_"+"CVPR_"+["actions", "tools", "attributes"][0]+feature_set
    features = "cnn_rgb_"+"CVPR_"+["actions", "tools", "attributes"][1]+"_no_motion"+feature_set
    # features = "cnn_rgb_"+"Tues_sig_4x4"+feature_set # EndoVis
    # features = "cnn_rgb_"+"MICCAI_attributes_Feb4"+feature_set
    # features = "vgg_16" 
else:
    feature_set = ["low", "mid", "high", "eval"][eval_idx] if dataset=="50Salads" else ""
    if dataset=="JIGSAWS":
        features = ["WACV", "DenseTraj_bow"][0]
    elif dataset == "50Salads":
        features = ["accel/"+feature_set, "DenseTraj_bow"][0]
    elif dataset == "EndoVis":
        features = ["attributes", "tools"][0]

# ---Dataset params---
save_name = ""+feature_set
n_nodes = 64
nb_epoch = 15
sample_rate = 1 if video else 30
# sample_rate = 1
model_type = ['cvpr', 'icra', 'dtw'][2]

if dataset == "JIGSAWS": conv = 200
elif dataset == "50Salads": conv = 200
elif dataset == "EndoVis": conv = 200
else: print("Dataset not specified")
conv //= sample_rate

# --------------- Load data ------------------------
if dataset == "JIGSAWS": data = datasets.JIGSAWS(base_dir)
elif dataset == "50Salads": data = datasets.Salads(base_dir)
elif dataset == "EndoVis": data = datasets.EndoVis(base_dir)
else: print("Dataset not specified")

avgs, avgs_seg = {}, {}

idx_task = 1
# if 1:
for idx_task in range(1, data.n_splits+1):
    # Load Data
    X_train, y_train, X_test, y_test = data.load_split(features, idx_task, sample_rate)

    if video:
        _, y_train = subsample(X_train, y_train, 30)
        _, y_test = subsample(X_test, y_test, 30)
        X_train, y_train = match_lengths(X_train, y_train)
        X_test, y_test = match_lengths(X_test, y_test)

    n_feat = data.n_features
    n_classes = data.n_classes

    # Preprocess VGG
    if 1 or features == "vgg_16":
        from sklearn.svm import LinearSVC
        svm = LinearSVC()
        svm.fit(np.hstack(X_train).T, np.hstack(y_train))
        # svm.score(np.hstack(X_train).T, np.hstack(y_train))
        print("Test acc: {:.04}%".format(svm.score(np.hstack(X_test).T, np.hstack(y_test))*100))

        P_test = [svm.predict(x.T).T for x in X_test]
        X_train = [svm.decision_function(x.T).T for x in X_train]
        X_test = [svm.decision_function(x.T).T for x in X_test]

    # --------- CVPR model ----------
    if model_type == 'cvpr':
        # Go from y_t = {1...C} to one-hot vector (e.g. y_t = [0, 0, 1, 0])
        Y_train = [np_utils.to_categorical(x, n_classes).T for x in y_train]
        Y_test = [np_utils.to_categorical(x, n_classes).T for x in y_test]

        # Mask data such that it's all the same length
        X_train, Y_train, M_train = mask_data(X_train, Y_train)
        X_test, Y_test, M_test = mask_data(X_test, Y_test)
        n_timesteps = Y_train[0].shape[0]

        # Learning params
        loss_type = ["mse", "categorical_crossentropy", "hinge", "msle"][0]
        max_len = M_train.shape[1]
        model = build_cvpr(n_nodes, conv, n_classes, n_feat, max_len, 
                            loss_type=loss_type, optimizer="adaprop")
        # model = build_icra(n_nodes, conv, n_classes, n_feat, max_len, loss_type=loss_type)
        # model = build_lstm(n_nodes, conv, n_classes, n_feat, max_len, loss_type=loss_type)
        # model = build_conv_lstm(n_nodes, conv, n_classes, n_feat, max_len, loss_type=loss_type)

        # For Keras, use [n_samples x n_timesteps x n_features]
        X_train = np.array([x.T for x in X_train])
        Y_train = np.array([x.T for x in Y_train])
        X_test = np.array([x.T for x in X_test])
        Y_test = np.array([x.T for x in Y_test])

        model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=4, 
                  verbose=1, sample_weight=M_train[:,:,0])

        # Predict classes
        P_train_raw = model.predict(X_train, verbose=0)
        P_test_raw = model.predict(X_test, verbose=0)
        P_test = model.predict_classes(X_test, verbose=0)

        # Unmask data
        P_train_raw = unmask(P_train_raw, M_train)
        P_test_raw = unmask(P_test_raw, M_test)
        P_test = unmask(P_test, M_test)
        Y_test = unmask(Y_test, M_test)

        # Add structured model
        model2 = models.PretrainedModel(skip=conv, debug=True)
        P_train_raw_ = [p.T-.5 for p in P_train_raw]
        P_test_raw_ = [p.T-.5 for p in P_test_raw]
        model2.fit(P_train_raw_, y_train, n_iter=300, learning_rate=.01, pretrain=True)

        # Evaluate using framewise inference
        S_test = model2.predict(P_test_raw_, inference="segmental")

        # Perform segmental inference
        # max_segs = utils.max_seg_count(y_train)
        #S_test = [segmental_inference(P_test_raw[i], max_segs) for i in range(len(y_test))]
    
    # --------- ICRA model ----------
    elif model_type == 'icra':
        # Add structured model
        # skip = conv
        # conv = 100
        skip = 0
        conv = 1
        model = models.LatentConvModel(n_latent=3, conv_len=conv, skip=skip, prior=True, debug=True)
        # model = models.SegmentalModel(pretrained=False)
        model.fit(X_train, y_train, n_iter=300, learning_rate=.1, pretrain=True)
        # model = models.PretrainedModel(skip=skip, debug=True)
        # from LCTM import utils
        # from LCTM.energies import pairwise
        # y_tmp = [utils.segment_labels(y) for y in y_train]
        # pw = np.sum([pairwise.pw_cost(y, n_classes) for y in y_tmp], 0)
        # model.ws['pw'][pw<=0] = -9999

        y_tmp = [utils.segment_labels(y) for y in y_test][0]

        # Evaluate using structured model
        P_test = model.predict(X_test, inference="framewise")
        S_test = model.predict(X_test, inference="segmental", known_order=y_tmp)
    
    # --------- DTW model ----------
    elif model_type =='dtw':
        S_test = []
        scores_tmp = {idx_task:[]}  
        for i in range(len(X_test)):
            dists, preds = [], []
            for j in range(len(X_train)):
                d, c = DTW(X_train[j], X_test[i], output_correspondences=True)
                d /= X_train[j].shape[1]
                dists += [d]
                preds += [y_train[j][c]]
                scores_tmp[idx_task] += [np.mean(y_test[0] == y_train[j][c])]
            idx_best = np.argmin(dists)
            S_test += [preds[idx_best]]
        P_test = S_test
    else:
        print("Error: model not defined")

    # Compute metrics
    metric_fcn = [sm.accuracy_score, sm.precision_score, sm.recall_score][0]
    print("Split {}".format(idx_task))
    # acc_sCNN = np.mean([metric_fcn(P_cnn_test[i], y_test[i])*100 for i in range(len(y_test))])
    acc_tCNN = np.mean([metric_fcn(P_test[i],y_test[i])*100 for i in range(len(y_test))])
    acc_seg = np.mean([metric_fcn(S_test[i],y_test[i])*100 for i in range(len(y_test))])
    # acc_clf = np.mean([metric_fcn(clf_pred[i],clf_truth[i])*100 for i in range(len(y_test))])
    acc_clf = metrics.classification_accuracy(P_test, y_test)
    print("FW: {:.04}".format(acc_tCNN))
    print("Seg: {:.04}".format(acc_seg))
    # print("Clf: {:.04}".format(acc_clf))
    print()     
    avgs[idx_task]= acc_tCNN
    avgs_seg[idx_task]= acc_seg

    # ----- Save predictions -----
    if save:
        experiment_name = save_name
        dir_out_base = expanduser(base_dir+"predictions/{}/".format(experiment_name))

        # save_predictions(dir_out_base+"sCNN", y_test, P_cnn_test, idx_task)
        save_predictions(dir_out_base+"stCNN", y_test, P_test, idx_task)
        save_predictions(dir_out_base+"seg", y_test, S_test, idx_task)
        # save_predictions(dir_out_base+"clf", clf_truth, clf_pred, idx_task)

    # ---- Viz predictions -----
    if 1:
        # # Output all truth/prediction pairs
        plt.figure(idx_task, figsize=(10,10))
        for i in range(len(y_test)):
            plt.subplot(len(y_test),1,i+1)
            # tmp = np.vstack([y_test[i][:,None].T, P_test[i][:,None].T])
            tmp = np.vstack([y_test[i][:,None].T, P_test[i][:,None].T, S_test[i][:,None].T])
            imshow_(tmp)
            plt.xticks([])

    # ---- Viz weights -----
    if 0:
        # Output weights at the first layer
        plt.figure(2, figsize=(10,10))
        ws = model.get_weights()[0]
        for i in range(min(36, len(ws[0]))):
            plt.subplot(6,6,i+1)
            # imshow_(model.get_weights()[0][i][:,:,0]+model.get_weights()[1][i])
            imshow_(ws[i][:,:,0])

    # ---- Viz confusion -----
    if 0:
        # Show confusion matrix
        if dataset == "50Salads":
            labels = ["bg"]+ np.loadtxt(base_dir+"{}/labels/{}_level_actions.txt".format(dataset, feature_set.strip("_")), str).tolist()
        elif dataset=="JIGSAWS":
            labels = np.loadtxt(base_dir+"{}/labels/suture_actions.txt".format(dataset), str, delimiter='\n').tolist()
        conf = sm.confusion_matrix(np.hstack(y_test), np.hstack(P_test), labels=np.arange(n_classes))*1.
        conf /= conf.sum(0)
        conf = np.nan_to_num(conf)
        plt.figure('conf', figsize=(12,12));
        imshow_(conf)
        fontsize = 20
        xticks(np.arange(len(labels)), labels, rotation=60,fontsize=fontsize)
        yticks(np.arange(len(labels)), labels, fontsize=fontsize)

for i in avgs:
    print("{}: {:.04}%, {:.04}%".format(i, avgs[i], avgs_seg[i]))
avg_mean = np.mean(list(avgs.values()))
avg_seg_mean = np.mean(list(avgs_seg.values()))
print("Avg: {:.04}%, {:.04}%".format(avg_mean, avg_seg_mean))

