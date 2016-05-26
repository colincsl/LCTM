%matplotlib inline

import sys, os
from os.path import expanduser
import numpy as np
from scipy import io as sio
import sklearn.metrics as sm
import matplotlib.pylab as plt
# import theano
from keras.utils import np_utils
plt.rcParams['figure.figsize']=[10,10]


# From CVPR16 folder
from models import *

os.chdir(expanduser("~/libs/LCTM/"))
from LCTM import metrics, models, datasets
from LCTM.dtw import DTW
from LCTM.infer import segmental_inference
from LCTM.utils import subsample, match_lengths, mask_data, unmask, save_predictions, imshow_
from LCTM import utils

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
    dataset = ["50Salads", "JIGSAWS", "EndoVis", "GTEA", "MPII"][1]
    eval_idx = -1
    idx_task = 1
    # idx_task = 'youtube'
    save = [False, True][0]
print("tCNN: ({}, {}, {})".format(dataset, eval_idx, idx_task))

video = [False, True][1]
if video:
    feature_set = "_"+["low", "mid", "high", "eval"][eval_idx] if dataset=="50Salads" else ""
    # features = "cnn_rgb_"+"CVPR_"+["actions", "tools", "attributes"][2]+feature_set
    # features = "cnn_rgb_"+"CVPR_"+["actions", "tools", "attributes"][2]+"_no_motion"+feature_set
    # features = "cnn_rgb_"+"MICCAI_attributes_Feb4"+feature_set

    features = "cnn_rgb_"+"Tues_sig_3x3"+feature_set # JIG/50
    # features = "cnn_rgb_"+"Tues_sig_4x4"+feature_set # JIG/50
    # features = "vgg_16" 
else:
    feature_set = ["low", "mid", "high", "eval"][eval_idx] if dataset=="50Salads" else ""
    if dataset=="JIGSAWS":
        features = ["WACV", "DenseTraj_bow", "PVG", "PG"][2]
    elif dataset == "50Salads":
        features = ["accel/"+feature_set, "DenseTraj_bow"][0]
    elif dataset == "EndoVis":
        features = ["attributes", "tools"][0]

# ---Dataset params---
save_name = ""+feature_set
n_nodes = 64
nb_epoch = 15
sample_rate = 1 if video else 10#60*2
# sample_rate = 5
model_type = ['svm', 'cvpr', 'icra', 'dtw'][-1]
remove_bg = [False, True][1]
metric_fcn = [sm.accuracy_score, sm.precision_score, sm.recall_score][0]

if dataset == "JIGSAWS": conv = 200
elif dataset == "50Salads": conv = 200
elif dataset == "EndoVis": conv = 200
else: print("Dataset not specified")
conv //= sample_rate
conv = 10

# --------------- Load data ------------------------
if dataset == "JIGSAWS": data = datasets.JIGSAWS(base_dir)
elif dataset == "50Salads": data = datasets.Salads(base_dir)
elif dataset == "EndoVis": data = datasets.EndoVis(base_dir)
else: print("Dataset not specified")

from collections import OrderedDict
avgs, avgs_seg = OrderedDict(), OrderedDict()
edit_tCNN, edit_seg, edit_rob = OrderedDict(), OrderedDict(), OrderedDict()

idx_task = 1
# if 1:
for idx_task in range(1, data.n_splits+1):
    P_test = None

    # Load Data
    # X_train, y_train, X_test, y_test = data.load_split(features, idx_task=idx_task, 
                                                        # sample_rate=sample_rate)
    # feature_type = "A_0" if video else "X"  
    feature_type = "X"                       
    # sample_rate = 1                     
    X_train, y_train, X_test, y_test = data.load_split(features, idx_task=idx_task, 
                                                        sample_rate=sample_rate,
                                                        feature_type=feature_type)
    n_feat = data.n_features

    if video:
        feature_type = "Y"                                   
        y_train, _, y_test, _ = data.load_split(features, idx_task=idx_task, 
                                                            sample_rate=sample_rate,
                                                            feature_type=feature_type)

        y_train = [y[0].astype(np.int) for y in y_train]
        y_test = [y[0].astype(np.int) for y in y_test]

        if dataset is "JIGSAWS":
            y_train = [y-2 for y in y_train]
            y_test = [y-2 for y in y_test]
            valid = [y>=0 for y in y_train]
            X_train = [X_train[i][:,valid[i]] for i in range(len(y_train))]
            y_train = [y_train[i][valid[i]] for i in range(len(y_train))]
            valid = [y>=0 for y in y_test]
            X_test = [X_test[i][:,valid[i]] for i in range(len(y_test))]    
            y_test = [y_test[i][valid[i]] for i in range(len(y_test))]    

            # Get rid of class=5
            valid = [y!=5 for y in y_train]
            X_train = [X_train[i][:,valid[i]] for i in range(len(y_train))]
            y_train = [y_train[i][valid[i]] for i in range(len(y_train))]
            valid = [y!=5 for y in y_test]
            X_test = [X_test[i][:,valid[i]] for i in range(len(y_test))]    
            y_test = [y_test[i][valid[i]] for i in range(len(y_test))]    

            y_train = [y*(y<5)+(y-1)*(y>=5) for y in y_train]
            y_test = [y*(y<5)+(y-1)*(y>=5) for y in y_test]

    # lens = []
    # for i in range(len(X_train)):
    #     # lens += [np.max([xx.shape[1] for xx in utils.segment_data(X_train[i], y_train[i])[0] if xx.ndim>1])]
    #     lens += [len(utils.segment_labels(y_train[i]))]
    # print(np.max(lens))
    # lens = []
    # for i in range(len(X_test)):
    #     # lens += [np.max([xx.shape[1] for xx in utils.segment_data(X_test[i], y_test[i])[0] if xx.ndim>1])]
    #     lens += [len(utils.segment_labels(y_test[i]))]
    # print(np.max(lens))    
    # break

    # n_feat = 12
    n_classes = data.n_classes
    n_train = len(X_train)
    n_test = len(X_test)            

    if video:
        # _, y_train = subsample(X_train, y_train, 3*sample_rate)
        # _, y_test = subsample(X_test, y_test, 3*sample_rate)
        X_train, y_train = match_lengths(X_train, y_train, n_feat)
        X_test, y_test = match_lengths(X_test, y_test, n_feat)

        # X_train, y_train = subsample(X_train, y_train, 3)
        # X_test, y_test = subsample(X_test, y_test, 3)
        # _, y_train = subsample(X_train, y_train, 30*sample_rate)
        # _, y_test = subsample(X_test, y_test, 30*sample_rate)
    assert X_train[0].shape[1]==y_train[0].shape[0], "Wrong size"

    if dataset=="JIGSAWS" and not video:
        # X_train, y_train = match_lengths(X_train, y_train, n_feat)
        # X_test, y_test = match_lengths(X_test, y_test, n_feat)

        y_out = utils.remap_labels(np.hstack([y_train, y_test]))
        y_train = y_out[:n_train]
        y_test = y_out[-n_test:]

        X_std = np.hstack(X_train).std(1)[:,None]
        X_train = [(x-x.mean(1)[:,None])/X_std for x in X_train]
        X_test = [(x-x.mean(1)[:,None])/X_std for x in X_test]
    # else:
        # X_train = [x>0 for x in X_train]
        # X_test = [x>0 for x in X_test]


    if remove_bg and dataset == "50Salads":
        valid = [y!=0 for y in y_train]
        X_train = [X_train[i][:,valid[i]] for i in range(len(y_train))]
        y_train = [y_train[i][valid[i]]-1 for i in range(len(y_train))]
        # valid = [y!=0 for y in y_test]
        # X_test = [X_test[i][:,valid[i]] for i in range(len(y_test))]    
        # y_test = [y_test[i][valid[i]]-1 for i in range(len(y_test))]


    if 0:
        from sklearn.decomposition import PCA, FastICA, KernelPCA
        pca = PCA(6)
        pca.fit(np.hstack(X_train).T)
        X_train = [pca.transform(x.T).T for x in X_train]
        X_test = [pca.transform(x.T).T for x in X_test]

    # --------------------------------------------------------------
    # Preprocess VGG
    if model_type == "svm":
        from sklearn.svm import LinearSVC
        # from sklearn.ensemble import RandomForestClassifier
        svm = LinearSVC()
        # svm = RandomForestClassifier()
        svm.fit(np.hstack(X_train).T, np.hstack(y_train))
        # svm.score(np.hstack(X_train).T, np.hstack(y_train))
        print("Test acc: {:.04}%".format(svm.score(np.hstack(X_test).T, np.hstack(y_test))*100))

        P_test = [svm.predict(x.T).T for x in X_test]
        S_test = P_test
        X_train = [svm.decision_function(x.T).T for x in X_train]
        X_test = [svm.decision_function(x.T).T for x in X_test]
        # X_train = [svm.predict_proba(x.T).T for x in X_train]
        # X_test = [svm.predict_proba(x.T).T for x in X_test]       

    # --------- CVPR model ----------
    if model_type == 'cvpr':
        # Go from y_t = {1...C} to one-hot vector (e.g. y_t = [0, 0, 1, 0])
        Y_train = [np_utils.to_categorical(x, n_classes).T for x in y_train]
        Y_test = [np_utils.to_categorical(x, n_classes).T for x in y_test]

        # Mask data such that it's all the same length
        max_len = max(map(len, np.hstack([y_train, y_test])))
        X_train, Y_train, M_train = mask_data(X_train, Y_train, max_len)
        X_test, Y_test, M_test = mask_data(X_test, Y_test, max_len)
        n_timesteps = Y_train[0].shape[0]

        # Learning params
        loss_type = ["mse", "categorical_crossentropy", "hinge", "msle"][0]
        model = build_cvpr(n_nodes, conv, n_classes, n_feat, max_len, 
                            loss_type=loss_type, optimizer="adam")
        # model = build_lstm(n_nodes, conv, n_classes, n_feat, max_len, loss_type=loss_type)
        # model = build_conv_lstm(n_nodes, conv, n_classes, n_feat, max_len, loss_type=loss_type)

        # For Keras, use [n_samples x n_timesteps x n_features]
        X_train = np.array([x.T for x in X_train])
        Y_train = np.array([x.T for x in Y_train])
        X_test = np.array([x.T for x in X_test])
        Y_test = np.array([x.T for x in Y_test])

        model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=1, 
                  verbose=1, sample_weight=M_train[:,:,0])

        # Predict classes
        P_train_raw = model.predict(X_train, verbose=0)
        P_test_raw = model.predict(X_test, verbose=0)
        P_test = model.predict_classes(X_test, verbose=0)
        S_test = P_test

        # Unmask data
        P_train_raw = unmask(P_train_raw, M_train)
        P_test_raw = unmask(P_test_raw, M_test)
        P_test = unmask(P_test, M_test)
        S_test = unmask(S_test, M_test)
        Y_test = unmask(Y_test, M_test)

        # # Add structured model
        model2 = models.PretrainedModel(skip=0, prior=True, debug=True)
        P_train_raw = [p.T for p in P_train_raw]
        P_test_raw = [p.T for p in P_test_raw]
        model2.fit(P_train_raw, y_train, n_iter=100, learning_rate=.1, pretrain=True)
        # model2.filter_len = skip//2 if skip>1 else 1
        # P_test = model2.predict(P_test_raw, inference="filtered")
        S_test = P_test

        # Evaluate using framewise inference
        P_test = model2.predict(P_test_raw, inference="segmental")
        S_test = P_test

        # Perform segmental inference
        # max_segs = utils.max_seg_count(y_train)
        #S_test = [segmental_inferences(P_test_raw[i], max_segs) for i in range(len(y_test))]

    # X_train = [x.T for x in P_train_raw]
    # X_test = [x.T for x in P_test_raw]
    # model_type = "dtw"
    # --------- ICRA model ----------
    # for skip in [0, 20, 25, 30, 400, 500, 600]:
    if model_type == 'icra':
        # Add structured model
        # skip = 25 if video else 20
        # skip = 100//sample_rate
        # conv = skip
        # conv = 10
        skip = conv
        # Just primitives
        # model = models.LatentConvModel(n_latent=1, conv_len=conv, skip=0, prior=False, debug=True)
        model = models.LatentConvModel(n_latent=3, conv_len=conv, skip=skip, prior=True, debug=True)
        
        # model = models.SegmentalModel(pretrained=False)
        model.fit(X_train, y_train, n_iter=300, learning_rate=.1, pretrain=True, verbose=False)

        # Evaluate using structured model
        model.filter_len = skip//2 if skip>1 else 1
        P_test = model.predict(X_test, inference="filtered")
        # P_test = model.predict(X_test, inference="framewise")
        # S_test = model.predict(X_test, inference="segmental")
        # y_tmp = [utils.segment_labels(y) for y in y_train][0]
        # S_test = model.predict(X_test, inference="segmental", known_order=y_tmp)

        from LCTM.energies import pairwise
        pw = np.sum([pairwise.segmental_pw_cost(y, data.n_classes-1) for y in y_train], 0)
        pw /= np.sum(pw, 1)[:,None]
        pw = np.nan_to_num(pw)
        pw = np.log(pw)
        model.potentials["seg_pw"] = pairwise.segmental_pairwise(name="seg_pw")        
        model.ws['seg_pw'] = pw
        # S_test = model.predict(X_test, inference="segmental")
        # P_test = model.predict(X_test, inference="segmental")
        S_test = P_test

    # X_train, X_test = GMM_feat(X_train, y_train, X_test)
    # --------- DTW model ----------
    # if model_type =='dtw':
    #     # X_train, y_train = per_sequences(X_train, y_train)
    #     # X_train, y_train = per_class_sequences(X_train, y_train, ref=0)

    #     X_per_class = per_class_segs(X_train, y_train, ref=0)
    #     seqs_unique = get_unique_orderings(y_train)

    #     if 1:
    #         from sklearn.decomposition import PCA, FastICA, KernelPCA
    #         pca = PCA(6)
    #         # pca = KernelPCA(6, kernel="linear")
    #         # pca = FastICA(6)
    #         pca.fit(np.hstack(X_train).T)
    #         X_train = [pca.transform(x.T).T for x in X_train]
    #         X_test = [pca.transform(x.T).T for x in X_test]

    #     S_test = []
    #     scores_tmp = {idx_task:[]}  
    #     for i in range(len(X_test)):
    #         dists, preds = [], []
    #         scores_tmp[idx_task] = []
    #         for j in range(len(X_train)):
    #             d, c = DTW(X_train[j], X_test[i], output_correspondences=True)
    #             # d /= X_train[j].shape[1]
    #             dists += [d]
    #             preds += [y_train[j][c]]
    #             scores_tmp[idx_task] += [np.mean(y_test[i] == preds[-1])]
    #         idx_best = np.argmin(dists)
    #         # idx_best = np.argmax(scores_tmp[idx_task])
    #         S_test += [preds[idx_best]]
    #     P_test = S_test if P_test is None else P_test

    # This create a sequence using the correct ordering
    if model_type =='dtw':
        # X_train, y_train = per_sequences(X_train, y_train)
        # X_train, y_train = per_class_sequences(X_train, y_train, ref=0)

        X_per_class = per_class_segs(X_train, y_train, ref=0)
        # X_per_class = [x*0+x.mean(1)[:,None] for x in X_per_class]


        S_test = []
        scores_tmp = {idx_task:[]}  
        for i in range(len(X_test)):
            seqs_unique = get_unique_orderings(y_train)
            # True 
            # seq_true = utils.segment_labels(y_test[i])
            # seqs_unique += [seq_true]
            # seqs_unique = [seq_true]
            X_train_, y_train_ = create_X_from_order(X_per_class, seqs_unique)

            if 1:
                from sklearn.decomposition import PCA, FastICA, KernelPCA
                pca = PCA(6)
                # pca = KernelPCA(6, kernel="linear")
                # pca = FastICA(6)
                pca.fit(np.hstack(X_train_).T)
                X_train_ = [pca.transform(x.T).T for x in X_train_]
                X_test_ = [pca.transform(x.T).T for x in X_test]
            else:
                # X_train_ = X_train
                X_test_ = X_test


            dists, preds = [], []
            scores_tmp[idx_task] = []
            for j in range(len(X_train_)):
                dist, c = DTW(X_train_[j], X_test_[i], output_correspondences=True)
                # d /= X_train[j].shape[1]
                dists += [dist]
                preds += [y_train_[j][c]]
                scores_tmp[idx_task] += [np.mean(y_test[i] == preds[-1])]
            idx_best = np.argmin(dists)
            # idx_best = np.argmax(scores_tmp[idx_task])
            S_test += [preds[idx_best]]
        P_test = S_test# if P_test is None else P_test


    if remove_bg and dataset == "50Salads":
        valid = [y!=0 for y in y_test]
        S_test = [S_test[i][valid[i]] for i in range(len(y_test))]
        P_test = [P_test[i][valid[i]] for i in range(len(y_test))]
        y_test = [y_test[i][valid[i]]-1 for i in range(len(y_test))]


    # Compute metrics
    # acc_sCNN = np.mean([metric_fcn(P_cnn_test[i], y_test[i])*100 for i in range(len(y_test))])
    acc_tCNN = np.mean([metric_fcn(P_test[i],y_test[i])*100 for i in range(len(y_test))])
    acc_seg = np.mean([metric_fcn(S_test[i],y_test[i])*100 for i in range(len(y_test))])
    acc_clf = metrics.classification_accuracy(P_test, y_test)
    # idx_task = skip

    avgs[idx_task]= acc_tCNN
    # avgs_seg[idx_task]= acc_seg
    edit_tCNN[idx_task] = metrics.edit_score(P_test, y_test, True)
    edit_rob[idx_task] = metrics.edit_score(P_test, y_test, False)

    # Max_Segs
    max_segs = max([len(utils.segment_labels(y)) for y in y_train])
    max_segs = max(max_segs, max([len(utils.segment_labels(y)) for y in y_test]))
    edit_rob[idx_task] = 100* (1- (edit_rob[idx_task] / max_segs))
    # edit_seg[idx_task] = metrics.edit_score(S_test, y_test)
    # print("{} (Split {} = {:.04}, {:.04})".format(dataset, idx_task, acc_tCNN, acc_seg))
    # print("\t edit = {:.04}, {:.04}".format(edit_tCNN[idx_task], edit_seg[idx_task]))

    print("{} (Split {} = {:.04}, {:.04})".format(dataset, idx_task, acc_tCNN, edit_tCNN[idx_task]))
    # print("\t edit = {:.04}, {:.04}".format(edit_tCNN[idx_task], edit_seg[idx_task]))
    

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
        max_classes = data.n_classes - 1
        # # Output all truth/prediction pairs
        plt.figure(idx_task, figsize=(10,10))
        for i in range(len(y_test)):
            plt.subplot(len(y_test),1,i+1)
            # tmp = np.vstack([y_test[i][:,None].T, P_test[i][:,None].T])
            # tmp = np.vstack([y_test[i][:,None].T, P_test[i][:,None].T, S_test[i][:,None].T])
            # tmp = np.vstack([y_test[i][:,None].T, P_test[i][:,None].T, S_test[i][:,None].T])
            plt.subplot(3,1,1); imshow_(X_test[0])
            plt.subplot(3,1,2); imshow_(y_test[0], vmin=0, vmax=max_classes)
            plt.subplot(3,1,3); imshow_(P_test[0], vmin=0, vmax=max_classes)
            plt.xticks([])

    # ---- Viz weights -----
    if 0:
        # # Output weights at the first layer
        # plt.figure(2, figsize=(10,10))
        # ws = model.get_weights()[0]
        # for i in range(min(36, len(ws[0]))):
        #     plt.subplot(6,6,i+1)
        #     # imshow_(model.get_weights()[0][i][:,:,0]+model.get_weights()[1][i])
        #     imshow_(ws[i][:,:,0])
            # Output weights at the first layer
        plt.figure(2, figsize=(model.n_latent*5,2*model.n_classes))
        # ws = model.get_weights()[0]
        ws = model.ws['conv']
        for i in range(model.n_nodes):
            plt.subplot(model.n_classes, model.n_latent,i+1)
            # imshow_(model.get_weights()[0][i][:,:,0]+model.get_weights()[1][i])
            imshow_(ws[i])
            plt.axis("off")        

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

# print("Skip", skip)
for i in avgs:
    print("{}: {:.04}%, {:.04}%".format(i, avgs[i], edit_tCNN[i]))
avg_mean = np.mean(list(avgs.values()))
edit_seg_mean = np.mean(list(edit_tCNN.values()))
print("Avg: {:.04}%, {:.04}%".format(avg_mean, edit_seg_mean))
avg_mean = np.std(list(avgs.values()))
edit_seg_mean = np.std(list(edit_tCNN.values()))
print("Std: {:.04}%, {:.04}%".format(avg_mean, edit_seg_mean))


if 0:
    Y_all = np.hstack([y_train, y_test])
    mean_len = np.hstack([utils.segment_lengths(y) for y in y_train]).mean()
    total_segs = np.hstack([len(utils.segment_lengths(y)) for y in y_train]).sum()
    frames = np.hstack(Y_all).shape[0]*.01
    pct_avg_segs = frames/mean_len
    n_segs = pct_avg_segs/total_segs

# Distribution of actions per trial
if 0:
    # If idx_task=8 then these are in order...
    Y_all = np.hstack([y_train, y_test])
    segs = [utils.segment_labels(y) for y in Y_all]
    [np.histogram(s, 10)[0] for s in segs]
    hists = np.vstack([np.histogram(s, 10)[0] for s in segs])
    imshow_(hists)



