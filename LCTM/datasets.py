import os
import numpy as np
import scipy.ndimage as nd
import scipy.io as sio
from LCTM import utils

class Dataset:
    name = ""
    features = ""
    n_classes = None
    n_features = None
    sep_splits = False

    def __init__(self, base_dir, features, sep_splits=False):
        self.base_dir = base_dir + "{}/".format(self.name)
        self.features = features
        self.sep_splits = sep_splits


    def get_files(self, idx_task=None):
        if self.sep_splits:
            files_features = np.sort(os.listdir(self.dir_labels+"/Split_{}/".format(idx_task)))
        else:
            files_features = np.sort(os.listdir(self.dir_labels))
            
        files_features = [f for f in files_features if f.find(".mat")>=0]
        # files_features = [f.replace(".mat", "") for f in files_features]
        return files_features

    def load_split(self, idx_task):

        # Get splits for this partion of data
        file_train = open(os.path.expanduser(self.base_dir+"splits/sequences/{}/train.txt".format(idx_task))).readlines()
        file_train = [f.strip() for f in file_train]
        file_test = open( os.path.expanduser(self.base_dir+"splits/sequences/{}/test.txt".format(idx_task))).readlines()
        file_test = [f.strip() for f in file_test]

        # Format the train/test split names
        files_features = self.get_files(idx_task)

        # Load data
        if self.sep_splits:
            Y_all = [ sio.loadmat( "{}/Split_{}/{}".format(self.dir_labels,idx_task,f) )["Y"].ravel() for f in files_features]
            X_all = [ sio.loadmat( "{}/Split_{}/{}".format(self.dir_features,idx_task, f) )["X"].astype(np.float64) for f in files_features]
        else:
            Y_all = [ sio.loadmat( "{}{}".format(self.dir_labels,f) )["Y"].ravel() for f in files_features]
            X_all = [ sio.loadmat( "{}/{}".format(self.dir_features, f) )["X"].astype(np.float64) for f in files_features]

        # Make sure labels are sequential
        Y_all = utils.remap_labels(Y_all)

        if self.name == "50Salads":
            Y_all = [nd.median_filter(y, 300) for y in Y_all]

        # Make sure axes are correct (FxT not TxF for F=feat, T=time)
        if X_all[0].shape[0] > X_all[0].shape[1]:
            X_all = [x.T for x in X_all]

        self.n_classes = len(np.unique(np.hstack(Y_all)))
        self.n_features = X_all[0].shape[0]

        # ------------Train/test Splits---------------------------
        # Split data/labels into train/test splits
        fid2idx = self.fix2idx(files_features)
        X_train = [X_all[fid2idx[f]] for f in file_train if f in fid2idx]
        X_test = [X_all[fid2idx[f]] for f in file_test if f in fid2idx]
        y_train = [Y_all[fid2idx[f]] for f in file_train if f in fid2idx]
        y_test = [Y_all[fid2idx[f]] for f in file_test if f in fid2idx]

        return X_train, y_train, X_test, y_test
        
    
class JIGSAWS(Dataset):
    n_splits = 7
    name = "JIGSAWS"

    def __init__(self, base_dir, features, sep_splits=False):
        Dataset.__init__(self, base_dir, features, sep_splits)

        # Setup directory and filenames
        self.dir_labels = os.path.expanduser(self.base_dir+"labels/sequences/Suturing/")
        self.dir_features = os.path.expanduser(self.base_dir+"features/{}/".format(self.features))

    def fix2idx(self, files_features):
        if files_features[0].find(".mat"):
            return {files_features[i].replace(".mat",""):i for i in range(len(files_features))}
        else:
            return {files_features[i]:i for i in range(len(files_features))}

class Salads(Dataset):
    n_splits = 5
    name = "50Salads"

    def __init__(self, base_dir, features, sep_splits=False):
        Dataset.__init__(self, base_dir, features, sep_splits)

        # Setup directory and filenames
        self.dir_features = os.path.expanduser(self.base_dir+"features/{}/Split_1/".format(self.features))
        self.dir_labels = os.path.expanduser(self.base_dir+"features/{}/Split_1/".format(self.features))

    def fix2idx(self, files_features):
        return {files_features[i].replace("rgb-","").replace(".mat","").replace(".avi",""):i for i in range(len(files_features))}        

class EndoVis(Dataset):
    n_splits = 7
    name = "EndoVis"

    def __init__(self, base_dir, features, sep_splits=False):
        Dataset.__init__(self, base_dir, features, sep_splits)

        # Setup directory and filenames
        self.dir_features = os.path.expanduser(self.base_dir+"features/{}/".format(self.features))
        self.dir_labels = os.path.expanduser(self.base_dir+"features/{}/".format(self.features))

    def fix2idx(self, files_features):
        return {files_features[i].replace(".mat",""):i for i in range(len(files_features))}

