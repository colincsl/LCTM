import os
import numpy as np
import scipy.ndimage as nd
import scipy.io as sio
from LCTM import utils

def closest_file(fid, extension=".mat"):
    # Fix occasional issues with extensions (e.g. X.mp4.mat)
    basename = os.path.basename(fid)
    dirname = os.path.dirname(fid)
    dirfiles = os.listdir(dirname)
    
    if basename in dirfiles:
        return fid
    else:
        basename = basename.split(".")[0]
        files = [f for f in dirfiles if basename in f]
        if extension is not None:
            files = [f for f in files if extension in f]
        if len(files) > 0:
            return dirname+"/"+files[0]
        else:
            print("Error: can't find file")

class Dataset:
    name = ""
    n_classes = None
    n_features = None

    def __init__(self, base_dir=""):
        self.base_dir = base_dir + "{}/".format(self.name)

    def feature_path(self, features):
        return os.path.expanduser(self.base_dir+"features/{}/".format(features))

    def label_path(self, features=""):
        print("Error: Not implemented")
        return None

    def get_files(self, dir_labels, idx_task=None):
        if "Split_1" in os.listdir(dir_labels):
            files_features = np.sort(os.listdir(dir_labels+"/Split_{}/".format(idx_task)))
        else:
            files_features = np.sort(os.listdir(dir_labels))
            
        files_features = [f for f in files_features if f.find(".mat")>=0]
        # files_features = [f.replace(".mat", "") for f in files_features]
        return files_features

    def load_split(self, features="", idx_task=None, feature_type="X", sample_rate=1):
        # Setup directory and filenames
        dir_features = self.feature_path(features)
        dir_labels = self.label_path(features)

        # Get splits for this partion of data
        file_train = open(os.path.expanduser(self.base_dir+"splits/sequences/{}/train.txt".format(idx_task))).readlines()
        file_test = open( os.path.expanduser(self.base_dir+"splits/sequences/{}/test.txt".format(idx_task))).readlines()
        file_train = [f.split(".")[0].strip() for f in file_train]
        file_test = [f.split(".")[0].strip() for f in file_test]

        # Format the train/test split names
        files_features = self.get_files(dir_labels, idx_task)

        # # Load data
        # if "Split_1" in os.listdir(dir_labels):
        #     Y_all = [ sio.loadmat( closest_file("{}/Split_{}/{}".format(dir_labels,idx_task,f)) )["Y"].ravel() for f in files_features]
        # else:
        #     Y_all = [ sio.loadmat( closest_file("{}{}".format(dir_labels,f)) )["Y"].ravel() for f in files_features]

        if "Split_1" in os.listdir(dir_features):
            X_all = [ sio.loadmat( closest_file("{}Split_{}/{}".format(dir_features,idx_task, f)) )[feature_type].astype(np.float64) for f in files_features]
            # Y_all = [ sio.loadmat( closest_file("{}/Split_{}/{}".format(dir_features,idx_task,f)) )["Y"].ravel() for f in files_features]
            Y_all = [ sio.loadmat( closest_file("{}/Split_{}/{}".format(dir_features,idx_task,f)) )["Y"] for f in files_features]
        else:        
            X_all = [ sio.loadmat( closest_file("{}/{}".format(dir_features, f)) )[feature_type].astype(np.float64) for f in files_features]
            # Y_all = [ sio.loadmat( closest_file("{}{}".format(dir_features,f)) )["Y"].ravel() for f in files_features]
            Y_all = [ sio.loadmat( closest_file("{}{}".format(dir_features,f)) )["Y"] for f in files_features]

        # Make sure labels are sequential
        # Y_all = utils.remap_labels(Y_all)

        # Make sure axes are correct (FxT not TxF for F=feat, T=time)
        if X_all[0].shape[0]!=X_all[1].shape[0]:
            X_all = [x.T for x in X_all]
        self.n_features = X_all[0].shape[0]

        # Subsample the data
        if sample_rate > 1:
            X_all, _ = utils.subsample(X_all, Y_all, sample_rate)
            Y_all, _ = utils.subsample(Y_all, Y_all, sample_rate)

        self.n_classes = len(np.unique(np.hstack(Y_all)))

        # ------------Train/test Splits---------------------------
        # Split data/labels into train/test splits
        fid2idx = self.fix2idx(files_features)
        X_train = [X_all[fid2idx[f]] for f in file_train if f in fid2idx]
        X_test = [X_all[fid2idx[f]] for f in file_test if f in fid2idx]
        y_train = [Y_all[fid2idx[f]] for f in file_train if f in fid2idx]
        y_test = [Y_all[fid2idx[f]] for f in file_test if f in fid2idx]

        if len(X_train)==0:
            print("Error loading data")

        return X_train, y_train, X_test, y_test
        
    def load_auxillary(self, features, idx_task=1, feature_type="X", sample_rate=1):
        # Setup directory and filenames
        # dir_features_tmp = self.dir_features
        # self.dir_features = os.path.expanduser(self.base_dir+"features/{}/".format(features))
        
        Z_train, _, Z_test, _ = self.load_split(features, idx_task, feature_type=feature_type, sample_rate=sample_rate)
        # self.dir_features = dir_features_tmp
        
        return Z_train, Z_test
    
    
class JIGSAWS(Dataset):
    n_splits = 8
    name = "JIGSAWS"

    def __init__(self, *args):
        Dataset.__init__(self, *args)

    def label_path(self, features=""):
        return os.path.expanduser(self.base_dir+"labels/sequences/Suturing/")

    def fix2idx(self, files_features):
        if files_features[0].find(".mat"):
            return {files_features[i].replace(".mat",""):i for i in range(len(files_features))}
        else:
            return {files_features[i]:i for i in range(len(files_features))}


class Salads(Dataset):
    n_splits = 5
    name = "50Salads"

    def __init__(self, *args):
        Dataset.__init__(self, *args)

    def label_path(self, features=""):
        return os.path.expanduser(self.base_dir+"features/{}/".format(features))

    def fix2idx(self, files_features):
        return {files_features[i].replace("rgb-","").replace(".mat","").replace(".avi",""):i for i in range(len(files_features))}        


class EndoVis(Dataset):
    n_splits = 7
    name = "EndoVis"

    def __init__(self, *args):
        Dataset.__init__(self, *args)

    def label_path(self, features=""):
        return os.path.expanduser(self.base_dir+"labels/sequences/")

    def fix2idx(self, files_features):
        return {files_features[i].replace(".mat",""):i for i in range(len(files_features))}

class EndoTube(Dataset):
    n_splits = 5
    name = "EndoTube"

    def __init__(self, *args):
        Dataset.__init__(self, *args)

    def label_path(self, features=""):
        # return os.path.expanduser(self.base_dir+"labels/sequences/")
        return os.path.expanduser(self.base_dir+"features/{}/".format(features))

    def fix2idx(self, files_features):
        return {files_features[i].replace(".mat",""):i for i in range(len(files_features))}

