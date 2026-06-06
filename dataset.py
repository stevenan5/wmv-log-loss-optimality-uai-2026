from os.path import join

import numpy as np
from scipy.io import loadmat


class Dataset:
    def __init__(self, name: str, dataset_dir="./datasets"):
        self.name = name
        self.dataset_dir = dataset_dir
        self.load_and_infer_properties()
        self.convert_labels_to_one_hot()

    def load_and_infer_properties(self):
        fname = join(self.dataset_dir, self.name + "_merged.mat")
        self.dataset_dict = loadmat(fname)
        self.train_pred_orig = np.array(self.dataset_dict["train_pred"]).T
        self.train_labels_orig = np.array(self.dataset_dict["train_labels"])

        (self.n_rules, self.n_points) = self.train_pred_orig.shape
        self.n_classes = np.max(self.train_labels_orig) + 1

    def convert_labels_to_one_hot(self):
        def convert_to_one_hot_stack(pred_or_labels):
            # convert pred_or_labels into a stack of matrices, one-hot encodings of each
            # classifier's predictions. index 0 is which classifier, index 1 is
            # the datapoint index, index 2 is the class index
            one_hot_labeling = (
                np.arange(self.n_classes) == pred_or_labels[..., None]
            ).astype(int)
            return one_hot_labeling

        self.train_preds = convert_to_one_hot_stack(self.train_pred_orig).reshape(
            (self.train_pred_orig.shape[0], -1)
        )
        self.train_labels = convert_to_one_hot_stack(self.train_labels_orig).squeeze()
        self.train_labels_flat = self.train_labels.flatten()
