import os
import glob
import itertools
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataset(dirs, extension=None, process_fn=None):
    """
    Create dataset (dict like {key: [values]}) from vctk-like dataset:
        dataset/
            cat/
                *.ext
            dog/
                *.ext

    :param dirs: path to dirs, for example /home/user/data/**
    :param extension: data extension you are looking for
    :param process_fn:
        function(path_to_file) -> object
        process function for found files, by default
    :return: 
    """
    extension = extension or "*"
    dataset = defaultdict(list)
    
    dirs = [os.path.expanduser(k) for k in dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    for d in dirs:
        label = os.path.basename(d.rstrip('/'))
        files = glob.glob(d + "/" + extension)
        if process_fn is None:
            dataset[label].extend(files)
        else:
            dataset[label].extend(list(map(lambda x: process_fn(x), files)))
    
    return dataset


def split_dataset(dataset, **train_test_split_args):
    """
    Split dataset in train and test parts.

    :param dataset: dict
    :param train_test_split_args:
        test_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split. If
            int, represents the absolute number of test samples. If None,
            the value is automatically set to the complement of the train size.
            If train size is also None, test size is set to 0.25.

        train_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        random_state : int or RandomState
            Pseudo-random number generator state used for random sampling.

        stratify : array-like or None (default is None)
            If not None, data is split in a stratified fashion, using this as
            the class labels.
    :return: train and test dicts
    """
    train_dataset = defaultdict(list)
    test_dataset = defaultdict(list)
    for key, value in dataset.items():
        train_ids, test_ids = train_test_split(
            range(len(value)), 
            **train_test_split_args)
        train_dataset[key].extend([value[i] for i in train_ids])
        test_dataset[key].extend([value[i] for i in test_ids])
    return train_dataset, test_dataset


def create_dataframe(dataset, **dataframe_args):
    """
    Create pd.DataFrame for dict like {key: [values]}

    :param dataset: dict like {key: [values]}
    :param dataframe_args:
        index : Index or array-like
            Index to use for resulting frame. Will default to np.arange(n) if
            no indexing information part of input data and no index provided
        columns : Index or array-like
            Column labels to use for resulting frame. Will default to
            np.arange(n) if no column labels are provided
        dtype : dtype, default None
            Data type to force, otherwise infer
    :return:
    """
    data = [(key, value) for key, values in dataset.items() for value in values]
    df = pd.DataFrame(data, **dataframe_args)
    return df


def split_dataframe(df, **train_test_split_args):
    """
    Split dataframe in train and test part.
    PS. exist cause I dont like this complicated sklearn import.

    :param df: pd.DataFrame to split
    :param train_test_split_args:
        test_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split. If
            int, represents the absolute number of test samples. If None,
            the value is automatically set to the complement of the train size.
            If train size is also None, test size is set to 0.25.

        train_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        random_state : int or RandomState
            Pseudo-random number generator state used for random sampling.

        stratify : array-like or None (default is None)
            If not None, data is split in a stratified fashion, using this as
            the class labels.
    :return: train and test DataFrames
    """
    df_train, df_test = train_test_split(
        df, **train_test_split_args)
    return df_train, df_test

