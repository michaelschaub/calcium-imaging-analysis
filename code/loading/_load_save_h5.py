import numpy as np
import pandas as pd
import h5py
import warnings

def reproducable_hash( a ):
    '''
    should create hashes of DataFrames and ndarrays, that are consitent between saving and loading
    does not work at all, so simply returns 0, may be better to return random number
    '''
    return 0 # not even the string works...

    if isinstance( a, pd.DataFrame ):
        return hash(self._df.to_csv()) # there must be a better way to hash a DataFrame, right?
        # spoiler: No.
    else:
        return hash(a.tostring()) # everything else seems to not work?!

def save_h5(data, file, df=None, attributes=[], attr_files=[], labels=[], hashes=[] ):
    if df is not None:
        df.to_hdf(file, "df", "w")
        h5_file = h5py.File(file, "a")
        h5_file.attrs[f"df_hash"] = hashes[0]
        hashes = hashes[1:]
    else:
        h5_file = h5py.File(file, "w")

    for attr, file, label, hsh  in zip(attributes, attr_files, labels, hashes):
        if  file is None:
            h5_file.create_dataset(label, data=attr)
        else:
            with h5py.File(file, "w") as h5_attr:
                h5_attr.create_dataset(label, data=attr)
            h5_file.attrs[f"{label}_file"] = file
        h5_file.attrs[f"{label}_hash"] = hsh

    return h5_file

def load_h5(file, attr_files=[], labels=[]):
    h5_file = h5py.File(file, "r")
    try:
        df = pd.read_hdf(file, "df")
    except KeyError:
        df = None

    attributes = []
    for file, label in zip( attr_files, labels):
        if file is None:
            if label in h5_file:
                attr = np.array(h5_file[label])
            elif f"{label}_file" in h5_file.attrs:
                with h5py.File(h5_file.attrs[f"{label}_file"], "r") as h5_attr:
                    attr = np.array(h5_attr[label])
            else:
                raise ValueError
        else:
            with h5py.File(file, "r") as h5_attr:
                attr = np.array(h5_attr[label])
        if f"{label}_hash" in h5_file.attrs and h5_file.attrs[f"{label}_hash"] != reproducable_hash(attr):
            warnings.warn(f"{label} hashes do not match", Warning)
        attributes.append(attr)
    return (h5_file, df, *attributes)
