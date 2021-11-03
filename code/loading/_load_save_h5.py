import numpy as np
import pandas as pd
import h5py
import warnings

def reproducable_hash( a ):
    '''
    should create hashes of DataFrames and ndarrays, that are consitent between saving and loading
    does not really work, basically returns random number unique per data and run
    '''
    #TODO: fix
    if isinstance( a, pd.DataFrame ):
        return hash(a.to_csv()) # there must be a better way to hash a DataFrame, right?
        # spoiler: No.
    elif isinstance( a, np.ndarray ):
        return hash(a.tostring()) # everything else seems to not work?!
    elif isinstance( a, list):
        return hash(tuple(a))
    else:
        return hash(a)

def save_object_h5(h5_obj, label, attr):
    if isinstance(attr, dict):
        for key, val in attr.items():
            save_object_h5(h5_obj, f"{label}/{key}", val)
    else:
        h5_obj.create_dataset(label, data=attr)

def load_object_h5(h5_obj, label):
    obj = h5_obj[label]
    if isinstance(obj, h5py.Group):
        return { key: load_object_h5(obj, key) for key in obj.keys() }
    else:
        return np.array(obj)

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
            save_object_h5( h5_file, label, attr)
        else:
            with h5py.File(file, "w") as h5_attr:
                save_object_h5( h5_attr, label, attr)
            h5_file.attrs[f"{label}_file"] = file
        if hsh is not None:
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
                attr = load_object_h5(h5_file, label)
            elif f"{label}_file" in h5_file.attrs:
                with h5py.File(h5_file.attrs[f"{label}_file"], "r") as h5_attr:
                    attr = load_object_h5(h5_attr, label)
            else:
                raise ValueError
        else:
            with h5py.File(file, "r") as h5_attr:
                attr = load_object_h5(h5_attr, label)
        if f"{label}_hash" in h5_file.attrs and h5_file.attrs[f"{label}_hash"] != reproducable_hash(attr):
            warnings.warn(f"{label} hashes do not match", Warning)
        attributes.append(attr)
    return (h5_file, df, *attributes)
