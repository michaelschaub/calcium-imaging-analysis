import numpy as np
import pandas as pd
import h5py
import warnings

CHECK_HASH = False

def reproducable_hash( a, vtype=None ):
    '''
    should create hashes of DataFrames and ndarrays, that are consitent between saving and loading
    does not really work, basically returns random number unique per data and run
    '''
    #TODO: fix
    if vtype == "panda_frame" or isinstance( a, pd.DataFrame ):
        return hash(a.to_csv()) # there must be a better way to hash a DataFrame, right?
        # spoiler: No.
    elif vtype == "numpy_array" or isinstance( a, np.ndarray ):
        return hash(a.tostring()) # everything else seems to not work?!
    elif vtype == "dictionary" or isinstance(a, dict):
        return hash(tuple( (key, reproducable_hash(val)) for key,val in a.items() ))
    else:
        try:
            return hash(a)
        except TypeError as err:
            raise TypeError(f"Cannot reproducably hash object {a} of type {type(a)}.") from None


SAVEABLE_TYPES = [
        "numpy_array",
        "panda_frame",
        "dictionary",
        ]

def save_object_h5(h5_obj, label, attr):
    if isinstance(attr, dict):
        new_obj = h5_obj.create_group(label)
        old_path = h5_obj.name
        for key, val in attr.items():
            new_obj = save_object_h5(new_obj, key, val)
        new_obj.attrs["vtype"] = "dictionary"
        h5_obj = new_obj[old_path]
    elif isinstance( attr, np.ndarray ):
        h5_obj.create_dataset(label, data=attr)
        h5_obj[label].attrs["vtype"] = "numpy_array"
    elif isinstance( attr, pd.DataFrame ):
        path = h5_obj.name
        file = h5_obj.file.filename
        h5_obj.file.close()
        attr.to_hdf(file, f"{path}/{label}", "a")
        h5_obj = h5py.File(file, "a")[path]
        h5_obj[label].attrs["vtype"] = "pandas_frame"
    else:
        return TypeError(f"Cannot save object {attr} of type {type(attr)}.")
    return h5_obj

def load_object_h5(h5_obj, label):
    new_obj = h5_obj[label]
    if new_obj.attrs["vtype"] == "dictionary":
        old_path = h5_obj.name
        d = {}
        for key in new_obj.keys():
            new_obj, d[key] = load_object_h5(new_obj, key)
        return new_obj[old_path], d
    elif new_obj.attrs["vtype"] == "numpy_array":
        return h5_obj, np.array(new_obj)
    elif new_obj.attrs["vtype"] == "pandas_frame":
        return h5_obj, pd.read_hdf(new_obj.file.filename, new_obj.name)
    else:
        return TypeError(f"Object type {new_obj.attrs['type']} is unkown.")

def save_h5(data, file, attributes={} ):
    h5_file = h5py.File(file, "w")

    for label, attr in attributes.items():
        h5_file = save_object_h5( h5_file, label, attr)
        h5_file[label].attrs["hash"] = reproducable_hash(attr)

    return h5_file

def load_h5(file, labels=[] ):
    h5_file = h5py.File(file, "r")

    attributes = []
    for label in labels:
        h5_file, attr = load_object_h5(h5_file, label)

        if CHECK_HASH and ( "hash" in h5_file[label].attrs and
            h5_file[label].attrs["hash"] != reproducable_hash(attr, h5_file[label].attrs["vtype"])):
            warnings.warn(f"{label} hashes do not match", Warning)

        attributes.append(attr)
    return h5_file, *attributes
