import numpy as np
import pandas as pd
import h5py
import warnings

from hashlib import sha1

CHECK_HASH = True

def reproducable_hash( a, hsh=None, vtype=None ):
    '''
    creates hashes, that are consistent between saving and loading
    '''
    if hsh is None:
        hsh = sha1()

    if vtype == "panda_frame" or isinstance( a, pd.DataFrame ):
        hsh.update(a.to_csv().encode("utf-8"))
    elif vtype == "numpy_array" or isinstance( a, np.ndarray ):
        hsh.update(a.tobytes())
    elif vtype == "dictionary" or isinstance(a, dict):
        for key,val in a.items():
            hsh = reproducable_hash(key, hsh)
            hsh = reproducable_hash(val, hsh)
    elif isinstance(a, tuple):
        for val in a:
            hsh = reproducable_hash(val, hsh)
    #elif isinstance(a, str):
        #hsh.update( a.encode("utf-8") )
    else:
        try:
            hsh.update( a )
            return hsh
        except TypeError as err:
            pass

        try:
            arr = np.array(a)
            error = False
        except TypeError as err:
            error = True
        if error or arr.dtype.hasobject:
            raise TypeError(f"Cannot reproducably hash object {a} of type {type(a)}.")
        hsh.update(arr.tobytes())
    return hsh


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
        if(attr.dtype.type is np.str_):
            attr = np.array([str.encode('utf-8') for str in attr])
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
    if label in list(h5_obj.keys()):
        new_obj = h5_obj[label]
        if new_obj.attrs["vtype"] == "dictionary":
            old_path = h5_obj.name
            d = {}
            for key in new_obj.keys():
                new_obj, d[key] = load_object_h5(new_obj, key)
            return new_obj[old_path], d
        elif new_obj.attrs["vtype"] == "numpy_array":
            #if new_obj.dtype == "object"): #if we have more than one string attribute
            if label == "labels": #if we have other object attribtues that are no strings
                return h5_obj, np.array([str.decode('utf-8') for str in new_obj])
            else:
                return h5_obj, np.array(new_obj)
        elif new_obj.attrs["vtype"] == "pandas_frame":
            return h5_obj, pd.read_hdf(new_obj.file.filename, new_obj.name)
        else:
            return TypeError(f"Object type {new_obj.attrs['type']} is unkown.")
    else:
        return h5_obj, None

def save_h5(data, file, attributes={} ):
    h5_file = h5py.File(file, "w")

    for label, attr in attributes.items():
        if attr is not None:
            h5_file = save_object_h5( h5_file, label, attr)
            h5_file[label].attrs["hash"] = reproducable_hash(attr).hexdigest()

    return h5_file

def load_h5(file, labels=[] ):
    h5_file = h5py.File(file, "r")

    attributes = []
    for label in labels:
        h5_file, attr = load_object_h5(h5_file, label)

        if hasattr(h5_file,label):
            if CHECK_HASH and ( "hash" in h5_file[label].attrs and
                h5_file[label].attrs["hash"] != reproducable_hash(attr, vtype=h5_file[label].attrs["vtype"]).hexdigest()):
                warnings.warn(f"{label} hashes do not match", Warning)

        attributes.append(attr)
    return h5_file, *attributes
