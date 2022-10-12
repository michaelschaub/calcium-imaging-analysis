import numpy as np
import pandas as pd
import h5py
import logging
LOGGER = logging.getLogger(__name__)

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
        # recursively hash all keys and values
        for key,val in a.items():
            hsh = reproducable_hash(key, hsh)
            hsh = reproducable_hash(val, hsh)
    elif isinstance(a, tuple):
        # recursively hash all items
        for val in a:
            hsh = reproducable_hash(val, hsh)
    #elif isinstance(a, str):
        #hsh.update( a.encode("utf-8") )
    else:
        # try to pass a directly into the hash function, works only if a supports buffer API
        try:
            hsh.update( a )
            return hsh
        except TypeError as err:
            pass

        # else try to convert a to np.ndarray (dtype!=object) and hash it
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
    '''
    saves an attribute into some h5 object

    Parameters
    ----------
        h5_obj : some wrapper for some HDF5 object
            The object to save in
        label : string
            The path at which the attribute is saved inside the object
        attr : some SAVEABLE_TYPE
            The attribute to be saved

    Returns
    -------
    h5_obj : wrapper for some HDF5 object
        Since the HDF5 file may have been closed and reopened by this function, a new version of h5_obj is returned
    '''
    if isinstance(attr, dict):
        # create group to store the dictionary in
        new_obj = h5_obj.create_group(label)
        # save file internal path to parent object
        old_path = h5_obj.name
        # recursively save all items of the dictionary
        for key, val in attr.items():
            new_obj = save_object_h5(new_obj, key, val)
        # save attr type as HDF5 attribute
        new_obj.attrs["vtype"] = "dictionary"
        # get updated parent object from new_obj
        h5_obj = new_obj[old_path]
    elif isinstance( attr, np.ndarray ):
        if(attr.dtype.type is np.str_):
            attr = np.array([str.encode('utf-8') for str in attr])
        h5_obj.create_dataset(label, data=attr)
        # save attr type as HDF5 attribute
        h5_obj[label].attrs["vtype"] = "numpy_array"
    elif isinstance( attr, pd.DataFrame ):
        # save file internal path to parent object
        path = h5_obj.name
        # save os path to file
        file = h5_obj.file.filename
        # close file to be able to use pandas to_hdf function
        h5_obj.file.close()
        # use pandas to_hdf function
        attr.to_hdf(file, f"{path}/{label}", "a")
        # reopen file at path
        h5_obj = h5py.File(file, "a")[path]
        # save attr type as HDF5 attribute
        h5_obj[label].attrs["vtype"] = "pandas_frame"
    else:
        return TypeError(f"Cannot save object {attr} of type {type(attr)}.")
    return h5_obj

def load_object_h5(h5_obj, label):
    '''
    loads an attribute from some h5 object

    Parameters
    ----------
    h5_obj : some wrapper for some HDF5 object
        The object to load from
    label : string
        The path at which the attribute is saved inside the object

    Returns
    -------
    h5_obj : HDF5 object
        Since the HDF5 file may have been closed and reopened by this function, a new version of h5_obj is returned
    attr : some SAVEABLE_TYPE
        The loaded attribute
    '''
    if label in list(h5_obj.keys()):
        new_obj = h5_obj[label]
        if new_obj.attrs["vtype"] == "dictionary":
            old_path = h5_obj.name
            d = {}
            for key in new_obj.keys():
                new_obj, d[key] = load_object_h5(new_obj, key)
            return (new_obj[old_path], d)
        elif new_obj.attrs["vtype"] == "numpy_array":
            #if new_obj.dtype == "object"): #if we have more than one string attribute
            if label == "labels": #if we have other object attribtues that are no strings
                return (h5_obj, np.array([str.decode('utf-8') for str in new_obj]))
            else:
                return (h5_obj, np.array(new_obj))
        elif new_obj.attrs["vtype"] == "pandas_frame":
            return (h5_obj, pd.read_hdf(new_obj.file.filename, new_obj.name))
        else:
            raise TypeError(f"Object type {new_obj.attrs['type']} is unkown.")
    else:
        return (h5_obj, None)

def save_h5(data, file, attributes={}, logger=LOGGER ):
    '''
    saves attributes into HDF5 file

    Parameters
    ----------
    file : pathlike
        The os path of the file to save into
    *attributes : {string : SAVEABLE_TYPE}
        The loaded attributes in form of a dictionary

    Returns
    -------
    h5_file : h5py.File object
        a refernce to the File object
    '''
    h5_file = h5py.File(file, "w")

    for label, attr in attributes.items():
        if attr is not None:
            h5_file = save_object_h5( h5_file, label, attr)
            logger.debug(h5_file)
            logger.debug(h5_file[label].attrs)
            h5_file[label].attrs["hash"] = reproducable_hash(attr).hexdigest()

    return h5_file

def load_h5(file, labels=[], logger=LOGGER):
    '''
    loads attributes from HDF5 file

    Parameters
    ----------
    file : pathlike
        The os path of the file to load from
    labels : [string]
        The paths at which the attributes are saved inside the file

    Returns
    -------
    h5_file : h5py.File object
        a refernce to the File object
    *attributes : [SAVEABLE_TYPE's]
        The loaded attributes
    '''
    h5_file = h5py.File(file, "r")

    attributes = []
    for label in labels:
        # load attribute from h5_file
        h5_file, attr = load_object_h5(h5_file, label)

        # warn if CHECK_HASH is true, hash attribute exists and does not match calculated hash
        if label in h5_file:
            if CHECK_HASH and ( "hash" in h5_file[label].attrs and
                h5_file[label].attrs["hash"] != reproducable_hash(attr, vtype=h5_file[label].attrs["vtype"]).hexdigest()):
                logger.warn(f"{label} hashes do not match" )

        attributes.append(attr)
    return (h5_file, *attributes)
