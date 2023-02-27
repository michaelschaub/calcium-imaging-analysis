import pytest

from ci_lib.data import DecompData
from ci_lib.features import Features, Raws, Means

SEED = 2354
#TODO replace with actual dummy data
DUMMY_DATA='results/data/GN06.03-26#GN06.03-29/anatomical/All/data.h5'

def test_raw_creation():
    data = DecompData.load(DUMMY_DATA)
    raws = Raws.create(data[:,:50])

def test_means_creation():
    data = DecompData.load(DUMMY_DATA)
    means = Means.create(data[:,:50])

def test_subsample():
    data = DecompData.load(DUMMY_DATA)
    raws = Raws.create(data[:,:50])
    s_data = data.subsample(10, SEED)
    s_raws = Raws.create(s_data[:,:50])
    assert((s_raws.feature == raws.subsample(10, SEED).feature).all())
