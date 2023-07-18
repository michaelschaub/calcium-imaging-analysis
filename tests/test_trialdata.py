import pandas as pd
import numpy as np

from ci_lib.data.trialdata import Event, TrialData

df = pd.DataFrame({
    "trial" : [0,1,2,3],
    "value" : [4.,2.,8.,1.]})

print("Testing simple Event")
starts = np.array([3,7,16,24])
trial_ind = Event(starts)
assert len(trial_ind) == 4
assert trial_ind.indices.shape == (4,1)
assert (trial_ind.indices[:,0] == starts).all()

print("Testing Event with complex times and arguments")
stim_t = [[4,5],[9,10],[17],[26,28]]
stim_m = ["t","v","vt","t"]
stimulus = Event(stim_t, modality=stim_m)
assert len(stimulus) == 4
assert np.all([np.all(s1 == s2) for s1,s2 in zip(stimulus.indices, stim_t)])
assert (stimulus.kwarg('modality') == stim_m).all()
assert (stimulus.modality == stim_m).all()

stim_t = np.array(stim_t, dtype=object)
stim_m = np.array(stim_m, dtype=object)
def test_event_getitem(key, key_alt=None):
    if key_alt is None:
        key_alt = key
    print(f"Testing Event slicing with {key=}")
    stim_slc = stimulus[key]
    assert len(stim_slc) == len(stim_t[key_alt])
    assert np.all([np.all(s1 == s2) for s1,s2 in zip(stim_slc.indices, stim_t[key_alt])])
    assert (stim_slc.kwarg('modality') == stim_m[key_alt]).all()
    assert (stim_slc.modality == stim_m[key_alt]).all()

test_event_getitem(1, slice(1,2))
test_event_getitem([0,1,3])
test_event_getitem(slice(0,None,2))
test_event_getitem(slice(0,2))
test_event_getitem([True,False,False,True])

print("Testing Event addition")
event1 = trial_ind + 3
assert (event1.indices[:,0] == starts+3).all()
offset = np.array([0,0,-2,-2])
event2 = trial_ind + offset
assert (event2.indices[:,0] == starts+offset).all()
event3 = stimulus + offset
assert np.all([ e == s + o
               for e_indx, stim, o in zip(event3.indices, stim_t, offset)
               for e,s in zip(e_indx,stim)])

print("Testing Constructor with event objects")
td1 = TrialData(df, trial=trial_ind, stimulus=stimulus)
assert (td1.trials == starts).all()
assert (td1.starts == starts).all()
assert (td1.stops == [*starts[1:], -1]).all()

trial_times = np.array([[4,2,-1],
                        [8,6,-1],
                        [11,10,-1],
                        [16,16,20]])

print("Testing Constructor with event indices or tuple")
td2 = TrialData(df, trial=trial_times, stimulus=(stim_t,{"modality":stim_m}))
assert (td2.trials == trial_times[:,0]).all()
assert (td2.starts == trial_times[:,1]).all()
assert (td2.stops == [*trial_times[1:,1], 20]).all()
assert np.all([np.all(s1 == s2).all() for s1,s2 in zip(td2.events["stimulus"].indices, stim_t)])
assert (td2.events["stimulus"].modality == stim_m).all()

print("Testing copy")
td3 = td1.copy()
assert (td1.trials == td3.trials).all()
assert np.all([ np.all([ (i1 == i2).all()
                         for i1,i2 in zip(td1.events[e].indices, td3.events[e].indices)])
                for e in td3.events])
assert np.all([td1.events[e] is not td3.events[e] for e in td3.events])
assert td1.dataframe is not td2.dataframe

print("Testing concat")
td4 = td2.copy().concat(td1.copy())
assert len(td4) == len(td2)+len(td1)
assert (td4.dataframe == pd.concat((td2.dataframe, td1.dataframe)).reset_index(drop=True)).all(axis=None)
assert np.all(td4.trials == np.concatenate((trial_times[:,0], starts+trial_times[-1,-1])))
assert np.all(td4.starts == np.concatenate((trial_times[:,-2], starts+trial_times[-1,-1])))
assert np.all(td4.stops  == np.concatenate((
    [*(trial_times[1:,-2]), trial_times[-1,-1]],
    [*(starts[1:]+trial_times[-1,-1]), -1])))
assert np.all(td4.events['stimulus'].modality == np.concatenate((td2.events['stimulus'].modality, td1.events['stimulus'].modality)))


def test_trialdata_getitem(key, key_alt=None, iloc=True):
    if key_alt is None:
        key_alt = key
    print(f"Testing TrialData slicing with {key=}")
    td_slc = td1[key]
    assert len(td_slc) == len(td1.events['trial'][key])
    assert np.all([(s1 == s2).all() for s1, s2 in zip(td_slc.events['stimulus'].indices, td1.events['stimulus'][key].indices)])
    if iloc:
        assert (td_slc.dataframe == td1.dataframe.iloc[key_alt].reset_index(drop=True)).all(axis=None)
    else:
        assert (td_slc.dataframe == td1.dataframe.loc[key_alt].reset_index(drop=True)).all(axis=None)

test_trialdata_getitem(1, slice(1,2))
test_trialdata_getitem([0,1,3])
test_trialdata_getitem(slice(0,None,2))
test_trialdata_getitem(slice(0,2))
test_trialdata_getitem([True,False,False,True], iloc=False)

print("Testing get_conditional with list of values")
condition1 = {"trial" : [0,1,2]}
td_cond1 = td1.get_conditional(condition1)
assert (td_cond1.dataframe == td1.dataframe.iloc[[0,1,2]].reset_index(drop=True)).all(axis=None)

print("Testing get_conditional with lambdas")
condition2 = {"value" : lambda x: x < 6}
td_cond2 = td1.get_conditional(condition2)
assert (td_cond2.dataframe == td1.dataframe.loc[td1.dataframe['value']<6].reset_index(drop=True)).all(axis=None)

print("Testing get_conditional multiple conditions")
condition = {**condition1, **condition2}
td_cond3 = td1.get_conditional(condition)
assert (td_cond3.dataframe == td1.dataframe.iloc[[0,1,2]].loc[td1.dataframe['value']<6].reset_index(drop=True)).all(axis=None)

print("Testing subsample")
td_sub = td2.subsample(3, seed=42)
