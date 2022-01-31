from glob import glob
import numpy as np
import h5py
import pandas as pd
import copy
from os import path
import pickle as pkl
from pathlib import Path
import logging
LOGGER = logging.getLogger(__name__)


def load_individual_session_data(date_folder, mouse_id, logger=None):
    logger = LOGGER if logger is None else logger.getChild(__name__)
    file_paths = list(date_folder.glob('*.h5'))
    if len(file_paths) == 0:
        file_paths = list((date_folder / r'task_data').glob('*.h5'))
    #

    file_paths = np.sort(file_paths)

    data = {'mouse_id': list(),
            'date_time': list(),
            'trial_id': list(),
            'modality': list(),
            'target_side_left': list(),
            'response_left': list(),
            'auto_reward': list(),
            'both_spouts': list(),
            'n_distractors': list(),
            'n_targets': list(),
            'cues_left': list(),
            'cues_right': list(),
            'cues_left_vis': list(),
            'cues_right_vis': list(),
            'cues_left_tact': list(),
            'cues_right_tact': list(),
            'control_condition_id': list()}

    for trial_file in file_paths:
        #logger.info(f"trial file: \"{trial_file}\"")
        # in case a file wasn't saved correctly.
        # e.g.: software was stopped in the middle of a trial no using the stop button
        # I'm first trying to read all the data and put it into a temporary dictionary and only if everything could be
        # read correctly them im copying the entire thing into data
        temp_data = copy.deepcopy(data)
        try:
            h5f = h5py.File(trial_file, 'r')

            temp_data['modality'].append(int(h5f['modality'][0]))
            temp_data['target_side_left'].append(int(h5f['target_side_left'][0]))
            temp_data['response_left'].append(int(h5f['Response_left'][0]))
            temp_data['auto_reward'].append(int(h5f['auto_reward'][0]))
            if 'both_spouts' in h5f.keys():
                temp_data['both_spouts'].append(int(h5f['both_spouts'][0]))
            else:
                temp_data['both_spouts'].append(np.nan)
            #
            try:
                if 'n_distractors' in h5f.keys():
                    temp_data['n_distractors'].append(int(h5f['n_distractors'][0]))
                    temp_data['n_targets'].append(6)
                else:
                    raise KeyError()
                #
            except:
                if temp_data['modality'][-1] == 1:
                    # tactile_case
                    temp_data['n_distractors'].append(int(min(h5f['cues_left_tactile'][:].sum(), h5f['cues_right_tactile'][:].sum())))
                    temp_data['n_targets'].append(int(max(h5f['cues_left_tactile'][:].sum(), h5f['cues_right_tactile'][:].sum())))
                else:
                    temp_data['n_distractors'].append(int(min(h5f['cues_left_visual'][:].sum(), h5f['cues_right_visual'][:].sum())))
                    temp_data['n_targets'].append(int(max(h5f['cues_left_visual'][:].sum(), h5f['cues_right_visual'][:].sum())))
                #
            #
            temp_data['mouse_id'].append(mouse_id)
            temp_data['date_time'].append(path.basename(date_folder))
            temp_data['trial_id'].append(int(path.splitext(path.basename(trial_file))[0][-6:]))

            if 'cues_left_visual' in h5f.keys():
                temp_data['cues_left'].append(np.int8(np.logical_or(np.array(h5f['cues_left_visual']) > 0,
                                              np.array(h5f['cues_left_tactile']) > 0)))
                temp_data['cues_right'].append(np.int8(np.logical_or(np.array(h5f['cues_right_visual']) > 0,
                                               np.array(h5f['cues_right_tactile']) > 0)))
                temp_data['cues_left_vis'].append(np.int8(np.array(h5f['cues_left_visual']) > 0))
                temp_data['cues_right_vis'].append(np.int8(np.array(h5f['cues_right_visual']) > 0))

                temp_data['cues_left_tact'].append(np.int8(np.array(h5f['cues_left_tactile']) > 0))
                temp_data['cues_right_tact'].append(np.int8(np.array(h5f['cues_right_tactile']) > 0))

                # h5f['cues_left_visual']
                # h5f['cues_left_tactile']
            else:
                temp_data['cues_left'].append(np.nan)
                temp_data['cues_right'].append(np.nan)

                temp_data['cues_left_vis'].append(np.nan)
                temp_data['cues_right_vis'].append(np.nan)

                temp_data['cues_left_tact'].append(np.nan)
                temp_data['cues_right_tact'].append(np.nan)
            #

            if 'control_condition_id' in h5f.keys():
                temp_data['control_condition_id'].append(int(h5f['control_condition_id'][0]))
            else:
                temp_data['control_condition_id'].append(int(0))
            #


            # All the additional data:
            # temp_data = np.array(h5f['DI']).copy()
            # temp_wheel = np.array(h5f['wheel']).copy()
            # temp_autoreward = np.array(h5f['wheel']).copy()
            h5f.close()

            # only accept temp_data if the file could be read correctly
            data = copy.deepcopy(temp_data)
        except ValueError:
            # this occurs when the session was interrupted during a trial. Usually the last trial in a session.
            pass
        except Exception as e:
            logger.info(e)
            break
        #
    #

    # # plt.plot(wheel)
    # plt.imshow(data, interpolation='nearest', aspect='auto')
    # plt.show()

    return data


#


def extract_session_data_and_save(root_paths,  mouse_dates_str , reextract=False, logger=None):
    # if type(root_paths) is str:
    #     root_paths = [root_paths]
    # #
    logger = LOGGER if logger is None else logger.getChild(__name__)
    logger.info(root_paths)
    initial_call = True
    save_individually = False
    for mouse_id, date_paths  in root_paths.items():
        logger.info(f"Mouse {mouse_id}")
        #date_paths = sorted((root_path / str(mouse_id)).glob( "*" )) ###
        logger.info(f"Path {date_paths}")
        # sessions = list()

        n_paths = len(date_paths)
        for n, date_folder in enumerate(date_paths):
            date_folder = Path(date_folder)
            logger.info(f"Processed {date_folder} ({n}/{n_paths})")
            # mouse_id = 'AB24'
            session_pkl = date_folder / 'performance_data_extracted.pkl'
            if (not reextract) and path.isfile(session_pkl):
                with open(session_pkl, 'rb') as handle:
                    data = pkl.load(handle)
                #
            else:

                #logger.info(f"date_folder: \"{root_path}\"")
                data = load_individual_session_data(date_folder=date_folder, mouse_id=mouse_id, logger=logger)
                if save_individually:
                    with open(session_pkl, 'wb') as handle:
                        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
            if initial_call:
                sessions = pd.DataFrame.from_dict(data)
                initial_call = False
            else:
                sessions = sessions.append(pd.DataFrame.from_dict(data), ignore_index=True)
            #
        logger.info(f"Processed {date_paths[-1]} ({n_paths}/{n_paths})")
        #

    # Some of the old files have a mixup of "_" and "-"
    #sessions = fix_dates(sessions)


    #pkl.dump(sessions, open(root_path/"MS_task_V2_1_extracted_data.pkl", 'wb')) # we don't want sideeffects besides the sankemake results folder
    #
    return sessions


def fix_dates(sessions, logger=None):
    logger = LOGGER if logger is None else logger.getChild(__name__)
    unique_dates = sessions['date_time'].unique()
    new_dates = unique_dates.copy()
    for i, date in enumerate(unique_dates):
        date_parts = date.replace('-', '_').split('_')
        # logger.info(date_parts)
        new_dates[i] = date_parts[0] + '-' + date_parts[1] + '-' + date_parts[2] + '_' + \
                       date_parts[3] + '-' + date_parts[4] + '-' + date_parts[5]
    #
    mismatching_ids = new_dates != unique_dates
    new_dates = new_dates[mismatching_ids]
    unique_dates = unique_dates[mismatching_ids]

    # now replace the mismatching entries
    for old_date, new_date in zip(unique_dates, new_dates):
        sessions.loc[sessions['date_time'] == old_date, 'date_time'] = new_date
    #
    return sessions
#


if __name__ == '__main__':
    modality_keys = ['visual', 'tactile', 'visual-tactile']
    mouse_ids = ['GN06', 'GN07', 'GN08', 'GN09', 'GN10', 'GN11']

    root_paths = [ Path(__file__).parent.parent / Path('data/GN06/2021-01-20_10-15-16') ]
    run_extraction = True

    # extract sessions if desired
    if run_extraction:
        sessions = extract_session_data_and_save(root_paths=root_paths, mouse_ids=mouse_ids, reextract=True)
    else:
        # load data
        path.basename(root_paths[0])
        with open(path.basename(root_paths[0]) + '_extracted_data.pkl', 'rb') as handle:
            sessions = pkl.load(handle)
            sessions = fix_dates(sessions)

        #
    #
#
