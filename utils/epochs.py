#%%imports

#import sys
#sys.path.append('/mnt/obob/staff/fschmidt/vsm/helpers/')
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

sys.path.append(f'{FABI_DIR}/utils/')

import mne
import numpy as np
from obob_mne.events import read_events_from_analogue
from utils.events import fix_events_from_analogue


def get_epochs(data_raw, 
               event_info, 
               epochs_settings,
               events = None):
    '''
    
    This function can be used to extract events of interest from the ncc experiment and them into epochs.

    task -> contains information whether the block was auditory=1, somatosensory=2 or visual=3
    StimuliOrder2 -> contains information about the stimulus 1,2,3,4
    StimuliOrder3 -> contains information about the stimulus category (should be 1)
    response -> 1,0 probably hit/miss

    '''
    print(f"---Now Doing: events = read_events_from_analogue(data_raw)---", flush=True)       
    #% get events
    events = read_events_from_analogue(data_raw) #need to do this as sth in mne is fucked up when it comes to getting events

    print("\n---------------------------------------------------------------------------------\n",
              f"Shape of the events; {events.shape[0]}\n",
               "---------------------------------------------------------------------------------\n", flush=True)
        
    # ------------------------------------------------------------------------------------------------------- v This is what I added
    if events.shape[0] < 1440:
        assert events.shape[0] in [1296, 1320], "There are events missing! But the number of events is not 108 as it should be if all catch trials were missing! It also isn't 1320 which would be the case for subject 'smsr' because one block is missing"
        print("\n---------------------------------------------------------------------------------\n",
              f"There are only {events.shape[0]} instead of 1400 events in the data \n",
               "---------------------------------------------------------------------------------\n", flush=True)
        if events.shape[0] == 1296:
            events,_,_ = fix_events_from_analogue(data_raw, event_info)
            print("\n---------------------------------------------------------------------------------\n",
              "There are some events without triggers in the data! These are probably catch trials. \n",
               "Let's reconstruct the missing triggers...\n",
               "---------------------------------------------------------------------------------\n", flush=True)
    # ------------------------------------------------------------------------------------------------------- ^
    
    # Check if any subID is empty and remove corresponding entries from all event_info keys
    indices_to_remove = [i for i, subid in enumerate(event_info['subID']) if not subid]
    for idx in sorted(indices_to_remove, reverse=True):
        print("\n---------------------------------------------------------------------------------\n",
              f"Found an empty block in the behavioural data!! Removing the empty block from the event_info...\n",
               "---------------------------------------------------------------------------------\n", flush=True)
        for key in event_info:
            event_info[key].pop(idx)

    #reformat information in sensible conditions
    condition = np.concatenate([np.zeros(120) + info for info in event_info['task']])
    stim_type = np.concatenate(event_info['StimuliOrder2'])
    stim_cat = np.concatenate(event_info['StimuliOrder3'])
    response = np.concatenate(event_info['response'])

    print(condition.shape)
    print(stim_type.shape)
    print(stim_cat.shape)
    print(response.shape)

    if np.isnan(response).sum() > 0: #nans are added when a subject didnt manage to respond in time -> is considered a miss
        response[np.isnan(response)] = 0

    #events[:,2] = [int(str(int(cur_c)) + str(int(cur_stim))) for cur_c, cur_stim in zip(condition, stim_type)]
    #events[:,2] = [int(str(int(cur_c)) + str(int(cur_stim)) + str(int(cur_resp))) for cur_c, cur_stim, cur_resp in zip(condition, stim_type, response)]
    events[:,2] = [int(str(int(cur_c)) + str(int(cur_resp)) + str(int(cur_stim))) for cur_c, cur_resp, cur_stim in zip(condition, response, stim_type)]

    # print(f"---Now Doing: events = events[stim_cat == 1]---")
    events = events[stim_cat == 1]

    #create an event dictionary    
    # event_keys = ['auditory/1/hit', 'auditory/2/hit', 'auditory/3/hit', 'auditory/4/hit',
    #               'auditory/1/miss', 'auditory/2/miss', 'auditory/3/miss', 'auditory/4/miss',
    #                'somato/1/hit', 'somato/2/hit', 'somato/3/hit', 'somato/4/hit',
    #                'somato/1/miss', 'somato/2/miss', 'somato/3/miss', 'somato/4/miss',
    #                'visual/1/hit', 'visual/2/hit', 'visual/3/hit', 'visual/4/hit',
    #                'visual/1/miss', 'visual/2/miss', 'visual/3/miss', 'visual/4/miss']
    event_keys = ['auditory/hit/1', 'auditory/hit/2', 'auditory/hit/3', 'auditory/hit/4',
                  'auditory/miss/1', 'auditory/miss/2', 'auditory/miss/3', 'auditory/miss/4',
                   'somato/hit/1', 'somato/hit/2', 'somato/hit/3', 'somato/hit/4',
                   'somato/miss/1', 'somato/miss/2', 'somato/miss/3', 'somato/miss/4',
                   'visual/hit/1', 'visual/hit/2', 'visual/hit/3', 'visual/hit/4',
                   'visual/miss/1', 'visual/miss/2', 'visual/miss/3', 'visual/miss/4']
    
    # event_values = [111,121,131,141,
    #                 110,120,130,140,
    #                 211,221,231,241,
    #                 210,220,230,240,
    #                 311,321,331,341,
    #                 310,320,330,340]

    event_values = [111,112,113,114,
                    101,102,103,104,
                    211,212,213,214,
                    201,202,203,204,
                    311,312,313,314,
                    301,302,303,304]
    # print(f"---Now Doing: event_dict = dict(zip(event_keys, event_values))---")
    event_dict = dict(zip(event_keys, event_values))

    #%% chunk in epochs  
    # print(f"---Now Doing: epochs = mne.Epochs()---")

    epochs = mne.Epochs(data_raw,
                    events=events,
                    event_id=event_dict,
                    **epochs_settings)
    # print(f"---Finished Doing: epochs = mne.Epochs()---")
    #%% TODO: ADD METADATA
    return epochs, events



'''
I added the "sel_condition" argument, so we can choose which condition (Near threshold, High intensity or Catch) we want to select

'''
def get_epochs_R(data_raw, 
               event_info, 
               epochs_settings,
               events = None,
            #    sel_condition = 2 # 1: NT, 2: HI, 3: catch
               ):
    '''
    
    This function can be used to extract events of interest from the ncc experiment and them into epochs.

    task -> contains information whether the block was auditory=1, somatosensory=2 or visual=3
    StimuliOrder2 -> contains information about the stimulus 1,2,3,4
    StimuliOrder3 -> contains information about the stimulus category (should be 1)
    response -> 1,0 probably hit/miss

    '''
  
    
    print(f"---Now Doing: events = read_events_from_analogue(data_raw)---", flush=True)       
    
    # get events
    events = read_events_from_analogue(data_raw) #need to do this as sth in mne is fucked up when it comes to getting events

    print("\n---------------------------------------------------------------------------------\n",
              f"Shape of the events; {events.shape[0]}\n",
               "---------------------------------------------------------------------------------\n", flush=True)
        
    indices_to_remove = [
    i for i, subid in enumerate(event_info['subID'])
        if (
            subid is None
            or (isinstance(subid, (str, bytes)) and subid == "")
            or (isinstance(subid, np.ndarray) and subid.size == 0)
        )
    ]
    
    for idx in sorted(indices_to_remove, reverse=True):
        print("\n---------------------------------------------------------------------------------\n",
              f"Found an empty block in the behavioural data!! Removing the empty block from the event_info...\n",
               "---------------------------------------------------------------------------------\n", flush=True)
        for key in event_info:
            event_info[key].pop(idx)

    #reformat information in sensible conditions
    condition = np.concatenate([np.zeros(120) + info for info in event_info['task']])
    stim_type = np.concatenate(event_info['StimuliOrder2'])
    stim_cat = np.concatenate(event_info['StimuliOrder3'])#NT/HI or catch
    response = np.concatenate(event_info['response'])

    print(condition.shape)
    print(stim_type.shape)
    print(stim_cat.shape)
    print(response.shape)

    if np.isnan(response).sum() > 0: #nans are added when a subject didnt manage to respond in time -> is considered a miss
        response[np.isnan(response)] = 0

    #events[:,2] = [int(str(int(cur_c)) + str(int(cur_stim))) for cur_c, cur_stim in zip(condition, stim_type)]
    #events[:,2] = [int(str(int(cur_c)) + str(int(cur_stim)) + str(int(cur_resp))) for cur_c, cur_stim, cur_resp in zip(condition, stim_type, response)]
    events[:,2] = [int(str(int(cur_cat)) + str(int(cur_c)) + str(int(cur_resp)) + str(int(cur_stim))) for cur_cat, cur_c, cur_resp, cur_stim in zip(stim_cat, condition, response, stim_type)]
    print(f"events in my epochs function: \n\t shape of events: {events.shape}\n\tevents: {events}", flush=True)

    event_keys = ['NT/auditory/hit/1', 'NT/auditory/hit/2', 'NT/auditory/hit/3', 'NT/auditory/hit/4',
                  'NT/auditory/miss/1', 'NT/auditory/miss/2', 'NT/auditory/miss/3', 'NT/auditory/miss/4',
                   'NT/somato/hit/1', 'NT/somato/hit/2', 'NT/somato/hit/3', 'NT/somato/hit/4',
                   'NT/somato/miss/1', 'NT/somato/miss/2', 'NT/somato/miss/3', 'NT/somato/miss/4',
                   'NT/visual/hit/1', 'NT/visual/hit/2', 'NT/visual/hit/3', 'NT/visual/hit/4',
                   'NT/visual/miss/1', 'NT/visual/miss/2', 'NT/visual/miss/3', 'NT/visual/miss/4',

                   'HI/auditory/hit/1', 'HI/auditory/hit/2', 'HI/auditory/hit/3', 'HI/auditory/hit/4',
                  'HI/auditory/miss/1', 'HI/auditory/miss/2', 'HI/auditory/miss/3', 'HI/auditory/miss/4',
                   'HI/somato/hit/1', 'HI/somato/hit/2', 'HI/somato/hit/3', 'HI/somato/hit/4',
                   'HI/somato/miss/1', 'HI/somato/miss/2', 'HI/somato/miss/3', 'HI/somato/miss/4',
                   'HI/visual/hit/1', 'HI/visual/hit/2', 'HI/visual/hit/3', 'HI/visual/hit/4',
                   'HI/visual/miss/1', 'HI/visual/miss/2', 'HI/visual/miss/3', 'HI/visual/miss/4',

                    'catch/auditory/hit/1', 'catch/auditory/hit/2', 'catch/auditory/hit/3', 'catch/auditory/hit/4',
                  'catch/auditory/miss/1', 'catch/auditory/miss/2', 'catch/auditory/miss/3', 'catch/auditory/miss/4',
                   'catch/somato/hit/1', 'catch/somato/hit/2', 'catch/somato/hit/3', 'catch/somato/hit/4',
                   'catch/somato/miss/1', 'catch/somato/miss/2', 'catch/somato/miss/3', 'catch/somato/miss/4',
                   'catch/visual/hit/1', 'catch/visual/hit/2', 'catch/visual/hit/3', 'catch/visual/hit/4',
                   'catch/visual/miss/1', 'catch/visual/miss/2', 'catch/visual/miss/3', 'catch/visual/miss/4',


                   ]
    
    event_values = [1111,1112,1113,1114,
                    1101,1102,1103,1104,
                    1211,1212,1213,1214,
                    1201,1202,1203,1204,
                    1311,1312,1313,1314,
                    1301,1302,1303,1304,

                    2111,2112,2113,2114,
                    2101,2102,2103,2104,
                    2211,2212,2213,2214,
                    2201,2202,2203,2204,
                    2311,2312,2313,2314,
                    2301,2302,2303,2304,

                    3111,3112,3113,3114,
                    3101,3102,3103,3104,
                    3211,3212,3213,3214,
                    3201,3202,3203,3204,
                    3311,3312,3313,3314,
                    3301,3302,3303,3304
                    ]
    

    # print(f"---Now Doing: event_dict = dict(zip(event_keys, event_values))---")
    event_dict = dict(zip(event_keys, event_values,))

    # Remove event_dict entries that don't exist in events[:,2]
    event_dict = {k: v for k, v in event_dict.items() if v in events[:,2]} # <============================================= This is new!


    #%% chunk in epochs  
    # print(f"---Now Doing: epochs = mne.Epochs()---")

    epochs = mne.Epochs(data_raw,
                    events=events,
                    event_id=event_dict,
                    **epochs_settings)
    # print(f"---Finished Doing: epochs = mne.Epochs()---")
    #%% TODO: ADD METADATA
    return epochs, events