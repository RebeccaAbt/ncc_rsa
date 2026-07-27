'''
Alternative to obob-mne "read_events_from_analogue". 
I messed up the experimental code which is why the triggers for the catch trials of most of the new subjects are missing.
With the funcion below we are able to reconstruct the triggers based on the existing event data + data from the matlab log files.
At the moment, the function requires that there was a button press in the missing catch trial 
(because we use the button press afterwards + postStim time for reconstruction).
 If there was no Button press in any of the subjects, we will have an error here. We can thenn add anothere condition that computes the 
 missing trigger based on the previous trial. But this is not implemented yert
'''

import numpy as np
import mne
from copy import deepcopy

def _get_events(raw, trigger_channels=None, trigger_type = 'stimulus', tolerance=1):

    if trigger_channels is None:
        if trigger_type == 'button':
            trigger_channels = ['STI015', 'STI016']
        else:
            trigger_channels = ['STI001', 'STI002', 'STI003', 'STI004', 'STI005', 'STI006', 'STI007', 'STI008']

    trigger_data = raw[mne.pick_channels(raw.ch_names, trigger_channels)][0] / 5  # noqa

    bit_mult = 2 ** np.arange(0, trigger_data.shape[0])
    trigger_values = np.sum((trigger_data.T * bit_mult).T, axis=0).astype(int)
    trigger_idx = np.where(np.diff(trigger_values) > 0)[0] + 1
    tmp_events = np.zeros((trigger_idx.shape[0], 4))
    tmp_events[:, 0] = trigger_idx + raw.first_samp # <---------------------- we need to add this again later!!!
    tmp_events[:, 2] = trigger_values[trigger_idx]
    if trigger_type == 'stimulus':
        tmp_events[:, 3] = trigger_values[trigger_idx]*100 # so we can distinguish stimulus events from button presses
    elif trigger_type == 'button':
        tmp_events[:, 3] = trigger_values[trigger_idx]

    bad_triggers_idx = np.where(np.diff(tmp_events[:, 0]) <= tolerance)[0]

    assert bad_triggers_idx.size == 0, "Bad trigger indices found! But I don't know what that means so we need to check this!"

    tmp_events[bad_triggers_idx + 1, 0] = tmp_events[bad_triggers_idx, 0]

    trigger_events = np.delete(tmp_events, bad_triggers_idx, axis=0)
    return trigger_events

def _get_last_button_idx(trigger_events, button_events, event_info):
    '''
    Docstring for _get_last_button_idx
    
    :param trigger_events: List with length n_blocks --> trigger_events already separated by block
    :param button_events: Button events of al blocks concatenated
    '''
    last_triggers = [b[-1,0] for b in trigger_events]

    print(f"last_triggers: {last_triggers}")

    block_end_idx = []
    for b, sample in enumerate(last_triggers[:-1]):
        if event_info['StimuliOrder3'][0][-1] != 3: # if last trial was not a catch trial and hence did have a trigger
            print(f"Block {b}: using first button press after last trigger")
            block_end_idx.append(np.where(button_events[:,0] > sample)[0][0])
        elif event_info['StimuliOrder3'][0][-2] != 3: 
            print(f"Block {b}: using second button press after last trigger")
            block_end_idx.append(np.where(button_events[:,0] > sample)[0][1])
        else:
            assert not event_info['StimuliOrder3'][b][-1] == 3 & event_info['StimuliOrder3'][b][-2] == 3, "Last 2 trials are catch trials. We need to refine how the last button press is defined!"
                
    print(f"\nblock_end_idx: \n\n {block_end_idx}\n")   

    print(f"Button samples at block_end_idx: \n\n {button_events[block_end_idx,0].astype(int)}") 
    return np.array(block_end_idx)

def _get_button_values(event_info, block=None):
    if block==None:
        buttons = []
        for block, (button_order,response) in enumerate(zip(event_info['ResponseOrder'],event_info['response'])):
            buttons.append(_get_button_values(event_info, block))

    else:   
        button_order = event_info['ResponseOrder'][block]
        response = event_info['response'][block]
        buttons = []

        for tr, (res, ord) in enumerate(zip(response, button_order)):
            if np.isnan(res): # nan in "button"
                buttons.append(np.nan)
            elif ord == 1:
                buttons.append(res.astype(int)+1)
            else:
                buttons.append((res.astype(int)-ord.astype(int))*(-1))
    return buttons

def _map_events_per_block(events, block_idx=None):
    event_blocks = []

    if block_idx is None:
        block_idx = np.where(np.diff(events[:,2])!=0)[0]

    for idx in  np.arange(0, block_idx.shape[0] + 1 ):
        # print(idx)
        if idx == 0:
            event_blocks.append(events[0:block_idx[idx]+1,:])
        elif idx == block_idx.shape[0]:
            event_blocks.append(events[block_idx[idx-1]+1:,])
        else:
            event_blocks.append(events[block_idx[idx-1]+1:block_idx[idx]+1,:])
    return event_blocks

def get_trigger_intervals_info(event_info, irun):
# data: behav data
# trigger-trigger diff = from stim-onset of previous trial to stim-onset of current trial
    n_trials = event_info['trials'][0].shape[0]
    trigger_intervals = []

    for trial in np.arange(0, n_trials-1):
        trial_interval = (
            event_info['time'][irun]['Stimulus'][trial]+
            event_info['time'][irun]['PostStim'][trial]+
            event_info['time'][irun]['response'][trial]+
            event_info['time'][irun]['ITI'][trial+1]+
            event_info['time'][irun]['Fixation'][trial+1]
            )*1000+2 # 2 is an adjustment value
        trigger_intervals.append(np.round(trial_interval))
    return np.array(trigger_intervals).astype(int)


def _fix_events(raw, event_info, stim_events, button_events, buttons, block):
    all_catch_trials = np.where(event_info['StimuliOrder3'][block]==3)[0]
    
    trigger_intervals = get_trigger_intervals_info(event_info, block)

    assert np.unique(stim_events[:,2])[0].size == 1, "Was expecting only to found one trigger value, but there are multiple...This is a problem!"
    dummy_trigger = stim_events[0].astype(int)
    trigger_value = dummy_trigger[-1]
    n_trials = event_info['StimuliOrder3'][0].shape[0]
    # print(f"n_trials: {n_trials}")
    for c, this_catch in enumerate(all_catch_trials):

        if this_catch == n_trials-1:  # use previous trigger if we need to reconstruct last trigger 
            print(f"this catch {this_catch} is this_catch == n_trials-1")
            previous_trigger_sample = stim_events[this_catch-1][0] 
            missingTrigger_sample = previous_trigger_sample + trigger_intervals[this_catch-1]
            
        elif this_catch < n_trials-1 and event_info['StimuliOrder3'][block][this_catch+1] == 3:
            previous_trigger_sample = stim_events[this_catch-1][0] 
            missingTrigger_sample = previous_trigger_sample + trigger_intervals[this_catch-1]

        else:  
            consecut_trigger_sample = stim_events[this_catch][0] 
            missingTrigger_sample = consecut_trigger_sample - trigger_intervals[this_catch]   
            
        new_trigger = deepcopy(dummy_trigger)
        new_trigger[0] = missingTrigger_sample
        stim_events = np.vstack((stim_events, new_trigger))
        stim_events = stim_events[stim_events[:, 0].argsort()]

    all_events = np.vstack((stim_events, button_events))
    all_events = np.array([e.astype(int) for e in all_events[all_events[:, 0].argsort()]])

    stim_events = all_events[all_events[:, -1] == trigger_value, :3].copy()
    # all_triggers[:, 0] = all_triggers[:, 0] + raw.first_samp
    return stim_events, all_events

def fix_events_from_analogue(raw, event_info, block=None):
    '''
    This functions expects the rata to contain
        - either data from only one run (that can only contain one trigger value for stimuli!! )
        - Or data from concatenated runs, where each run is one condition of multiple conditions. 
          Triggers for different conditions should have different values.

    I used the function to reconstruct the Triggers that were missing in the "catch" trials of our NCC experiment.
    THe data was blocked. 12 runs --> 12 blocks. Order of conditions was regularly 
    (eg. auditory, visual, tactile, auditory, visual, tactile, auditory, visual, tactile, auditory, visual, tactile)
    
    :param raw: Description
    :param event_info: Description
    :param block: Description
    '''

    trigger_events = _get_events(raw, trigger_type = 'stimulus')
    button_events = _get_events(raw, trigger_type = 'button' )

    buttons = _get_button_values(event_info)

    if np.unique(trigger_events[:,2]).shape[0] > 1:
        print("Doing everythin block-wise!")
        trigger_event_list = _map_events_per_block(trigger_events) # to do: delete the block_idx stuff!!

        block_end_idx = _get_last_button_idx(trigger_event_list, button_events, event_info) # idx of last button pres in block
        button_event_list = _map_events_per_block(button_events, block_end_idx) # devide data per blocks
  
        blocks = np.arange(0, len(trigger_events))

        all_triggers_list = []
        
        for t_event, b_event, b_value, block in zip(trigger_event_list, button_event_list, buttons, blocks):
            stim_events, all_events = _fix_events(raw, event_info, t_event, b_event, b_value, block)   
            all_triggers_list.append(stim_events)
        all_triggers = np.vstack([b for b in all_triggers_list])

    else:
        assert block!=None, f"We found only one trigger type in the data. This probably means that the input was the data from one run/block. Therefore you need to specifyx the 'block' argument so we can select the correct data from the event_info variable"
        all_triggers, all_events = _fix_events(raw, event_info, trigger_events, button_events, buttons, block)

    return all_triggers, all_triggers_list, all_events