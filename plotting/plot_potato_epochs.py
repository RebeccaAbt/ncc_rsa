#%%
import mne
import matplotlib
import os
import joblib
import mne
import numpy as np

import matplotlib.pyplot as plt
mne.viz.set_browser_backend('matplotlib')

#%%

def chunk_list(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

#%% 
inFolder = 'epochs_potato_old2'
subjectID = '19840930bigs'
potatoFile = f'/home/reabt/experiments/ncc/MEG/data/{inFolder}/{subjectID}/{subjectID}_maxfilter_True__ica_True__l_pass_30__downsample_f_1000__h_pass_1__-1.0-1.0s_potato_info.pkl'
epochsFile_clean = f'/home/reabt/experiments/ncc/MEG/data/{inFolder}/{subjectID}/{subjectID}_maxfilter_True__ica_True__l_pass_30__downsample_f_1000__h_pass_1__-1.0-1.0s__useField_True_meg-epo_clean.fif'
epochsFile_unclean = f'/home/reabt/experiments/ncc/MEG/data/{inFolder}/{subjectID}/{subjectID}_maxfilter_True__ica_True__l_pass_30__downsample_f_1000__h_pass_1__-1.0-1.0s_meg-epo.fif'
potato = joblib.load(potatoFile)

inFolder = 'epochs_potato'
potatoFile_8 = f'/home/reabt/experiments/ncc/MEG/data/{inFolder}/{subjectID}/{subjectID}_MF__ICA__filter_0_1-30__fs_1000__-1.0-1.0s_potato4mag__estim_oas__thres_z3_potato_info.pkl'
potato_8 = joblib.load(potatoFile_8)
# epochs = mne.read_epochs(epochsFile)


#%%
tmp_potato = potato_8
[print(f'{idx}:\t{tmp_potato['z_scores'][idx]}\t{round(tmp_potato['clean_probability'][idx],6):f}') for idx in tmp_potato['bad_idx']]

#%%
epochs = mne.read_epochs(epochsFile_unclean, preload=True)
epochs_clean = mne.read_epochs(epochsFile_clean, preload=False)

#%% count events

event_id = epochs.event_id
unique_events, counts = np.unique(epochs.events[:,-1], return_counts=True)
event_counts = dict([[u.item(), c.item()] for u, c in zip(unique_events, counts)])
event_matched = {k: event_counts[v] for k, v in event_id.items()}

event_id2 = epochs_clean.event_id
unique_events2, counts2 = np.unique(epochs_clean.events[:,-1], return_counts=True)
event_counts2 = dict([[u.item(), c.item()] for u, c in zip(unique_events2, counts2)])
event_matched2 = {k: event_counts2[v] for k, v in event_id2.items()}
#%% show differences in event counts

event_diff = {k: v1-v2 for (k, v1), v2 in zip(event_matched.items(), event_matched2.values())}

#%% 
# get indices of bad epochs
bad_idx_1 = [i for i, v in enumerate(epochs_clean.drop_log) if v]

# from log output:
bad_idx_2 = [5, 22, 26, 27, 40, 43, 53, 96, 101, 112, 114, 125, 126, 140, 142, 143, 144, 156, 158, 202, 212, 215, 216, 228, 230, 240, 247, 250, 259, 276, 301, 303, 314, 315, 366, 368, 376, 385, 389, 390, 391, 394, 399, 401, 418, 424, 440, 446, 450, 453, 455, 462, 472, 479, 480, 482, 486, 543, 556, 585, 586, 587, 588, 589, 597, 640, 642, 680, 699, 719, 720, 729, 749, 768, 769, 796, 800, 841, 859, 861, 864, 866, 872, 874, 878, 889, 897, 905, 911, 915, 916, 938, 939, 942, 945, 952, 954, 1041, 1053, 1079, 1087, 1088, 1097, 1099, 1110, 1111, 1151, 1156, 1169, 1201, 1210, 1213, 1237, 1284, 1326, 1329, 1334, 1342, 1347, 1387, 1391, 1397, 1400, 1401, 1402, 1406] # 

#from job folder 008
bad_idx_8 = [26, 216, 228, 276, 389, 399, 462, 479, 585, 586, 587, 589, 699, 719, 769, 861, 864, 866, 872, 874, 889, 912, 915, 916, 938, 939, 952, 1079, 1087, 1097, 1111, 1151, 1210, 1213, 1342, 1397, 1401]

bad_idx_9 = [ 26, 96, 142, 276, 303, 389, 462, 585, 586, 587, 589, 719, 769, 861, 864, 866, 872, 874, 878, 889, 915, 916, 938, 952, 1079, 1087, 1097, 1111, 1151, 1210, 1213, 1237, 1284, 1397, 1401]

bad_idx_12 = [26, 53, 276, 389, 462, 479, 589, 719, 769, 864, 866, 872, 874, 889, 915, 916, 1079, 1151, 1347, 1397, 1401]

#z =2
bad_idx_10 = [5, 21, 22, 26, 27, 29, 30, 40, 42, 43, 53, 96, 99, 112, 115, 125, 142, 155, 156, 157, 158, 199, 202, 205, 211, 212, 215, 216, 228, 230, 231, 247, 259, 276, 285, 288, 303, 312, 314, 315, 334, 340, 350, 353, 354, 359, 366, 368, 376, 378, 385, 389, 390, 392, 393, 394, 395, 399, 401, 403, 418, 424, 426, 440, 446, 453, 455, 462, 463, 472, 479, 480, 481, 483, 486, 487, 514, 531, 543, 546, 555, 556, 558, 560, 567, 581, 585, 586, 587, 588, 589, 597, 599, 603, 611, 621, 624, 640, 642, 643, 646, 660, 670, 674, 678, 679, 690, 699, 703, 717, 719, 720, 729, 748, 749, 769, 780, 796, 823, 841, 843, 859, 861, 864, 866, 870, 872, 873, 874, 878, 885, 889, 905, 912, 914, 915, 916, 921, 930, 938, 939, 942, 943, 945, 951, 952, 954, 960, 971, 972, 999, 1002, 1003, 1030, 1041, 1052, 1079, 1086, 1087, 1088, 1089, 1097, 1099, 1108, 1110, 1111, 1128, 1134, 1141, 1149, 1151, 1152, 1156, 1169, 1172, 1180, 1183, 1190, 1191, 1197, 1199, 1201, 1210, 1213, 1216, 1217, 1225, 1234, 1235, 1236, 1237, 1273, 1278, 1284, 1326, 1327, 1329, 1331, 1334, 1336, 1337, 1342, 1347, 1351, 1358, 1360, 1387, 1389, 1391, 1397, 1400, 1401, 1402, 1406, 1416, 1423, 1424]
bad_idx_11 = [1, 5, 6, 21, 22, 26, 27, 29, 30, 40, 42, 43, 44, 47, 53, 60, 61, 75, 96, 99, 101, 103, 112, 114, 115, 116, 125, 126, 137, 139, 140, 142, 143, 144, 155, 156, 157, 158, 170, 178, 188, 190, 199, 200, 202, 205, 211, 212, 215, 216, 224, 228, 229, 230, 231, 236, 247, 248, 250, 255, 259, 276, 285, 288, 289, 301, 303, 312, 313, 314, 315, 316, 317, 325, 334, 340, 350, 353, 354, 359, 366, 368, 370, 376, 378, 385, 389, 390, 391, 392, 393, 394, 395, 398, 399, 400, 401, 403, 418, 424, 426, 440, 445, 446, 450, 453, 455, 462, 463, 472, 475, 479, 480, 481, 482, 483, 486, 487, 498, 514, 531, 542, 543, 546, 555, 556, 558, 560, 566, 567, 581, 585, 586, 587, 588, 589, 593, 597, 599, 603, 611, 613, 621, 622, 624, 640, 642, 643, 646, 670, 678, 680, 690, 698, 699, 703, 717, 718, 719, 720, 721, 729, 734, 749, 750, 768, 769, 779, 780, 789, 796, 800, 815, 819, 823, 837, 841, 843, 847, 849, 859, 861, 864, 866, 870, 872, 873, 874, 877, 878, 885, 889, 897, 905, 911, 912, 914, 915, 916, 921, 922, 930, 938, 939, 942, 943, 945, 951, 952, 954, 957, 960, 969, 971, 997, 999, 1002, 1003, 1005, 1030, 1032, 1041, 1052, 1053, 1079, 1083, 1086, 1087, 1088, 1089, 1091, 1097, 1099, 1108, 1110, 1111, 1128, 1134, 1140, 1141, 1146, 1149, 1151, 1152, 1156, 1167, 1169, 1172, 1180, 1183, 1190, 1197, 1199, 1201, 1210, 1213, 1214, 1217, 1225, 1234, 1235, 1236, 1237, 1271, 1273, 1278, 1284, 1315, 1326, 1327, 1329, 1331, 1334, 1336, 1342, 1347, 1351, 1358, 1360, 1387, 1389, 1391, 1397, 1400, 1401, 1402, 1406, 1416, 1418, 1423]

#%%
# 9 --> mags, 13= grads
bad_idx_13 = [53, 96, 143, 250, 303, 366, 368, 376, 385, 418, 462, 486, 589, 719, 720, 749, 769, 864, 866, 872, 874, 878, 889, 911, 915, 916, 945, 1053, 1284, 1347, 1397, 1401]

#%% plot bad wpochs
for i in bad_idx_1[0:1]:
    epochs[i].load_data().pick('meg').plot(n_epochs = 1,butterfly=True, 
                                                show_scrollbars=False, 
                                                show=False, 
                                                show_scalebars=False
                                                )
    plt.tight_layout()
    plt.show()


#%%
for i in bad_idx_1[0:10]:
    epochs[i:i+20].load_data().pick('meg').plot(n_epochs = 10,butterfly=True, 
                                                show_scrollbars=False, 
                                                show=False, 
                                                show_scalebars=False
                                                )
    plt.tight_layout()
    plt.show()

#%% ===============================================================plot chunked bad epochs
onlyBad_2_1 = np.sort(list(set(bad_idx_2).difference(set(bad_idx_1))))
onlyBad_1_2 = np.sort(list(set(bad_idx_1).difference(set(bad_idx_2))))
onlyBad_8_1 = np.sort(list(set(bad_idx_8).difference(set(bad_idx_1))))
onlyBad_8_2 = np.sort(list(set(bad_idx_8).difference(set(bad_idx_2))))
onlyBad_1_8 = np.sort(list(set(bad_idx_1).difference(set(bad_idx_8))))
onlyBad_2_8 = np.sort(list(set(bad_idx_2).difference(set(bad_idx_8))))
onlyBad_9_8 = np.sort(list(set(bad_idx_9).difference(set(bad_idx_8))))
onlyBad_8_9 = np.sort(list(set(bad_idx_8).difference(set(bad_idx_9))))
# onlyBad_1_8 = np.sort(list(set(bad_idx_1).difference(set(bad_idx_8))))
# onlyBad_2_8 = np.sort(list(set(bad_idx_2).difference(set(bad_idx_8))))

bad_merged_8_9 = sorted(set(bad_idx_8) | set(bad_idx_9))
bad_merged_9_13 = sorted(set(bad_idx_9) | set(bad_idx_13))

print(bad_merged_8_9)


#%%
combined_1_2 = sorted((set(bad_idx_1) & set(bad_idx_2)))

combined_10_11 = sorted((set(bad_idx_10) & set(bad_idx_11)))



# for idx in chunk_list(np.sort(list(set(bad_idx_1).difference(set(bad_merged_8_9)))), 8)[0:5]: # in list 1 but neither in list 1 nor 2

# for idx in chunk_list(np.sort(list(set(bad_idx_9).difference(set(combined_1_2)))), 8)[0:5]: # in bad_idx_9 but not in combi of 1 and 2
# for idx in chunk_list(bad_merged_9_13, 8)[0:5]:
for idx in chunk_list(np.sort(list(set(bad_idx_1).difference(set(bad_merged_9_13)))), 8)[0:5]: # in list 1 but neither in list 1 nor 2


    # fig = epochs[idx].load_data().pick('meg').plot(n_epochs = 10,butterfly=True,
    #                                             show_scrollbars=False, 
    #                                             show=False, 
    #                                             show_scalebars=False
    #                                             )
    fig = epochs[idx].pick('meg').plot(n_epochs = 10,butterfly=True,
                                                show_scrollbars=False, 
                                                show=False, 
                                                show_scalebars=False
                                                )
    fig.set_figheight(2.5)
    fig.set_figwidth(15)

    ax = fig.axes[0]
    ax.set_xlabel('')

    for l in ax.findobj(matplotlib.lines.Line2D):
        c = l.get_color()
        if isinstance(c, np.ndarray):
            if  np.isclose(c[2], 0.54509804):
                l.set_ydata(np.array([x+0.3 for x in l.get_ydata()]))
            elif  np.isclose(c[2], 1):
                l.set_ydata(np.array([x-0.3 for x in l.get_ydata()]))

    ax.set_yticks([0.3, 0.7])
    ax.set_yticklabels(['mag', 'grad'])
    ax.set_ylim(1, 0)

#%% show values of potato_o z scores and probabilities for epochs that were rejected in other setups

tmp_potato = potato_9
tmp_idx = onlyBad_1_9
[print(f'{idx}:\t{tmp_potato['z_scores'][idx]}\t{round(tmp_potato['clean_probability'][idx],6):f}') for idx in tmp_idx]
