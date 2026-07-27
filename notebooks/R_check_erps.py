#%% imports
from pathlib import Path
import joblib

import seaborn as sns
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style('ticks')
sns.set_context('talk')

#============================================================================== ORIGINAL
'''
#%% set path & load
indir = '/home/schmidtfa/experiments/ncc/data/preproc_new_hdm'
files2load = list(Path(indir).glob('*/*_downsample_f_100__h_pass_1.dat'))

#%%

all_subs = []

for file in files2load:
    cur_epos = joblib.load(file)['epochs_meg']
    cur_sid = str(file).split('/')[-2]
    print(cur_sid)

    ga_evoked = cur_epos['hit'].get_data().mean(axis=0)
    snr_hit = (ga_evoked / cur_epos['hit'].get_data().std(axis=0)).mean(axis=0)

    ga_evoked = cur_epos['miss'].get_data().mean(axis=0)
    snr_miss = (ga_evoked / cur_epos['miss'].get_data().std(axis=0)).mean(axis=0)

    all_subs.append(pd.DataFrame({'snr_hit': snr_hit,
                                  'snr_miss': snr_miss,
                                  'times': cur_epos.times,
                                  'subject_id': cur_sid}))
#%% pick a subject
df_snr = pd.concat(all_subs).query('times > 0.1').query('times < .5').groupby('subject_id').mean().reset_index()

# %% pick erf presets
df_cmb = pd.read_csv('pre_cmb_alpha_snr.csv').merge(df_snr, on='subject_id')
# %%
df_cmb.groupby('hit_miss').corr(numeric_only=True)


'''
# %%
#============================================================================== My stuff

#%% set path & load
indir = '/home/reabt/experiments/ncc/MEG/data/preproc'
# subj_pat = ['20010917rswg', # NEUE Subjects
# # '19880331igse', # fehlt noch
# '19840930bigs',
# '19921205crfi',
# '19961123crsh',
# '20000118sbnb',
# '20070324hlti',
# '20050610atbu',
# '20050204vrao',
# #'20040630gbaf',  # fehlt noch
# #'20050615buea',  # bad data
# '19930306sbeh',
# '20021027sldn',
# # '19940328fbjm', # falsch benannt!!!
# '19942803fbjm',
# '19970801cabd'
# ]

subj_pat = [ # ALTE subjects (zumindest ein apar davon )
    # '20010917rswg', # new
#  '20050204vrao',
 '19960630cahi',
 '19970302urmr',
#  '20000118sbnb',
#  '20050610atbu',
 '20070324hlti',
#  '19930306sbeh',
#  '19840930bigs',
 '20040627vrrj',
 '19970520smsr',
#  '19961123crsh',
 '19951227eipo',
#  '19921205crfi',
 '19910823ssld',
 '19960628gblm',
#  '19970801cabd',
#  '19942803fbjm',
 '19910703eigl',
#  '20021027sldn'
 ]



files2load = [list(Path(indir).glob(f'{subj}/*_downsample_f_100__h_pass_1.dat'))[0] for subj in subj_pat]

all_subs = []

for file in files2load:
    
    print(f'\nLoading file: {file}\n')

    cur_epos = joblib.load(file)['epochs_meg']
    cur_sid = str(file).split('/')[-2]
    subj_df = pd.DataFrame(pd.DataFrame({ 'times': cur_epos.times,
                                  'subject_id': cur_sid}))

    for c in ['auditory', 'somato', 'visual']:
        for p in ['hit', 'miss']:
            condition = "".join([c, '/', p])
            data = cur_epos[condition].get_data().mean(axis=0)
            data_avg = data.mean(axis=0)
            snr = (data_avg / data.std(axis=0)).mean(axis=0)

            subj_df = pd.concat([subj_df, pd.DataFrame({
                f"avg_{condition.replace('/', '_')}": data_avg,
                f"snr_{condition.replace('/', '_')}": snr})], axis = 1)

    for condition in ['hit', 'miss']:
        print(condition)
        data = cur_epos[condition].get_data().mean(axis=0)
        data_avg = data.mean(axis=0)
        snr = (data_avg / data.std(axis=0)).mean(axis=0)

        subj_df = pd.concat([subj_df, pd.DataFrame({
            f"avg_{condition}": data_avg,
            f"snr_{condition}": snr})], axis = 1)
            
    all_subs.append(subj_df)


joblib.dump(subj_df, 'tmp_data4erp_9old_subj.pkl')

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_all = pd.concat(all_subs, ignore_index=True)

modalities = ["auditory", "somato", "visual"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, modality in zip(axes, modalities):
    conditions = [f"{modality}_hit", f"{modality}_miss"]
    value_cols = [f"avg_{c}" for c in conditions]

    df_long = df_all.melt(
        id_vars=["times", "subject_id"],
        value_vars=value_cols,
        var_name="condition",
        value_name="amplitude"
    )

    df_long["condition"] = df_long["condition"].str.replace("^avg_", "", regex=True)

    df_summary = (
        df_long
        .groupby(["times", "condition"])["amplitude"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    df_summary["sem"] = df_summary["std"] / np.sqrt(df_summary["count"])

    for cond in conditions:
        cur = df_summary[df_summary["condition"] == cond]
        ax.plot(cur["times"], cur["mean"], label=cond)
        ax.fill_between(
            cur["times"],
            cur["mean"] - cur["sem"],
            cur["mean"] + cur["sem"],
            alpha=0.2
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(modality)
    ax.set_xlabel("Time (s)")

axes[0].set_ylabel("Amplitude")
axes[-1].legend()
plt.tight_layout()
plt.show()
# %%
