#%%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants
import hashlib
import plus_sync.config
from utils.subj import *



#%%

subjects = get_MRI_subjects(remove_bad=False)

#%%
def hash_string(value: str, project_name: str | None = None) -> str:
    if project_name is None:
        config = plus_sync.config.Config.from_cmdargs()
        project_name = config.project_name

    h = hashlib.sha256()
    h.update(value.encode())
    h.update(project_name.encode())
    return h.hexdigest()[:12]



inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MRI/aw_ncc/'


def replace_hashed_in_path(root_dir: str, hashed_id: str, subj: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # print(dirpath)
        # print(f'dirnames: {dirnames}')
        for name in filenames:
            if hashed_id in name:
                old_path = os.path.join(dirpath, name)
                new_path = os.path.join(dirpath, name.replace(hashed_id, subj))
                print(f"old path: {old_path}\n\t new path: {new_path}")
                os.rename(old_path, new_path)
        if len(dirnames)>0:
            for name in dirnames:
                if hashed_id in name:
                    old_path = os.path.join(dirpath, name)
                    new_path = os.path.join(dirpath, name.replace(hashed_id, subj))
                    os.rename(old_path, new_path)
        else:
            print(f'{subj} ({hashed_id}) not in BIDS data')
#%%%
subjects = ['sub-19991211mrbn']
for subj in subjects:
    hashed_ID = hash_string(subj, 'aw_ncc')
    replace_hashed_in_path('/home/scc_e_393956/ncc/rsa/outputs_MRI/bids/fmriprep', hashed_ID, subj)

#%%
# 19930306sbeh, 19970801cabd
for subj in ['19991211mrbn']:
    hashed_ID = hash_string(subj, 'aw_ncc')
    print(f"{subj} --> {hashed_ID}")


#%%

hashed_ID = hash_string('19930306sbpe', 'aw_ncc')
print(f"{'19930306sbpe'} --> {hashed_ID}")
