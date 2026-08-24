#%%
import os
from pathlib import Path

# inDir = Path("/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/")
inDir = Path("/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean_mean_head_pos/")
OLD = "_meg-epo_clean"
# OLD = "_meg_clean-epo"
NEW = "_clean_meg-epo"


for dirpath, dirnames, filenames in os.walk(inDir):
	for name in filenames:
		if OLD in name:
			old_path = Path(dirpath) / name
			new_name = name.replace(OLD, NEW)
			new_path = Path(dirpath) / new_name
			try:
				old_path.rename(new_path)
				print(f"Renamed: {old_path} -> {new_path}")
			except Exception as e:
				print(f"Failed to rename {old_path}: {e}")



