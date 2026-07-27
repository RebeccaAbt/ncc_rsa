
#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import * # directories + constants

from utils.subj import *
from utils.submit_jobs import auto_args, job_setup

#%%

all_subjects = get_MRI_subjects()
#%%

for subjectID in all_subjects:

	thisConfig = 'MRIconfig_C5'
	directory = f"{MRI_RSA_DIR}/{thisConfig}/{subjectID}/"

	for file in directory.glob("*_SL_rdms.pkl"):
		print(file.name)
		if "_corr_cov" not in file.name:
			continue

		new_name = file.name.replace("_corr_cov", "")
		new_file = file.with_name(new_name)

		print(f"Renaming:\n  {file.name}\n→ {new_file.name}")
		file.rename(new_file)

	thisConfig = 'MRIconfig_C2'
	directory = f"{MRI_RSA_DIR}/{thisConfig}/{subjectID}/"

	for file in directory.glob("*_SL_rdms.pkl"):
		print(file.name)
		if "_cosine" not in file.name:
			continue

		new_name = file.name.replace("_cosine", "")
		new_file = file.with_name(new_name)

		print(f"Renaming:\n  {file.name}\n→ {new_file.name}")
		file.rename(new_file)


#%%

for subjectID in all_subjects:

	thisConfig = 'MRIconfig_C5'
	directory = f"{MRI_RSA_DIR}/{thisConfig}/{subjectID}/"

	for file in directory.glob("*_info.pkl"):
		print(file.name)
		if "_corr_cov" not in file.name:
			continue

		new_name = file.name.replace("_corr_cov", "_cosine_cov")
		new_file = file.with_name(new_name)

		print(f"Renaming:\n  {file.name}\n→ {new_file.name}")
		file.rename(new_file)

	thisConfig = 'MRIconfig_C2'
	directory = f"{MRI_RSA_DIR}/{thisConfig}/{subjectID}/"

	for file in directory.glob("*_info.pkl"):
		print(file.name)
		if "_cosine" not in file.name:
			continue

		new_name = file.name.replace("_cosine", "_cosine_cov")
		new_file = file.with_name(new_name)

		print(f"Renaming:\n  {file.name}\n→ {new_file.name}")
		file.rename(new_file)
