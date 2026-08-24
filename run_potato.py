#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config2 import * # directories + constants


from clusterjobs.do_potato import SavePotato

from plus_slurm import JobCluster
from utils.submit_jobs import job_setup, auto_args
from utils.subj import get_MEG_raw_subjects
#%%
# all_subjects = get_MEG_raw_subjects()


# ---------------------------- all subjects
# all_subjects = [
# 	'19800616mrgu',
#  '19840930bigs',
#  '19880331igse',
#  '19910703eigl',
#  '19910823ssld',
#  '19920917gbse',
#  '19921205crfi',
#  '19930306sbeh',
#  '19942803fbjm',
#  '19950623ajrn',
#  '19951227eipo',
#  '19960531hibu',
#  '19960628gblm',
#  '19960630cahi',
#  '19961123crsh',
#  '19970302urmr',
#  '19970520smsr',
#  '19970605btre',
#  '19970801cabd',
#  '19971028mrhs',
#  '19980223zlde',
#  '19981005gndd',
#  '19990810mrkh',
#  '20000107ptfu',
#  '20000118sbnb',
#  '20010917rswg',
#  '20020705ttbr',
#  '20021027sldn',
#  '20031022ekse',
#  '20040627vrrj',
#  '20040630gbaf',
#  '20040819knee',
#  '20050204vrao',
#  '20050610atbu',
#  '20070324hlti'
#  ]

# ---------------------------- subjects where I didnt run block-wise ICA
# all_subjects = [
# 	'19800616mrgu',
#  '19840930bigs',
#  '19880331igse',
#  '19920917gbse',
#  '19921205crfi',
#  '19930306sbeh',
#  '19942803fbjm',
#  '19950623ajrn',
#  '19951227eipo',
#  '19960531hibu',
#  '19960628gblm',
#  '19961123crsh',
#  '19970520smsr',
#  '19970605btre',
#  '19970801cabd',
#  '19980223zlde',
#  '19981005gndd',
#  '20000118sbnb',
#  '20010917rswg',
#  '20020705ttbr',
#  '20021027sldn',
#  '20040627vrrj',
#  '20040630gbaf',
#  '20040819knee',
#  '20050204vrao',
#  '20050610atbu',
#  ]

# ---------------------------- subjects where I tried block.wise ICA
all_subjects = [
#  '19910703eigl', # done
#  '19910823ssld', # done
#  '19960630cahi', # done
 '19970302urmr',
 '19971028mrhs',
 '19990810mrkh',
 '20000107ptfu',
#  '20031022ekse',
#  '20070324hlti'
 ]

#%%
job_kwargs = job_setup(ram  = '64G',       
                       cpus = 10,        
                       time = 2*60,       
                    #    qos  = 'high_prio',
                       name = 'potato.sh',
					   jobs_dir='potato'
                       )

job_cluster = JobCluster(**job_kwargs)

job_cluster.add_job(
    SavePotato,
	subjectID = auto_args(all_subjects)

)

job_cluster.submit(do_submit=True)
