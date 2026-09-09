#!/bin/bash
#SBATCH --job-name=firstlevel
#SBATCH --time=05:30:00 
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --export=ALL
#SBATCH --array=0-21
#SBATCH --output=/home/scc_e_393956/ncc/rsa/jobs/firstLevel/job_output_%A_%a.out

#---------------------------------
# config files for contrasts and conditions
#---------------------------------
CONFIG_FIRSTLEVEL="/home/scc_e_393956/ncc/rsa/configs/config_firstLevel.json"
CONFIG_CONTRASTS="/home/scc_e_393956/ncc/rsa/configs/config_contrasts.json"
DATADIR="/home/scc_e_393956/ncc/rsa/outputs_MRI/bids/fmriprep/"
# ------------------------------------------------------------------
# determine subject for this array job
# ------------------------------------------------------------------

# subjects=$(find $DATADIR -maxdepth 1 -type d -printf "%f\n" | grep -e 'sub-*')
mapfile -t subjects < <(find $DATADIR -maxdepth 1 -type d -printf "%f\n" | grep -e 'sub-*')

count=0
for item in $subjects; do
  echo $item
  ((count++))
done
echo "Total number of items: $count"


subject=${subjects[$SLURM_ARRAY_TASK_ID]}
# subject=$"sub-19840930bigs"

echo "subject: $subject"

echo "Processing ${subject}"

# ------------------------------------------------------------------
# run first level for 1...6 runs
# ------------------------------------------------------------------

echo "--------------------------------------"
echo "Subject      : $subject"
echo "configFile1  : $CONFIG_FIRSTLEVEL"
echo "configFile2  : $CONFIG_CONTRASTS"
echo "--------------------------------------"


matlab-r2026a -nodisplay -nosplash -softwareopengl -singleCompThread -nodesktop -batch "addpath('/home/scc_e_393956/ncc/rsa/matlab/clusterjobs/'); C_firstLevel_bids('${subject}','${CONFIG_FIRSTLEVEL}','${CONFIG_CONTRASTS}')"