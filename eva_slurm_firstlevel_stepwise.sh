#!/bin/bash
#
#SBATCH --job-name=thresholding
#SBATCH --time=05:30:00 
#SBATCH --cpus-per-task=2
#SBATCH --mem=3GB
#SBATCH --export=ALL
#SBATCH --array=0-26
#SBATCH --output=/home/esteckermeier/data/anclang/output/first_level/tedana/subjectwise_fine-contrasts_stimcorrection/log/job_output_%A_%a.out

#---------------------------------
# file determinining the input
#---------------------------------
INPUT_TSV="/mnt/ceph/groups_hdd/SCCGroup/neurocognition_lab/ANCLANG/output/firstlevel_stepwise/firstlevel_stepwise_inputfilelist.tsv"

#---------------------------------
# determine outputbase
#---------------------------------
OUTPUT_BASE="/mnt/ceph/groups_hdd/SCCGroup/neurocognition_lab/ANCLANG/output/firstlevel_stepwise"

# ------------------------------------------------------------------
# determine subject for this array job
# ------------------------------------------------------------------

subjects=($(tail -n +2 "$INPUT_TSV" | cut -f1 | uniq))

subject=${subjects[$SLURM_ARRAY_TASK_ID]}

echo "Processing ${subject}"

# ------------------------------------------------------------------
# run first level for 1...8 runs
# ------------------------------------------------------------------

for nruns in {1..6}; do
	
	# analysis specific output subdir
	OUTDIR="${OUTPUT_BASE}/nruns-${nruns}/${subject}"

    row=$(awk -F'\t' -v sub="$subject" -v nr="$nruns" \
        '$1==sub && $2==nr {print; exit}' "$INPUT_TSV")

    files=$(echo "$row" | cut -f3)
    events=$(echo "$row" | cut -f4)
    confounds=$(echo "$row" | cut -f5)

    echo "--------------------------------------"
    echo "Subject   : $subject"
    echo "Runs      : $nruns"
    echo "Files     : $files"
    echo "Events    : $events"
    echo "Confounds : $confounds"

	matlab-r2026a -nodisplay -nosplash -softwareopengl -singleCompThread -batch "addpath('/mnt/ceph/groups_hdd/SCCGroup/neurocognition_lab/ANCLANG/anclang-doku/scripts/'); run_firstlevel('${subject}','${files}','${eventfiles}','${confoundfiles}','${output_dir}')"

done