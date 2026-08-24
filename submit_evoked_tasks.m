clear all

myDir = '/home/scc_e_393956/ncc/rsa';
cd(myDir)
addpath('/home/scc_e_393956/ncc/rsa/clusterjobs_mat/')

cluster = parcluster('SCC');
cluster.JobStorageLocation = '/home/scc_e_393956/ncc/rsa/jobs/matlab';

cluster.NumWorkers = 4;
cluster.NumThreads = 8;
%%
cluster.AdditionalProperties.AdditionalSubmitArgs = '--mem=64G';

options.inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/';
options.pattern = 'maxfilter_True__ica_True__0.5-20Hz__fs_100__[-4_4]s_detrend_None_clean_meg-epo';
options.outDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/matlab_evoked/';

preproc_cfg = struct();
preproc_cfg.detrend         = 'no';
preproc_cfg.polyremoval     = 'no';
% preproc_cfg.rectify = 'yes';
% preproc_cfg.demean          = 'yes';
% preproc_cfg.baselinewindow  = [-0.2 0];

%{
------------------------------------------- R E A D !
ACHTUNG!
in der scc_evoked function habe ich das ft_combine planar auskommentiert.
Außerdem ist keeptrials = 'yes' gerade definiert, weswegen speichern/laden
deutlich länger dauert!!
-------------------------------------------

%}
output_suffix = '_combined_afterAvg';

outPath = fullfile(options.outDir, options.pattern);
if ~isfolder(outPath)
    mkdir(outPath)
end

inDirs = dir(options.inDir);
all_subjects = {inDirs([inDirs.isdir]).name};
all_subjects = all_subjects(3:end);
% all_subjects = all_subjects(3:4);
% submit one job that contains multiple tasks (one task per subject)
job = createJob(cluster, 'Name', 'evoked');
%%
for i = 1:numel(all_subjects)
    createTask(job, @scc_evoked, 0, {all_subjects{i}, options, preproc_cfg, output_suffix});
end
submit(job);