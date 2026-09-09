

myDir = '/home/scc_e_393956/ncc/rsa';
cd(myDir)
addpath('/home/scc_e_393956/ncc/rsa/clusterjobs_mat/')

cluster = parcluster('SCC');
cluster.JobStorageLocation = '/home/scc_e_393956/ncc/rsa/jobs/matlab';

cluster.NumWorkers = 16;
cluster.AdditionalProperties.AdditionalSubmitArgs = '--mem=256 --time=05:00:00 --job-name=mat_evoked';
cluster.NumThreads = 1;


options.inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/';
options.pattern = 'maxfilter_True__ica_True__0.5-NoneHz__fs_1000__[-1.5_1.5]s_detrend_0_meg_clean-epo';
options.outDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/matlab_evoked/';

outPath = fullfile(options.outDir, options.pattern);
if ~isfolder(outPath)
    mkdir(outPath)
end

inDirs = dir(options.inDir);
all_subjects = {inDirs([inDirs.isdir]).name};
all_subjects = all_subjects(3:end);
all_subjects = all_subjects(3:7);

job = batch(cluster, @compute_and_save_grand, 0, {all_subjects, options}, 'Pool', 16, 'CaptureDiary', true);

% job = batch( ...
%     cluster, ...
%     @compute_and_save_grand, ...
%     0, ...
%     {all_subjects, options});

