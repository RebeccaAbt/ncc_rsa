
myDir = '/home/scc_e_393956/ncc/rsa';
cd(myDir)
addpath('/home/scc_e_393956/ncc/rsa/clusterjobs_mat/')

cluster = parcluster('SCC');
cluster.JobStorageLocation = '/home/scc_e_393956/ncc/rsa/jobs/matlab';
%%
cluster.NumWorkers = 4;
cluster.NumThreads = 8;

cluster.AdditionalProperties.AdditionalSubmitArgs = '--mem=128G';

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
all_subjects = all_subjects(3:4);

jobs = cell(size(all_subjects));

for i = 1:numel(all_subjects)

    jobs{i} = batch( ...
        cluster, ...
        @scc_evoked, ...
        0, ...
        {all_subjects{i}, options});

end