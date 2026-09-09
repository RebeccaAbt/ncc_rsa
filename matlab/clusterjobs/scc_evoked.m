% requires the Add-On "catstruct" !!!

function scc_evoked(subjectID, options, preproc_cfg, output_suffix)
    addpath('/home/scc_e_393956/MATLAB Add-Ons/Collections/fieldtrip/')
    ft_defaults()
    addpath('/home/scc_e_393956/ncc/rsa/matlab/helpers')
    addpath('/home/scc_e_393956/MATLAB Add-Ons/obob_ownft/')

    opt_default.inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/';
    opt_default.pattern = 'maxfilter_True__ica_True__0.5-NoneHz__fs_1000__[-1.5_1.5]s_detrend_0_meg_clean-epo';
    opt_default.outDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/matlab_evoked/';

    options = catstruct(opt_default, options);

    inDir = options.inDir;
    pattern = options.pattern;
    outDir = options.outDir;

    outFile = fullfile(outDir, pattern, strcat(subjectID, '_evoked', output_suffix, '.mat'));
    
    % automatically find out whether input dataset consists of multiple partial files

    primaryFile = fullfile(inDir, subjectID, strcat(subjectID, '_', pattern, '.fif'));
    file1 = fullfile(inDir, subjectID, strcat(subjectID, '_', pattern, '-1.fif'));
    file2 = fullfile(inDir, subjectID, strcat(subjectID, '_', pattern, '-2.fif'));
    ds = { };
    ds{end+1} = primaryFile; 
    if exist(file1, 'file') == 2;  ds{end+1} = file1; end
    if exist(file2, 'file') == 2;  ds{end+1} = file2; end
    if numel(ds) == 1; dataset = ds{1}; else dataset = ds; end


    sprintf('dataset: %s', dataset)

    cfg = [];
    cfg.dataset = dataset;
    cfg = catstruct(cfg, preproc_cfg);
    disp('cfg:')
    disp(cfg)

    data = ft_preprocessing(cfg);

    % % -------------------------------------------------------------- % Test: ADD FIXCHANNELS FUNCTION 
    % disp('--- tra before:')
    % disp(data.grad.tra)
    % disp('running fixchannels function...')
    % cfg = [];
    % cfg.interp = 'average';
    % cfg.load_default = 1;
    % cfg.neigh_method = 'triangulation'; % 'triangulation' 'template' 'spline'
    % data = obob_fixchannels(cfg, data); 


    % disp('--- tra after:')
    % disp(data.grad.tra)
    % % -------------------------------------------------------------- % COMBINE GRADIOMETERS before averaging
    % disp('combining gradiometers...')
    % 
    % data = ft_combineplanar([], data);

    events = [];
    if iscell(cfg.dataset)

        for i = 1:numel(cfg.dataset)
            hdr = ft_read_header(cfg.dataset{i});
            events = [events; hdr.orig.epochs.events];
        end
    else
        hdr = ft_read_header(cfg.dataset);
        events = hdr.orig.epochs.events;
    end
    assert(size(events,1) == numel(data.trial), 'Number of events does not match number of trials.')

    codes = events(:,3);

    trialinfo = [
        floor(codes/1000), ...                 % category
        mod(floor(codes/100),10), ...          % modality
        mod(floor(codes/10),10), ...           % response
        mod(codes,10)                          % stimulus
        ];
    data.trialinfo = trialinfo;

    [modalities, modalities_codes, conditions, conditions_codes] = get_modalities();

    evoked_conditions = struct();
    disp('generating trialinfo')
    for m = 1:3
        moda = modalities{m};
        i_mod = modalities_codes(m);
        for c = 1:4
            cond = conditions{c};
            i_cond = conditions_codes(c,:);
            evoked_conditions.(moda).(cond) = ...
                data.trialinfo(:,1)==i_cond(1) & ...
                data.trialinfo(:,2)==i_mod & ...
                data.trialinfo(:,3)==i_cond(2);
        end
    end


    % Compute evoked responses
    disp('Compute evoked responses')
    evoked = struct();

    numel(modalities)
    numel(conditions)

    for m = 1:numel(modalities)
        modality = modalities{m};
        for c = 1:numel(conditions)
            condition = conditions{c};

            cfg = [];
            cfg.trials = evoked_conditions.(modality).(condition);
            % cfg.keeptrials = 'yes';
            cfg.channel = 'meg';

            fprintf('%s %s: %d trials\n', modality, condition, sum(cfg.trials));
            
            ev = ft_timelockanalysis(cfg, data);

            % -------------------------------------------------------------- % COMBINE GRADIOMETERS after averaging
            disp('combining gradiometers...')
            ev = ft_combineplanar([], ev);

           evoked.(modality).(condition) = ev;

        end %for conditions
    end %for modalities
    disp('saving outFile to...')
    disp(outFile)
    save(outFile, 'evoked', '-v7.3')

end %function
