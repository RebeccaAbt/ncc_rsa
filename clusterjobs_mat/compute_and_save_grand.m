function compute_and_save_grand(subjectList, options)
    % Compute evoked per subject in parallel, save all_evoked, compute and save grand.
    % subjectList : cell array of subject IDs
    % options : struct with fields inDir, pattern, outDir (optional)

    % defaults
    rmpath /opt/matlab/R2026a/toolbox/fixedpoint/fixedpoint/

    opt_default.inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/';
    opt_default.pattern = 'maxfilter_True__ica_True__0.5-NoneHz__fs_1000__[-1.5_1.5]s_detrend_1_meg_clean-epo';
    opt_default.outDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/matlab_evoked/';
    if nargin < 2
        options = struct();
    end
    options = catstruct(opt_default, options);

    inDir = options.inDir;
    pattern = options.pattern;
    outDir = options.outDir;

    if ~iscell(subjectList)
        error('subjectList must be a cell array of subject IDs');
    end

    nsub = numel(subjectList);
    all_evoked = cell(nsub,1);

    % prepare parallel pool if none
    pool = gcp('nocreate');
    if isempty(pool)
        parpool; % use default profile
    end

    parfor si = 1:nsub
        subj = subjectList{si};
        try
            % build dataset list (up to 3 files as in scc_evoked)
            ds = {
                fullfile(inDir, subj, strcat(subj, '_', pattern, '.fif'))
                fullfile(inDir, subj, strcat(subj, '_', pattern, '-1.fif'))
                fullfile(inDir, subj, strcat(subj, '_', pattern, '-2.fif'))
                };
            % remove non-existing paths to avoid errors
            ds = ds(cellfun(@(x) exist(x,'file')==2, ds));
            if isempty(ds)
                error('No dataset files found for subject %s', subj);
            end

            % Preprocessing
            cfg = [];
            cfg.dataset = ds;
            data = ft_preprocessing(cfg);

            data = ft_combineplanar([],data);

            % collect events across files if necessary
            events = [];
            for k = 1:numel(cfg.dataset)
                hdr = ft_read_header(cfg.dataset{k});
                if isfield(hdr,'orig') && isfield(hdr.orig,'epochs') && isfield(hdr.orig.epochs,'events')
                    events = [events; hdr.orig.epochs.events]; %#ok<AGROW>
                end
            end
            if ~isempty(events)
                assert(size(events,1) == numel(data.trial), 'Number of events does not match number of trials for %s', subj);
                codes = events(:,3);
                trialinfo = [
                    floor(codes/1000), ...
                    mod(floor(codes/100),10), ...
                    mod(floor(codes/10),10), ...
                    mod(codes,10)
                    ];
                data.trialinfo = trialinfo;
            else
                % If no events found, attempt to use existing data.trialinfo (if present)
                if ~isfield(data,'trialinfo')
                    error('No event/trialinfo found for subject %s', subj);
                end
            end

            % get modalities/conditions definitions from helper if available
            [modalities, modalities_codes, conditions, conditions_codes] = get_modalities();


            % build logical indices for each condition
            evoked_conditions = struct();
            for m = 1:numel(modalities)
                moda = modalities{m};
                i_mod = modalities_codes(m);
                for c = 1:numel(conditions)
                    cond = conditions{c};
                    i_cond = conditions_codes(c,:);
                    evoked_conditions.(moda).(cond) = ...
                        data.trialinfo(:,1)==i_cond(1) & ...
                        data.trialinfo(:,2)==i_mod & ...
                        data.trialinfo(:,3)==i_cond(2);
                end
            end

            % compute evoked for this subject
            evoked = struct();
            for m = 1:numel(modalities)
                modality = modalities{m};
                for c = 1:numel(conditions)
                    condition = conditions{c};
                    cfg = [];
                    cfg.trials = evoked_conditions.(modality).(condition);
                    cfg.channel = 'meg';
                    % If zero trials, skip and create empty placeholder
                    if ~any(cfg.trials)
                        evoked.(modality).(condition) = [];
                    else
                        evoked.(modality).(condition) = ft_timelockanalysis(cfg, data);
                    end
                end
            end

            all_evoked{si} = evoked;
        catch ME
            warning('Subject %s failed: %s', subjectList{si}, ME.message);
            all_evoked{si} = [];
        end
    end

    % save the collected evoked structs into one MAT file
    outPath = fullfile(outDir, pattern);
    if ~isfolder(outPath)
        mkdir(outPath);
    end
    allFile = fullfile(outPath, strcat('all_evoked', options.pattern, '.mat'));
    s = struct('all_evoked', {all_evoked});
    save(allFile, '-v7.3', '-struct', 's');

    % Compute grand average (same logic as your previous script)
    % Discover modalities/conditions from first non-empty subject
    firstIdx = find(~cellfun(@isempty, all_evoked), 1);
    if isempty(firstIdx)
        error('No evoked data computed for any subject.');
    end
    first = all_evoked{firstIdx};
    modalities = fieldnames(first);
    conditions = fieldnames(first.(modalities{1}));

    % prepare grand inputs
    grand_inputs = struct();
    for m = 1:numel(modalities)
        modname = modalities{m};
        for c = 1:numel(conditions)
            condname = conditions{c};
            grand_inputs.(modname).(condname) = {};
        end
    end

    % collect per-subject, select meg channels
    for s = 1:numel(all_evoked)
        ev = all_evoked{s};
        if isempty(ev)
            continue
        end
        for m = 1:numel(modalities)
            modname = modalities{m};
            for c = 1:numel(conditions)
                condname = conditions{c};
                if ~isfield(ev.(modname), condname) || isempty(ev.(modname).(condname))
                    continue
                end
                subj_ev = ev.(modname).(condname);
                cfg = [];
                cfg.channel = 'meg';
                sel = ft_selectdata(cfg, subj_ev);
                grand_inputs.(modname).(condname){end+1} = sel; %#ok<AGROW>
            end
        end
    end

    % compute grand averages
    grand = struct();
    for m = 1:numel(modalities)
        modname = modalities{m};
        for c = 1:numel(conditions)
            condname = conditions{c};
            list = grand_inputs.(modname).(condname);
            if isempty(list)
                grand.(modname).(condname) = [];
                continue
            end
            grand.(modname).(condname) = ft_timelockgrandaverage([], list{:});
        end
    end

    % save grand averages
    grandFile = fullfile(outPath, strcat('grand_evoked_', options.pattern, '.mat'));
    g = struct('grand', grand);
    save(grandFile, '-v7.3', '-struct', 'g');

end
