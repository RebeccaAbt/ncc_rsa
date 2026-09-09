
function timeStamps = events2timestamps_38regressors(subjectID, bids_root_raw)


    Run = {'run1', 'run2', 'run3', 'run4', 'run5', 'run6'};
    modality = {'aud', 'tac', 'vis'};
    stimuli = {'_1', '_2', '_3', '_4'};
    conditions = {'_miss', '_hit', '_HI'}; % here: missed NT, percevides NT & High intensity
    regressors = {'instruct', 'respScreen'};

    inFiles = dir([bids_root_raw subjectID '/ses-1/func/*events.tsv']);

    % ===================================================================== preallocate timestamps structure & create fieldnames for all conditions

    timeStamps = [];

    for iRun = 1:length(Run)

        timeStamps(iRun).runNr = Run{iRun};

        for iMods = 1:length(modality)
            for iResp = 1:2 % length(conditions) % --> better append HI trials at the end to avoid confusion
                for iStim = 1:length(stimuli)
                    fieldName = [modality{iMods} conditions{iResp} stimuli{iStim}];
                    timeStamps(iRun).(fieldName) = [];
                end
            end
        end
        iResp = 3; % append HI trials at the end to avoid confusion
        for iMods = 1:length(modality)
            for iStim = 1:length(stimuli)
                fieldName = [modality{iMods} conditions{iResp} stimuli{iStim}];
                timeStamps(iRun).(fieldName) = [];
            end
        end
    end

    % create fieldnames for all regressors
    for iRun = 1:length(Run)
        for iRegs = 1:length(regressors)
            fieldName = regressors{iRegs};
            timeStamps(iRun).(fieldName) = [];
        end
    end


    % ===================================================================== Fill structure with onsets and durations from events file
    for iRun = 1:length(inFiles)

        disp('reading table')
        inFile = fullfile(inFiles(iRun).folder, inFiles(iRun).name);

        assert(contains(inFiles(iRun).name, sprintf('ncc%i', iRun)), "fileRun '%s' not found in filename '%s'", sprintf('ncc%i', iRun), inFiles(iRun).name);

        t = readtable(inFile, "FileType","text",'Delimiter', '\t');

        idx_stim = find(cellfun(@(x) strcmp(x, 'Stimulus'), t.event));
        idx_resp = find(cellfun(@(x) strcmp(x, 'Response'), t.event));
        idx_intro = find(cellfun(@(x) strcmp(x, 'Intro'), t.event) | cellfun(@(x) strcmp(x, 'Outro'), t.event));
        idx_respScr = find(cellfun(@(x) strcmp(x, 'ResponseScreen'), t.event));

        assert (length(idx_stim) == length(idx_resp), 'number of Stimulus indices is unequal the number if response indices')

        onsets = t.onset(idx_stim);
        durations = t.duration(idx_stim);

        for i = 1:length(idx_stim) % loop over all Stimulus events

            % ====== get fieldname:
            mod = modality{t.modality(idx_stim(i))};
            stim = stimuli{t.stimulus(idx_stim(i))};
            condition = t.condition(idx_stim(i));

            switch condition
                case 1	% NT --> distinguish hits and misses
                    resp = t.response(idx_resp(i));
                    if strcmp(resp, 'n/a')
                        continue % no reponse --> not modelled as condition
                    else
                        cond = conditions{str2double(resp{:})+1};
                    end

                case 2	% HI
                    cond = conditions{3};
                case 3 % catch
                    continue % count catch trials to baseline

            end

            fieldName = [mod cond stim];
            % ====== get onset and duration:
            ts = [onsets(i) durations(i)];
            % ====== append onset and duration to structure:
            timeStamps(iRun).(fieldName) = [timeStamps(iRun).(fieldName); ts]; % append new timestamp
        end


        fieldName = regressors{1};
        onsets = t.onset(idx_intro);
        durations = t.duration(idx_intro);
        for i = 1:length(idx_intro)
            ts = [onsets(i) durations(i)];
            timeStamps(iRun).(fieldName) = [timeStamps(iRun).(fieldName); ts]; % append new timestamp
        end


        fieldName = regressors{2};
        onsets = t.onset(idx_respScr);
        durations = t.duration(idx_respScr);
        for i = 1:length(idx_respScr)
            ts = [onsets(i) durations(i)];
            timeStamps(iRun).(fieldName) = [timeStamps(iRun).(fieldName); ts]; % append new timestamp
        end

    end% for iRun

end %function


