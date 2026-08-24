%%
addpath('/home/scc_e_393956/MATLAB Add-Ons/Collections/FieldTrip')
ft_defaults()

myDir = '/home/scc_e_393956/ncc/rsa';
cd(myDir)

%%

subjectID ='19800616mrgu';
inDir = '/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/19800616mrgu (copy 1)';
pattern = 'maxfilter_True__ica_True__0.5-NoneHz__fs_1000__[-1.5_1.5]s_detrend_0_meg_clean-epo';

fullfile(inDir, strcat(subjectID, '_', pattern, '.fif'))

cfg = [];
cfg.dataset = {
    fullfile(inDir, strcat(subjectID, pattern, '.fif'))
    fullfile(inDir, strcat(subjectID, pattern, '-1.fif'))
    fullfile(inDir, strcat(subjectID, pattern, '-2.fif'))
    };

dataAll = ft_preprocessing(cfg);

%%

inDirs =  dir('/home/scc_e_393956/Desktop/reabt/ncc/MEG/epochs_clean2/manual_finish/');
all_subjects = {inDirs([inDirs.isdir]).name};

%%


allEvents = [];

for i = 1:numel(cfg.dataset)
    hdr = ft_read_header(cfg.dataset{i});
    allEvents = [allEvents; hdr.orig.epochs.events];
end

% Sanity check
assert(size(allEvents,1) == numel(dataAll.trial), 'Number of events does not match number of trials.')

codes = allEvents(:,3);

trialinfo = [
    floor(codes/1000), ...                 % category
    mod(floor(codes/100),10), ...          % modality
    mod(floor(codes/10),10), ...           % response
    mod(codes,10)                          % stimulus
    ];
dataAll.trialinfo = trialinfo;


%%

modalities = {'auditory', 'tactile', 'visual'};
modalities_codes = [1, 2, 3];
conditions = {'NT_hit', 'NT_miss', 'HI', 'catch'};
conditions_codes = [
    1,1;
    1,0;
    2,1;
    3,0];

evoked_conditions = struct();

for m = 1:3
    moda = modalities{m};
    i_mod = modalities_codes(m);
    for c = 1:4
        cond = conditions{c};
        i_cond = conditions_codes(c,:);
        evoked_conditions.(moda).(cond) = ...
            dataAll.trialinfo(:,1)==i_cond(1) & ...   
            dataAll.trialinfo(:,2)==i_mod & ...  
            dataAll.trialinfo(:,3)==i_cond(2);        
    end
end


%% Compute evoked responses

evoked = struct();

for m = 1:numel(modalities)

    modality = modalities{m};

    for c = 1:numel(conditions)

        condition = conditions{c};

        cfg = [];
        cfg.trials = evoked_conditions.(modality).(condition);
        cfg.channel = 'meg';

        fprintf('%s %s: %d trials\n', ...
            modality, condition, sum(cfg.trials));

        evoked.(modality).(condition) = ft_timelockanalysis(cfg, dataAll);

    end
end

%%

%% Plot evoked responses (single subject)

modalities = {'auditory', 'tactile', 'visual'};
conditions_plot = {'HI', 'NT_hit', 'NT_miss', 'catch'};

colors = [
    0.85 0.33 0.10;   % HI
    0.00 0.45 0.74;   % NT hit
    0.47 0.67 0.19;   % NT miss
    0.49 0.18 0.56    % catch
    ];

n_best = 2;

figure;
set(gcf,'Position',[100 100 1400 400])


for m = 1:numel(modalities)

    modality = modalities{m};

    subplot(1,3,m)
    hold on

    % Find best channels based on HI

    HI = evoked.(modality).HI;

    % absolute maximum over time for every channel
    max_activity = max(abs(HI.avg), [], 2);
    [~, best_idx] = sort(max_activity, 'descend');
    best_idx = best_idx(1:n_best);

    for c = 1:numel(conditions_plot)    % Plot conditions

        condition = conditions_plot{c};
        evo = evoked.(modality).(condition);

        % average absolute signal over best channels
        y = mean(abs(evo.avg(best_idx,:)),1);

        plot(evo.time, y, ...
            'LineWidth', 2, ...
            'Color', colors(c,:));
    end

    title(modality)
    xlabel('Time (s)')
    ylabel('Amplitude')
    xlim([-0.2 1])
    grid on
    legend(conditions_plot, ...
        'Location','best')

end

%%


data = evoked.tactile.HI;

data_combined = ft_combineplanar([],data);


cfg = [];
cfg.layout = 'neuromag306cmb.lay';
ft_topoplotER(cfg,data_combined)
