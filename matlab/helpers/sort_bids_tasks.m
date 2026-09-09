
function [sorted_files, sort_idx] = sort_bids_tasks(bold_files)
    % Extract task numbers from filenames (e.g., task-ncc1 -> 1)
    task_nums = [];

    for i = 1:length(bold_files)
        % Find task-xxx pattern
        match = regexp(bold_files{i}, 'task-(\w+)', 'tokens');
        if ~isempty(match)
            task_name = match{1}{1};
            % Extract number from task name (e.g., ncc1 -> 1)
            nums = regexp(task_name, '\d+', 'match');
            if ~isempty(nums)
                task_nums(i) = str2double(nums{end});
            else
                task_nums(i) = i;
            end
        else
            task_nums(i) = i;
        end
    end

    [~, sort_idx] = sort(task_nums);
    sorted_files = bold_files(sort_idx);
end