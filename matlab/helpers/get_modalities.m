function [modalities, modalities_codes, conditions, conditions_codes] = get_modalities()
    modalities = {'auditory', 'tactile', 'visual'};
    modalities_codes = [1, 2, 3];
    conditions = {'NT_hit', 'NT_miss', 'HI', 'catch'};
    conditions_codes = [
        1,1;
        1,0;
        2,1;
        3,0];
    