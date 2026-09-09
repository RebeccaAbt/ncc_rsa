
function c = make_contrast(weights, regressorNames, nRepeats, padding)

    c = zeros(1, (numel(regressorNames) * nRepeats)+padding);

    fields = fieldnames(weights);

    for i = 1:numel(fields)

        name = fields{i};
        weight = weights.(name);

        idx = find(strcmp(regressorNames, name));

        if isempty(idx)
            error('Unknown regressor: %s', name);
        end

        cols = (idx-1)*nRepeats + (1:nRepeats);

        c(cols) = weight;
    end
end



