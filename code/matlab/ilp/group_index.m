% Group data in matrix A based on all columns.
% Returns the grouped matrix KEYS with the observation counts COUNTS.
% Returns a cell with the INDICES to rows in matrix A.
% Also returns the count of unique values in each column.

% Example:
%   [keys, counts, indices, dim_counts] = group_index(A)
%   indices{1}     % Get the indices for the first row in the grouped data. 
%                  % The indices are a column vector.
 
function [keys, counts, indices, dim_counts] = group_index(A)   
    % Initialization
    nrow = size(A, 1);
    ncol = size(A, 2);

    % Single sort
    [sorted, index] = sortrows(A);

    % Prealocation of the result
    counts = ones(nrow, 1);
    keys = repmat(sorted(1, :), nrow, 1);
    indices = cell(1, nrow);
    indices{1} = index(1);

    % Single pass over the data
    tail = 1;
    for row = 2:nrow
        if all(sorted(row-1, :) == sorted(row, :))
            counts(tail) = counts(tail)+1;
        else
            tail = tail+1;
            keys(tail,:) = sorted(row, :);
        end
        indices{tail} = [indices{tail}; index(row)];
    end

    % Truncate the prealocated arrays
    counts = counts(1:tail);
    keys = keys(1:tail, :);
    indices = indices(1:tail);

    % Get the count of unique values for each column
    dim_counts = nan(1, ncol);
    for col=1:ncol
        dim_counts(col) = length(unique(keys(:, col)));
    end
