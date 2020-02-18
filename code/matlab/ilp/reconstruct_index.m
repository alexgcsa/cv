% Convert the solution from ILP formulation into indices.
% The function returns a solution compatible with CVPARTITION.
% 
% Example: 
%   [assignment, grouped, result, indices] = ilp_slack5(data, 3)
%   solution = reconstruct_index(assignment, indices)

function solution = reconstruct_index(assignment, indices, seed, folds, data)

% Initialization
nrow = size(assignment, 1);
ncol = size(assignment, 2);
solution = nan(length(vertcat(indices{:})), 1);
pointers = ones(nrow, 1);                           % The pointer to the first unnasigned sample

% Randomly permute the rows within the same label power-set?
if nargin()==3
    rng(seed);
    for row=1:nrow
        subset = indices{row};
        randomized = randperm(length(subset));
        indices{row} = subset(randomized);
    end
end

% Use DOB-SCV subroutine?
if nargin()==5
    rng(seed);
    for row=1:nrow
        subset = indices{row};
        assigned = dobscv(data(subset, :), ones(length(subset), 1), folds);
        [~, sorted] = sort(assigned);
        indices{row} = subset(sorted);
    end
end

% Iterate over assignment 
for col=1:ncol
    for row=1:nrow
        cnt = assignment(row, col);                 % How many samples to assign
        i = indices{row};                           % Indexes of available samples
        j = i(pointers(row):(pointers(row)+cnt-1));
        solution(j) = col;
        
        % Increment the variables
        pointers(row) = pointers(row)+cnt;
    end
end

% QC: All samples must be assigned
assert(~any(isnan(solution)));
        