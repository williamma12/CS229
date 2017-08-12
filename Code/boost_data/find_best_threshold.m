function [ind, thresh] = find_best_threshold(X, y, p_dist)
% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

[mm, nn] = size(X);
ind = 1;
thresh = 0;

% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.

best_ind = 0;
best_thresh = 0;
best_error = intmax;
prev_best_error = intmax;

for j = 1:nn
    x_ind = X(:, j);
    [x_ind_sorted, sortIndex] = sort(x_ind);
    y_sorted = y(sortIndex);
    p_sorted = p_dist(sortIndex);
    
    best_ind_thresh = 0;
    best_ind_error = intmax;
    
    for m_0 = x_ind_sorted'
        ind_pos_error = 0;
        ind_neg_error = 0;
        for i = 1:length(x_ind_sorted)
            p_i = p_sorted(i);
            ind_pos_error = ind_pos_error + ...
                p_i * (m_0 - sign(x_ind_sorted(i)) ~= y_sorted(i));
            ind_neg_error = ind_neg_error + ...
                p_i * (sign(x_ind_sorted(i) - m_0) ~= y_sorted(i));
        end
        if best_ind_error > ind_pos_error
            best_ind_thresh = m_0;
            best_ind_error = ind_pos_error;
        elseif best_ind_error > ind_neg_error
            best_ind_thresh = m_0;
            best_ind_error = ind_neg_error;
        end
    end
    if best_error > best_ind_error
        best_ind = j;
        best_thresh = best_ind_thresh;
        prev_best_error = best_error;
        best_error = best_ind_error;
    end
    if prev_best_error < best_ind_error
        break
    end
end

ind = best_ind;
thresh = best_thresh;
