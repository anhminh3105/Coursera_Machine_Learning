function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
try_entries = [0.01 0.03 0.1 0.3 1 3 10 30];
combos = zeros(length(try_entries) * length(try_entries), 3);
i = 0;
for c = 1:length(try_entries)
    for s = 1:length(try_entries)
        model = svmTrain(X, y, try_entries(c), @(x1, x2) gaussianKernel(x1, x2, try_entries(s)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        i = i + 1;
        combos(i, :) = [try_entries(c), try_entries(s), error];
    end
end

[~, min_error] = min(combos(:, 3));
C = combos(min_error,1);
sigma = combos(min_error, 2);
fprintf("Selected C =%f and sigma=%f", C, sigma);
% =========================================================================

end
