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

all_models_errors = eye(64,3);
i = 0;

for C_new = [0.01 0.03 0.1 0.3 1 3 10 30]
    for sigma_new = [0.01 0.03 0.1 0.3 1 3 10 30]
        model = svmTrain(X, y, C_new, @(x1, x2) gaussianKernel(x1, x2, sigma_new));
        preds = svmPredict(model, Xval);
        curr_error = mean(double(preds ~= yval));

        i = i + 1;
        all_models_errors(i,:) = [C_new, sigma_new, curr_error];

    end
end


sorted_errors = sortrows(all_models_errors, 3);
C = sorted_errors(1,1);
sigma = sorted_errors(1,2);


% =========================================================================

end
