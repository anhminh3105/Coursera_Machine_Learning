function [error_train, error_val] = ...
    randExamplesLearningCurve(X, y, Xval, yval, lambda)
m = size(y);

for i=1:m
    X_perm_index = randperm(length(X));
    X_perm = X(X_perm_index);
    y_perm = y(X_perm_index);
    
    Xval_perm_index = randperm(length(Xval));
    Xval_perm = Xval(Xval_perm_index);
    yval_perm = yval(Xval_perm_index);
    
    theta = trainLinearReg(X_perm(1:i), y_perm(1:i), lambda);
    error_train = linearRegCostFunction(X_perm(1:i), y_perm(1:i), theta, 0);
    error_val = linearRegCostFunction(Xval_perm(1:i), y_perm(1:i), theta, 0);
end
