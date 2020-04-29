function w = RidgeRegression(DesignMatrix,y,lambda)
%% Least Squares with L2-Regularization

% add small ridge to the solution so that it becomes unique

% DesignMatrix = n×D training data
% y = nx1 vector from training data
% lambda = regularization parameter
% w = weights vector w of least squares regression as column vector D x 1

%% Using A\b instead of inv(A)*b is two to three times faster, and produces residuals on the order of machine accuracy relative to the magnitude of the data.

penality = lambda*eye(size(DesignMatrix,2));

% w = inv(DesignMatrix' * DesignMatrix + penality) * (DesignMatrix' * Y);

w = (DesignMatrix' * DesignMatrix + penality) \ (DesignMatrix' * y);



end
