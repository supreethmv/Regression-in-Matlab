function w = LeastSquares(DesignMatrix,Y)
% LEASTSQUARES Function
% Design matrix n×D from training data
% y = nx1 vector from training data
% w = weights vector w of least squares regression as column vector D x 1


%% Using A\b instead of inv(A)*b is two to three times faster, and produces residuals on the order of machine accuracy relative to the magnitude of the data.

% w = inv(DesignMatrix' * DesignMatrix) * (DesignMatrix' * Y);

w = (DesignMatrix' * DesignMatrix) \ (DesignMatrix' * Y);



end
