function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
hypothesis = sigmoid(X * theta);

% You need to return the following variables correctly 
J = ((-1 * y') * log(hypothesis) - (( 1 - y)' * log(1 - hypothesis)))/m;
grad = (X' * (hypothesis - y))./m;

theta(1) = 0;
J = J + (lambda/(2 * m)) * (theta' * theta);
grad = grad + (lambda/m) * theta;

grad = grad(:);

end
