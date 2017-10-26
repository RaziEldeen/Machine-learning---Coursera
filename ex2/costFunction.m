function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
o =ones(m,1);
g =sigmoid(X*theta);
% vectorized implemtations
J= -(y'*log(g) + (o-y)'*log(o-g))/m ;
% This is the unvectorized:
% J = -sum(y.*log(sigmoid(X*theta))+(o-y).*log(o-sigmoid(X*theta)))/m;

grad = X'*(sigmoid(X*theta)-y)/m;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
