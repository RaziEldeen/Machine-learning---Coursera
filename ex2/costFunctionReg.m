function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
n=length(theta);
grad=zeros(n,1);
% Initialize some useful values
m = length(y); % number of training examples
o=ones(m,1);
g=sigmoid(X*theta);
% You need to return the following variables correctly 
J= - (y'*log(g) + (o-y)'*log(o-g))/m + (lambda/(2*m))*(theta'*theta-theta(1)^2);
%J = -sum(y.*log(sigmoid(X*theta))+(o-y).*log(o-sigmoid(X*theta)))/m ...
% +(lambda/(2*m))*(theta'*theta-theta(1)^2);
help = X'*(sigmoid(X*theta)-y)/m;
grad(1)=help(1);
grad(2)=help(2)+(lambda/m)*theta(2);
grad(3)=help(3)+(lambda/m)*theta(3);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
