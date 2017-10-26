function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
% regularisation
m = length(y); 
o=ones(m,1);

J = -sum(y.*log(sigmoid(X*theta))+(o-y).*log(o-sigmoid(X*theta)))/m + ...
(lambda/(2*m))*(theta' * theta - theta(1)^2);
temp=theta;
temp(1)=0;
grad = X'*(sigmoid(X*theta)-y)/m;
grad=grad+(lambda /m).*temp;


grad = grad(:);

end
