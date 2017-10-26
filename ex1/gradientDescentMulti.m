function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
 
a=theta(1)-(alpha/m)*sum(X*theta-y);
b=theta(2)-(alpha/m)*(X*theta-y)'*X(:,2);
c=theta(3)-(alpha/m)*(X*theta-y)'*X(:,3);
theta(1)=a;
theta(2)=b;
theta(3)=c;

J_history(iter) = computeCostMulti(X, y, theta);

end

end
