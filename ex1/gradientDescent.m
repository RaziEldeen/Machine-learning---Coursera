function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
for iter = 1:num_iters 
    
a=theta(1)-(alpha/m)*sum(X*theta-y);
b=theta(2)-(alpha/m)*sum((X*theta-y).*X(:,2));
theta(1)=a;
theta(2)=b;

J_history(iter) = computeCost(X, y, theta);
end

end
