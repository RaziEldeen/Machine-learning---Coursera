function [theta, cost] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

    for iter = 1:num_iters
        theta = theta - (alpha)*X'*(sigmoid(X*theta) - y ) ;
    end
    [cost,~] = costFunction(theta, X,y);


  %  for j=1:length(theta)
   %     for i=1:m
    %        theta(j)=theta(j)-(alpha/m)*(X(i,:)*theta-y(i))*X(i,j);
     %   end
    %end
%b=theta(2)-(alpha/m)*(X*theta-y)'*X(:,2);
%c=theta(3)-(alpha/m)*(X*theta-y)'*X(:,3);
%theta(1)=a;
%theta(2)=b;
%theta(3)=c;

