function [J , grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
a1=[ones(m,1),X];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1),a2];
z3=a2*Theta2';
a3=sigmoid(z3);

one=ones(size(y_matrix));
J1=sum(sum(-y_matrix.*log(a3)-(one-y_matrix).*log(one-a3)))/m;
Tht1=Theta1(:,2:end);
Tht2=Theta2(:,2:end);
J2=(sum(sum(Tht1.^2))+sum(sum(Tht2.^2)))*(lambda/(2*m));

J=J1+J2;


d3=a3-y_matrix;
d2=d3*Tht2.*sigmoidGradient(z2);
Delta1=d2'*a1;
Delta2=d3'*a2;

Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;

Theta1(:,1)=0;
Theta2(:,1)=0;
tht1=Theta1.*(lambda/m);
tht2=Theta2.*(lambda/m);
   
Theta1_grad=Theta1_grad + tht1;
Theta2_grad=Theta2_grad + tht2;  

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
