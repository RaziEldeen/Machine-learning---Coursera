function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
a_1=[ones(m,1),X];
z_2=a_1*Theta1';
a_2=sigmoid(z_2);
a_2=[ones(m,1),a_2];
z_3=a_2*Theta2';
a_3=sigmoid(z_3);

[~,i]=max(a_3,[],2);
p=i;

% =========================================================================


end
