function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu(1)=mean(X(:,1));
mu(2)=mean(X(:,2));
sigma(1)=std(X(:,1));
sigma(2)=std(X(:,1));
 X_norm=[(X(:,1)-mu(1))/sigma(1),(X(:,2)-mu(2))/sigma(2)];

% ============================================================

end
