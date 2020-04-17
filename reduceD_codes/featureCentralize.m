function [X_norm, mu, sigma] = featureCentralize(X)

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
