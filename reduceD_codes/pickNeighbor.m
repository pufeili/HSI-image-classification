function [res_X,res_Y] = pick_neighbor(X,Y,mu,n)
n_del = size(X,1)-n;
dist = sum(bsxfun(@power,bsxfun(@minus,X,mu),2),2);
[val,idx] = sort(dist,'descend');
idx = idx(1:n_del);
X(idx,:) = [];
Y(idx) = [];
res_X = X;
res_Y = Y;
end