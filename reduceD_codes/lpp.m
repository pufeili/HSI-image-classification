function [U,S] = lpp(X,W) 

[m, n] = size(X);
U = zeros(n,n);
S = zeros(n,n);

D = diag(sum(W,2));
L = D - W;
T1 = X'*L*X;
T2 = X'*D*X;
T = pinv(T2)*T1;%%求解特征向量

[U,S,V] = svd(T);
[val,ind] = sort(diag(S));%%求解特征值，并将其排序
U = U(:,ind);
S = S(ind,:);

end
