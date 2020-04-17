% K type_num, mu mean of X
function [U, S]=lpdp(X,Y,W,mu,K)
[m, n] = size(X);
U = zeros(n);
S = zeros(n);
mc = zeros(K,n);
lengthCindex = zeros(1,K);

for i = 1:K
cindex = find(Y == i);
lengthCindex(i) = size(cindex,1);
    for j = 1:lengthCindex(i)
        mc(i,:) = X(cindex(j),:)+mc(i,:);
    end
mc(i,:) = mc(i,:)/lengthCindex(i);
end

SB=zeros(n,n); 
m = mean(X);
for i=1:K
    SB=SB+lengthCindex(i)*(mc(i,:)-m)'*(mc(i,:)-m);
end
SW=zeros(n,n); 

for i=1:K
    cindex = find(Y == i);
    lengthCindex(i) = size(cindex,1);
    for j = 1:lengthCindex(i)
        SW=SW+(X(cindex(j),:)-mc(i,:))'*(X(cindex(j),:)-mc(i,:));
    end
end

D = diag(sum(W,2));
L = D - W;
%% ”–Œ Ã‚£ø£ø
matrix=pinv(mu*SW+(X'*L*X))*(SB);
[U,S,V] = svd(matrix);

end
