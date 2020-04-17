function [U, S]=lda(X,Y,K)
[m, n] = size(X);
U = zeros(n);
S = zeros(n);
centroids = zeros(K,n);
lengthCindex = zeros(1,K);
%% 寻找类内中心
for i = 1:K
cindex = find(Y == i);
lengthCindex(i) = size(cindex,1);
    for j = 1:lengthCindex(i)
    centroids(i,:) = X(cindex(j),:)+centroids(i,:);
    end
centroids(i,:) = centroids(i,:)/lengthCindex(i);
end

SB=zeros(n,n); 
mu = mean(X);
for i=1:K
SB=SB+lengthCindex(i)*(centroids(i,:)-mu)'*(centroids(i,:)-mu);
end
SW=zeros(n,n); 

for i=1:K
    cindex = find(Y == i);
    lengthCindex(i) = size(cindex,1);
    for j = 1:lengthCindex(i)
        SW=SW+(X(cindex(j),:)-centroids(i,:))'*(X(cindex(j),:)-centroids(i,:));
    end
end

matrix=pinv(SW)*SB;
[U,S,V] = svd(matrix);

end
