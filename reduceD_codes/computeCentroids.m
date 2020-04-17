function centroids = computeCentroids(X, idx, K)

[m n] = size(X);
centroids = zeros(K, n);

for i = 1:K
cindex = find(idx == i);
lengthCindex = size(cindex,1);
    for j = 1:lengthCindex
    centroids(i,:) = X(cindex(j),:)+centroids(i,:);
    end
centroids(i,:) = centroids(i,:)/lengthCindex;
end

end

