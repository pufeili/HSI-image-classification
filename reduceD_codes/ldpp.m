function [U, S]=ldpp(X,Y,A,mu,gamma,type_num,ratio_b,ratio_w)
[N, D] = size(X);
U = zeros(D);
S = zeros(D);

SB=zeros(D); 
m = mean(X);
centroids_b = zeros(type_num,D);
lengthIndex_b = zeros(1,type_num);
[part_X,part_Y] = pickNeighbor(X,Y,m,round(N*ratio_b));
for i = 1:type_num
    index_b = find(part_Y == i);
        if isempty(index_b)
            index_b = find(Y == i,1);
        end
        lengthIndex_b(i) = size(index_b,1);
        for j = 1:lengthIndex_b(i)
            centroids_b(i,:) = part_X(index_b(j),:)+centroids_b(i,:);
        end
    centroids_b(i,:) = centroids_b(i,:)/lengthIndex_b(i);
end
for i=1:type_num
    SB=SB+lengthIndex_b(i)*(centroids_b(i,:)-m)'*(centroids_b(i,:)-m);
end

SW=zeros(D); 
centroids_w = zeros(type_num,D);
for i=1:type_num
    index_w = find(Y == i);
    N_index = size(index_w,1);
    mu_X = mean(X(index_w,:));
    [local_X,local_Y] = pickNeighbor(X(index_w,:),Y(index_w),mu_X,round(N_index*ratio_w));
    lengthIndex_w = size(local_Y,1);
    for j = 1:lengthIndex_w
        centroids_w(i,:) = local_X(j,:)+centroids_w(i,:);
    end
    centroids_w(i,:) = centroids_w(i,:)/lengthIndex_w;
    for j = 1:lengthIndex_w
        SW=SW+(local_X(j,:)-centroids_w(i,:))'*(local_X(j,:)-centroids_w(i,:));
    end
end

 
D = diag(sum(A,2));
L = D - A;

matrix=pinv(mu*SW+gamma*(X'*L*X))*(SB);
[U,S,V] = svd(matrix);

end
