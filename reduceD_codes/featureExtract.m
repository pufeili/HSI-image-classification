function [Z,U] = featureExtract(X,Y,method,type_num)

if nargin <4
   error('parameters are not enough!');
end

if (~exist('method','var'))
   method = [];
end

if ~isfield(method,'mode')
method.NeighborMode = 'pca';
end

switch method.mode
    case 'pca'
        if ~isfield(method,'K')
        method.K = 2;
        end
        U = pca(X);
    case'lpp'
        if ~isfield(method,'weightmode')
        method.weightmode = 'Binary';
        end
        if ~isfield(method,'knn_k')
        method.knn_k = 5;
        end
        if ~isfield(method,'t')
        method.t = 5;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        A = constructA(X,options);
        U = lpp(X,A);
    case 'lda'
        U = lda(X,Y,type_num);
    case 'lpdp'
        if ~isfield(method,'t')
        method.t = 10;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        A = constructA(X,options); 
        U = lpdp(X,Y,A,method.mu,type_num);
        case 'ldpp'
        if ~isfield(method,'t')
        method.t = 10;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        A = constructA(X,options); 
        U = ldpp(X,Y,A,method.mu,method.gamma,type_num,method.ratio_b,method.ratio_w);
    otherwise
        error('mode does not exist!');
end  

Z = projectData(X, U, method.K);
     
end