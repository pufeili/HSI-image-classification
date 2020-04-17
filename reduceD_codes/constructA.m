function A = constructA(X,options)
    if nargin <2
       error('parameters are not enough!');
    end

    if (~exist('options','var'))
       options = [];
    end
    
    if ~isfield(options,'k')
        options.k = 5;
    end

    if ~isfield(options,'WeightMode')
    options.WeightMode = 'binary';
    end

    [m,n] = size(X);
    A = zeros(m,m);
    dist = EuDist2(X(1:m,:),X,0);
    [val,idx] = sort(dist,2);
    idx = idx(:,1:options.k+1);   
    val = val(:,1:options.k+1);
    switch options.WeightMode
        case 'binary'       
            for i = 1:m
                A(i,idx(i,:)) = 1;
            end
        case 'heatkernel'
            if ~isfield(options,'t')
                options.t = 10;
            end
            val = exp(-val/(2*options.t^2));
            for i = 1:m
                A(i,idx(i,:)) = val(i,:);
            end
        otherwise
            error('WeightMode does not exist!');
    end  
    A = sparse(A);
    A = A - diag(diag(A));
    A = max(A,A');
    
end