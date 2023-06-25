function [Ubasis, Core, Tensor, k] = BestRankR(Tensor,maxitr,delta,Urank,mode,L,gmode)
%% 固定张量秩下的分解
% Input:
%  -- Tensor: an input N-th order tensor object
%  -- maxitr: maximum iteration
%  -- delta: tolerance error
%  -- Urank: given rank of factor matrix
%  -- mode: boolean (1 denotes the N-1)
%  -- L: Laplacian operator for factor matrix prior
%  -- gmode: an input N-th order tensor object
% Output:
%  -- Ubasis, Core: factor matrix, Core tensor
%  -- Tensor: an output N-th order tensor object
%  -- k: 

% Initializaion
if mode==1 % bool
    Ubasis = HOSVD(Tensor,'N-1'); % 返回元组，其中第一个元素是factor matrix
else
    Ubasis = HOSVD(Tensor,'N');
end
for d = 1 : ndims(Tensor)-mode
    Ubasis{1,d} = Ubasis{1,d}(:,1:Urank(d)); % 选取 Ubasis factor matrix 中某几列，Urank表示给定的秩
end

current_error = zeros(1,ndims(Tensor)-mode);
% Low rank approximation of best rank R
for k = 1 : maxitr
    last_error = current_error;
    for d1 = 1 : ndims(Tensor)-mode
        Z = Tensor;
        coef = 1;
        for d2 = 1 : ndims(Tensor)-mode
            if d1~=d2
                Z = TensorProduct(Z,Ubasis{1,d2}',d2);
                coef = coef*trace(Ubasis{1,d2}'*(L{d2}*L{d2})'*Ubasis{1,d2}); % multilinear graph embedding (factor prior)
            end
        end
        if gmode==0 || gmode==1, coef = gmode; end
        Z = shiftdim(Z,d1-1);
        Z = reshape(Z,size(Z,1),[]);
        [U, S] = eig(Z*Z'-coef*(L{d1}*L{d1}')); 
        [~,sidx] = sort(diag(S),'descend');
        U = U(:,sidx);
        current_error(d1) = norm(Ubasis{1,d1}'*U(:,1:Urank(d1)),'fro')/sqrt(length(Ubasis{1,d1}(:)));
        Ubasis{1,d1} = U(:,1:Urank(d1));
    end
    if isempty(find(current_error-(1-delta)*Urank<0, 1)) || isempty(find(abs(current_error-last_error)>10^-7, 1))
        break;
    end
end
clear Z;

% Rank-reduced approximation
for d = 1 : ndims(Tensor)-mode
    Tensor = TensorProduct(Tensor,Ubasis{1,d}',d);
end
Core = Tensor;
for d = 1 : ndims(Tensor)-mode
    Tensor = TensorProduct(Tensor,Ubasis{1,d},d);
end