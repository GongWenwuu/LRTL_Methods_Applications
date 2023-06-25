function [Ubasis, Core, Sigma] = HOSVD(Tensor,mode) 
%% 高阶张量分解
% Input:
%  -- Tensor: an input N-th order tensor object
%  -- mode: boolean (1 denotes the N-1)
% Output:
%  -- Ubasis, Core, Sigma: factor matrix, Core tensor

switch mode
    case 'N-1'
        for d = 1 : ndims(Tensor)-1 % ndims 表示张量的维度：vector 是2，3-order tensor 是3
            T = shiftdim(Tensor,d-1); % 第二个参数表示张量维度的index:0,1,2
            T = reshape(T,size(T,1),numel(T)/size(T,1)); % size(T,1)表示张量T在维度1下的长度；将张量T按照维度size(T,1)*prod(size(T)/size(T,1))=prod(size(T))大小 unfolding为矩阵
            [Ubasis{d}, S] = svd(T);
            Sigma{d} = diag(S);
        end
        Core = Tensor;
        for d = 1 : ndims(Tensor)-1
             Core = TensorProduct(Core,(Ubasis{d})',d); % Self-defined function, 求解core张量的公式
        end
    case 'N'
        for d = 1 : ndims(Tensor)
            T = shiftdim(Tensor,d-1);
            T = reshape(T,size(T,1),numel(T)/size(T,1));
            [Ubasis{d}, S] = svd(T);
            Sigma{d} = diag(S);
        end
        Core = Tensor;
        for d = 1 : ndims(Tensor)
             Core = TensorProduct(Core,(Ubasis{d})',d);
        end
end
