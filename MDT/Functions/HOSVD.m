function [Ubasis, Core, Sigma] = HOSVD(Tensor,mode) 
%% �߽������ֽ�
% Input:
%  -- Tensor: an input N-th order tensor object
%  -- mode: boolean (1 denotes the N-1)
% Output:
%  -- Ubasis, Core, Sigma: factor matrix, Core tensor

switch mode
    case 'N-1'
        for d = 1 : ndims(Tensor)-1 % ndims ��ʾ������ά�ȣ�vector ��2��3-order tensor ��3
            T = shiftdim(Tensor,d-1); % �ڶ���������ʾ����ά�ȵ�index:0,1,2
            T = reshape(T,size(T,1),numel(T)/size(T,1)); % size(T,1)��ʾ����T��ά��1�µĳ��ȣ�������T����ά��size(T,1)*prod(size(T)/size(T,1))=prod(size(T))��С unfoldingΪ����
            [Ubasis{d}, S] = svd(T);
            Sigma{d} = diag(S);
        end
        Core = Tensor;
        for d = 1 : ndims(Tensor)-1
             Core = TensorProduct(Core,(Ubasis{d})',d); % Self-defined function, ���core�����Ĺ�ʽ
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
