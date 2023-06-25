function X = TensorProduct(Tensor,U,d) % ȷ����������n-mode������

ndim0 = size(Tensor);
ndim0(d) = size(U,1);

X = shiftdim(Tensor,d-1); % ȷ��n-mode���
ndim = size(X);
X = reshape(X,size(X,1),numel(X)/size(X,1)); % unfoldingΪ����
X = U*X;
X = reshape(X,[size(X,1) ndim(2:end)]); % folding Ϊ����
X = shiftdim(X,ndims(Tensor)-(d-1));

X = reshape(X,ndim0);