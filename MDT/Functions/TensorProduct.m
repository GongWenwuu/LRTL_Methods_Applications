function X = TensorProduct(Tensor,U,d) % 确保张量按照n-mode情况相乘

ndim0 = size(Tensor);
ndim0(d) = size(U,1);

X = shiftdim(Tensor,d-1); % 确保n-mode相乘
ndim = size(X);
X = reshape(X,size(X,1),numel(X)/size(X,1)); % unfolding为矩阵
X = U*X;
X = reshape(X,[size(X,1) ndim(2:end)]); % folding 为张量
X = shiftdim(X,ndims(Tensor)-(d-1));

X = reshape(X,ndim0);