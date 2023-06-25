%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Z = Tensor_allprod_exc(G,U,tr,exc,Dg)
%
% This function claculates the exception n-mode multiplication 
% G * U{1} * U{2} * ...*{-U{m}}... * U{N}
%
% inputs:
%   -- G       : input tensor
%   -- U       : input (1xN)-cel factor matrices
%   -- tr      : input rank of the corresponding factor matrices
%   -- exc     : input m-mode exception
%   -- Dg      : input dimension vector of factor matrix (row or col), very important!  
%
% output:
%   -- Z       : output tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Z = tensor_allprod_exc(G,U,tr,exc,Dg)

  N = length(Dg);

  Z = G;
  for n = 1:N
    if ~isempty(U{n}) && n~=exc % another condition
      if tr == 0
        Z = tmult(Z,U{n},n,Dg);
        Dg(n) = size(U{n},1);
      else
        Z = tmult(Z,U{n}',n,Dg);
        Dg(n) = size(U{n},2);
      end
    end
  end
