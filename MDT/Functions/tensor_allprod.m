%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Z = Tensor_allprod(G,U,tr,Dg)
%
% This function claculates the n-mode multiplication 
% Such as 3-way array G(m*p*q) by multiplying U(r*p) along second dimension into Z(m*r*q)
%
% inputs:
%   -- G       : input tensor
%   -- U       : input (1xN)-cel factor matrices
%   -- tr      : input rank of the corresponding factor matrices
%   -- Dg      : input dimension vector of factor matrix (row or col), very important!  
%
% output:
%   -- Z       : output tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Z = tensor_allprod(G,U,tr,Dg) 

  N = length(Dg);

  Z = G;
  for n = 1:N
    if ~isempty(U{n})
      if tr == 0
        Z = tmult(Z,U{n},n,Dg); % n-mode multiplication (Self-defined),a loop along with the input dimension vector
        Dg(n) = size(U{n},1);
      else
        Z = tmult(Z,U{n}',n,Dg);
        Dg(n) = size(U{n},2);
      end
    end
  end
  
% function [A,Tnew,Da]=tmult(T,M,n,Dt)
% Dm=size(M);
% Da=Dt;
% Da(n)=Dm(1);
% Tn=unfold(T,n);
% Tnew=M*Tn;
% A=fold(Tnew,n,Da);

