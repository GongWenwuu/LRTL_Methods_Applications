%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [ X ] = DE_tensor_all_adjoint(V, S)
%
% This function claculates the Delay Embedding for all directions (modes)
% and used to inverse tensor data
%
% inputs:
%   -- V       : input incomplete tensor: (N-th-order tensor)
%   -- S       : mask tensor, 0:missing, 1:available (N-th-order tensor)
% 
% output:
%   -- X       : MDT tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ X ] = DE_tensor_all_adjoint( V, S )

  N2  = ndims(V);
  N   = N2/2;
  JJ  = size(V);
  if mod(length(JJ),2)
      JJ = [JJ 1];
  end
  JJ2 = JJ(1:2:end-1) .* JJ(2:2:end);
  if length(JJ2) == 1
      JJ2 = [JJ2 1];
  end
  V = reshape(V, JJ2); % 矩阵
  X = tensor_allprod(V,S,1,size(V));
  
end

% function [ V, S ] = DE_tensor_all( X, tau, S)
%   % Delay Embedding for all directions (modes)
%   % 
%   N  = ndims(X);
%   II = size(X);
% 
%   if isempty(S)
%     for n = 1:N
%       S{n} = make_duplication_matrix(II(n),tau(n)); % Self-defined 函数，矩阵转化为Hankel矩阵S
%     end
%   end
%   
%   I2= II - tau + 1;
%   JJ= [tau; I2]; JJ = JJ(:);
%   V = tensor_allprod(X,S,0,size(X)); % Self-defined 函数，张量Tucker分解式的乘积计算
%   V = reshape(full(V),[JJ']);
% end