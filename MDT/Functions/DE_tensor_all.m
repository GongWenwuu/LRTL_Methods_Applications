%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [ V, S ] = DE_tensor_all( X, tau, S)
%
% This function claculates the Delay Embedding for all directions (modes)
%
% inputs:
%   -- X       : tensor
%   -- tau     : transformer
%   -- S       : factor matrix
% 
% outputs:
%   -- V       : MDT tensor
%   -- S       : Hankel matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ V, S ] = DE_tensor_all( X, tau, S)
 
  N  = ndims(X);
  II = size(X);

  if isempty(S)
    for n = 1:N
      S{n} = make_duplication_matrix(II(n),tau(n)); % Self-defined ����������ת��ΪHankel����S,factor matrix
    end
  end
  
  I2= II - tau + 1;
  JJ= [tau; I2]; JJ = JJ(:);
  V = tensor_allprod(X,S,0,size(X)); % Self-defined ��������Ӧ���ھ����ϣ��õ�һ������
  V = reshape(full(V),JJ'); % ����һ������

end
