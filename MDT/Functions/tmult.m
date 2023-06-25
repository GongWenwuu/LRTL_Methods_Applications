function [A,Tnew,Da]=tmult(T,M,n,Dt)
% The Tensor multiplication also named the n-mode multiplication
% turns for instance a 3-way array T(m x p x q) by multiplying M(r x p) 
% along second dimension into A(m x r x q)
%
% Input
% T     tensor
% M     Jxsize(T,n)-matrix
% n     dimension to multiply
% Dt    dimension vector 
%
% Output:
% A     tensor
% Tnew  matrix given by M*matrizicing(T,n)

Dm=size(M);
Da=Dt;
Da(n)=Dm(1); % dimension is changeable along with n-mode 
Tn=unfold(T,n); % Self defined function
Tnew=M*Tn;
A=fold(Tnew,n,Da); % Self defined function

     

    
    
