%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [X, G, U, histo] = Tucker_completion(T,Q,G,U,maxiter,inloop,tol,verb)
%
% This program solves fixed rank Tucker decomposition of input incomplete tensor.
%
% min  || Q.*(T - X) ||_F^2
% s.t. X = G * U{1} * U{2} * ... * U{N},
%      size(G) = (R_1, R_2, ..., R_N),
%
% inputs:
%   -- T       : input incomplete tensor
%   -- Q       : mask tensor, 0:missing, 1:available
%   -- G       : initialization of core tensor of Tucker decomposition
%   -- U       : initialization of (1xN)-cel array consisting of factor matrices
%   -- maxiter : maximum number of iterations
%   -- inloop  : number of iterations for inner loop
%   -- tol     : tolerance parameter for checking convergence
%   -- verb    : verbosity for visualizing process of algorithm
%
% outputs:
%   -- X     : output complete tensor
%   -- G     : result of core tensor of Tucker decomposition
%   -- U     : result of (1xN)-cel array consisting of factor matrices
%   -- histo : history of || Q.*(T - X) ||_F^2 / |Q| recorded for each iteration
%
% This code was written by Tatsuya Yokota (2017.08.28)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, G, U, histo] = Tucker_completion(T,Q,G,U,maxiter,inloop,tol,verb)

  N  = length(U); % ������ά�ȴ�С
  for n = 1:N
    C(n) = size(U{n},1); % ����factor ����ά�ȴ�С��Ҳ�Ǿ�����ȣ�cell����U��ʹ��
    R(n) = size(U{n},2);
  end

  T = Q.*T; % Ԫ�ض�Ӧλ����ˣ��Ӷ�����ȱʧ����
  GU= tensor_alprod(G,U,0,R); % ��ʼ��Ŀ������������Tucker�ֽ�ķ�ʽ���ز�ȫ����

  obj = (1/sum(Q(:)))*norm(T(Q(:)==1) - GU(Q(:)==1))^2; % ��ȫ��������

  Z = T; % ȱʧ����
  for iter = 1:maxiter

    % update parameters  ��Գ�ʼ���Ĳ�ȫ�������и���
    Z(Q(:)~=1) = GU(Q(:)~=1); % ���� auxiliary function����GU����ȱʧ��������
    
    for iter2 = 1:inloop 
    for n = 1:N % �㷨����֮�����Ż�������������ʼ��ȱʧ������ʹ��ԭ����������ȱʧ�������ֽ����⣬����ALS���
      Y{n} = unfold(tensor_alprod_exc(Z,U,1,n,C),n); % ��ȱʧ���ݵ������ֽ�
      [U{n},~,~] = svds(Y{n}*Y{n}',R(n)); % factor ������и��£�ָ�������ȴ�С
    end
    end
    
    G = tensor_alprod(Z,U,1,C); % core ���������и��£�ע��ԱȲ���tr=1,Dg=C������
    
    GU= tensor_alprod(G,U,0,R); % ȱʧ������ȫ����

    % calc. cost function
    obj2 = (1/sum(Q(:)))*norm(T(Q(:)==1) - GU(Q(:)==1))^2;
    histo(iter) = obj2; % ÿ�ε�����������
    
    % show process
    if mod(iter,verb) == 0
      fprintf('iter %d:: cost = %e :: cost_diff = %e \n',iter,obj2,abs(obj2-obj));
    end

    % convergence check
    if abs(obj2 - obj) < tol
      break;
    else
      obj = obj2;
    end

  end
  X = GU;


