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

  N  = length(U); % 张量的维度大小
  for n = 1:N
    C(n) = size(U{n},1); % 储存factor 矩阵维度大小，也是矩阵的秩，cell数组U的使用
    R(n) = size(U{n},2);
  end

  T = Q.*T; % 元素对应位置相乘，从而构造缺失张量
  GU= tensor_alprod(G,U,0,R); % 初始化目标张量，基于Tucker分解的方式返回补全张量

  obj = (1/sum(Q(:)))*norm(T(Q(:)==1) - GU(Q(:)==1))^2; % 补全的误差计算

  Z = T; % 缺失张量
  for iter = 1:maxiter

    % update parameters  针对初始化的补全张量进行更新
    Z(Q(:)~=1) = GU(Q(:)~=1); % 引入 auxiliary function，即GU是无缺失数据张量
    
    for iter2 = 1:inloop 
    for n = 1:N % 算法创新之处，优化辅助函数（初始化缺失张量）使得原问题变成了无缺失的张量分解问题，采用ALS求解
      Y{n} = unfold(tensor_alprod_exc(Z,U,1,n,C),n); % 无缺失数据的张量分解
      [U{n},~,~] = svds(Y{n}*Y{n}',R(n)); % factor 矩阵进行更新，指定矩阵秩大小
    end
    end
    
    G = tensor_alprod(Z,U,1,C); % core 核张量进行更新，注意对比参数tr=1,Dg=C的区别
    
    GU= tensor_alprod(G,U,0,R); % 缺失张量补全更新

    % calc. cost function
    obj2 = (1/sum(Q(:)))*norm(T(Q(:)==1) - GU(Q(:)==1))^2;
    histo(iter) = obj2; % 每次迭代的误差项保存
    
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


