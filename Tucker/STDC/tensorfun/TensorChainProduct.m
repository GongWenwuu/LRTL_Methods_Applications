function X = TensorChainProduct(X,U,list) % loop for TensorProduct
for i = 1 : numel(list)
    X = TensorProduct(X,U{list(i)},list(i));
end

% function X = TensorChainProductT(X,U,list)
% for i = 1 : numel(list)
%     X = TensorProduct(X,U{list(i)}',list(i)); % 区别在于factor matrix转置了
% end