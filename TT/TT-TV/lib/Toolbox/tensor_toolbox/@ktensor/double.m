function A = double(X)
%DOUBLE Convert a ktensor to a double array.
%
%   A = double(X) converts X to a standard multidimensional array.
%
%   See also KTENSOR.
%
%MATLAB Tensor Toolbox.
%Copyright 2012, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2012) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt


if isempty(X.lambda) % check for empty tensor
    A = [];
    return;
end

sz = [size(X) 1];
A = X.lambda' * khatrirao(X.u,'r')';
A = reshape(A,sz);
