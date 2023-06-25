addpath(genpath('solvers'));
addpath(genpath('test_image'));



p = 0.7;
indimgs = [1:5];
for i = 1 : length(indimgs)
    %% read data and produce mask
    id = indimgs(i);
    pic_name = [ './test_image/',num2str(id),'.jpg'];
    I = double(imread(pic_name));
    X = I/255;
    [n1,n2,n3] = size(X);
    maxP = max(X(:));
    Omega = find(rand(n1*n2*n3,1)<p);
    
    
    %% tensor completion
    [ TC,psnr ] = TCTF( X,Omega );
    
    fprintf('%d-th image PSNR:%.2f\n',i,psnr);
end