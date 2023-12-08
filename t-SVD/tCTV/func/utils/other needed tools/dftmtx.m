function D = dftmtx(n)
validateattributes(n,{'numeric'},{'real','nonnegative','integer','scalar'},mfilename,'n',1);

D = fft(eye(n(1)));