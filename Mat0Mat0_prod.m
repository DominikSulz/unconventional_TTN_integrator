function [M] = Mat0Mat0_prod(U,V,a,b,d)

M = Mat0Mat0(U,V);
M = (b-a)^d * M;

end

