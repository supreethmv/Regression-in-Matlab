function DesignMatrix = Basis(X,k)
%BASIS Summary of this function goes here
% X n×1 from training data
% DesignMatrix nx(2k+1) is mapping of X using the Fourier basis functions

DesignMatrix = zeros(size(X,1),2*k+1);
DesignMatrix(:,1) = 1;
K_index = 2:2:2*k;
DesignMatrix(:,K_index) = 2./K_index.*cos(pi*K_index.*X); % Even frquency Components
DesignMatrix(:,K_index+1) = 2./K_index.*sin(pi*K_index.*X); % Odd frequency Components



end

