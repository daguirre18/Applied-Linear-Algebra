function [Q,R] = qrfactor(A) %call a function and it'll return Q and R 

[m,n] = size(A); %figure out size of A matrix
Q=eye(m); %make the Q matrix the identity
for k = 1:n %compute reflectors; 
            %starts at the 1st diagonal and makes everything below them 0
    % Find the HH reflector
    z = A(k:m,k); %pulling out the right columns and rows
    v = [ -sign(z(1))*norm(z) - z(1); -z(2:end) ]; %where to project
    v = v / sqrt(v'*v);   % remove v'*v  %normalization
    
    % Apply the HH reflection to each column of A and Q
    % updating the matrix by multiplying by Q_k 
    for j = 1:n
        A(k:m,j) = A(k:m,j) - v*( 2*(v'*A(k:m,j)) );
    end
    for j = 1:m
        Q(k:m,j) = Q(k:m,j) - v*( 2*(v'*Q(k:m,j)) );
    end
        
end

Q = Q'; %transpose
R = triu(A);  % exact triangularity; %this makes A perfectly triangular



