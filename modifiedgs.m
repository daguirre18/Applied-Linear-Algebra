function [Q,R] =  modifiedgs(X) %Modified Gram Schmidt 

[m,n] = size(X); %compute matrix size
 Q=zeros(m,n); %to preallocate matrix size
 R=zeros(n,n); %preallocate matrix size
  V=X;  %make V be the same as our starting matrix
     for j=1:n
          R(j,j) = norm(V(:,j)); %diagonals
          Q(:,j) = V(:,j)/R(j,j);
          for k=(j+1):n
               R(j,k)= Q(:,j)'*V(:,k); %update matrix
               V(:,k)=V(:,k)-R(j,k)*Q(:,j);
          end 
     end
     