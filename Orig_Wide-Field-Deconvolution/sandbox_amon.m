T = 5;

L1 = [zeros(T,1) [zeros(1,T-1); [zeros(1,T-1); [-eye(T-2), zeros(T-2,1)] + [zeros(T-2,1), eye(T-2)]]]];


%test = [zeros(T,1) [zeros(1,T-1)]];


z = [zeros(T,1); 5];
z(z==0) = 2;