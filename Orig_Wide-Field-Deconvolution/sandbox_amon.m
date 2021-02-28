% T = 5;
% 
% L1 = [zeros(T,1) [zeros(1,T-1); [zeros(1,T-1); [-eye(T-2), zeros(T-2,1)] + [zeros(T-2,1), eye(T-2)]]]];
% 
% 
% test = [zeros(T,1) [zeros(1,T-1)]];
% 
% 
% z = [zeros(T,1); 5];
% z(z==0) = 2;


h5disp('data_Jonas.hdf5')
data = h5read('data_Jonas.hdf5', '/DF_by_FO');
ROI = h5read('data_Jonas.hdf5', '/ROI');

% mean(data)