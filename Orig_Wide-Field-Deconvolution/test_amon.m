% This script follows all steps needed to deconvolve Df/f fluorescence data
% using the continuously-varying algorithm

% the data size is assume to be Txn with T - time points in each trial,
% n - number of trials

% to use this script with a different method simply switch the convar
% function to dynbin/lucric/firdif

% the script also uses the single neuron decay constant (which is the
% ratio of between the fluorescence at the next time step, compare to the 
% current time step when no spikes occur) 
% we analyze in the manuscript parallel recordings of wide-field and 
% spiking rate and find very similar decay constant in the wide-field

% load data
load('Clancy_etal_fluorescence_example.mat')
% a more convenient (and faster) scaling to work with
cal_data = cal_data*100;
% calcium decay rate (single neuron, based on 40Hz mesearmunts in Gcamp6f mice) 
gamma_40hz = 0.97;

% To eventually use the continuously-varying algorithm, first the best 
% parameter for the algorithm, for the data at hand needs to be found. 
% To this end the "odd" trace is deconvolved and the
% calcium reconstructed from it is compared to the "even" trace
% hence we divide the data
odd_traces = cal_data(1:2:end-1,:);
even_traces = cal_data(2:2:end,:);
% the calcium decay is needed to be fitted for 20hz of the eve/odd traces
ratio = 0.5;
gamma = 1-(1-gamma_40hz)/ratio;      
% number of points in each odd/even calcium trace
T = size(odd_traces,1);
% number of calcium traces
rep = size(cal_data,2);
        
% serach over a range of lambda/smoothing values to find the best one
% all_lambda = [20 10 7 5 3 2 1 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05];
all_lambda = [1];


% will be used later to reconstruct the calcium from the deconvoled rates
Dinv = zeros(T,T); 
insert_vec = 1;
for k = 1:T
    Dinv(k,1:k) = insert_vec;
    insert_vec = [gamma^k, insert_vec];
end

% saving the results
% here the penalty (l2) is the same as the fluctuations (l2)
penalty_size_convar = zeros(length(all_lambda),rep);
calcium_dif_convar = zeros(length(all_lambda),rep);
tic
for k = 1:length(all_lambda)
    lambda = all_lambda(k); 
    [r_firdif] = test_firdif_amon(odd_traces,gamma,3); 
%     [r_unconstr, r1,beta0] = convar_analytic_unconstraint(odd_traces,gamma,lambda); 
%     [r, r1,beta0] = convar(odd_traces,gamma,lambda); 

end

toc