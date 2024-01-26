% This code amis to compare the performance of SPPDG, SADMM, SPG with SAGA,
% SVRG, SARAH for solving image classification on 1-hidden layer network.
% 
%
% version: 2024/1/25

clc;
record = 1;
datasets = {'cifar-10'};
%methods = {'SPPDG-SVRG'};
methods = {'SPPDG-SVRG','SPPDG-SARAH','SPPDG-SAGA','SADMM-SVRG','SADMM-SARAH','SADMM-SAGA','SPG-SVRG','SPG-SARAH','SPG-SAGA'};
for k = 1:length(datasets)
dataset_name  = datasets{k};
    results  = cell(length(methods));
    for i = 1:length(methods)
        [results{i},file_name] = classification(methods{i},dataset_name,record);
    end
 %% Plot the results
 if record
 draw_figures_results(file_name,results);
 end
end
