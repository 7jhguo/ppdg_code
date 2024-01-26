function [options,file_name] = classification(method, dataset_name, record,  init, maxNP, seed)
% cifar10 classification using 1-hidden layer network.
%
% Input:
%       method      ---- SPPDG, SADMM, SPG with gradient estimators SAGA, SVRG, SARAH
%       init        ---- initialization schemes
%                       0: zeros initialization
%                       1: normalized random initialization
%                       2: random intialization
%       maxNP       ---- maximum propagations
%
% Output:
%       results     ---- contain all the training information and results;
%                        see each algorithm function for details.
%
%           options.params:     weights of the model
%           options.tr_times:   training timestamp at each iteration
%           options.tr_losses:  training loss at each iteration
%           options.tr_grads:   training gradient norm at each iteration
%           opitons.tr_errs:    training error at each iteration
%           options.te_errs:    test error at each iteration
%           options.cur_ter:    current itertion number(if not 0, resume
%                               training    
%
%  2024/1/25  

%%
resume_train = 0;
if nargin < 6
    seed = 0;
end
if nargin < 5
    maxNP = 1e8;
end
if nargin < 4
    init = 1;
end

if nargin < 3
    record = 1; 
end

addpath(genpath(pwd));% print work directory 
% seed = 1234;
% randn('state', seed );
% rand('twister', seed+1 );
rng(seed);

%% Datasets

%dataset_name = 'cifar-10'; % n = 50000; input = 3072; output = 10;

if strcmpi(dataset_name, 'cifar-10')
%% Read the data --- X: d x n, y: c x n
load('./datasets/cifar-10-batches-mat/data_batch_1.mat'); data1 = data; labels1 = labels;
load('./datasets/cifar-10-batches-mat/data_batch_2.mat'); data2 = data; labels2 = labels;
load('./datasets/cifar-10-batches-mat/data_batch_3.mat'); data3 = data; labels3 = labels;
load('./datasets/cifar-10-batches-mat/data_batch_4.mat'); data4 = data; labels4 = labels;
load('./datasets/cifar-10-batches-mat/data_batch_5.mat'); data5 = data; labels5 = labels;
X = double([data1; data2; data3; data4; data5])';
y = getClassLabels([labels1; labels2; labels3; labels4; labels5])';

load('./datasets/cifar-10-batches-mat/test_batch.mat'); X_test = double(data)'; y_test = getClassLabels(labels)';
[inputd, n] = size(X);
outputd = size(y,1);
%% Preprocess the images
mean_data = mean(X, 2);
std_data = std(X,0,2);
std_data(std_data==0) = 1;
X = bsxfun(@minus, X, mean_data);
X = bsxfun(@rdivide,X, std_data);
X_test = bsxfun(@minus, X_test, mean_data);
X_test = bsxfun(@rdivide,X_test, std_data);
%hiddend = 512;
else
 [X, y, n, X_test, y_test, ~, outputd, inputd, ~, ~] = ...
    load_dataset(dataset_name, './datasets/',  inf, inf, false); 
X = X'; X_test = X_test'; y = y'; y_test = y_test'; 
end

hiddend = floor(sqrt(outputd*inputd));

%% Specify the Neural Network Model
model.layersizes = [inputd, hiddend, outputd];
model.layertypes = {'logistic','softmax'};
model.numlayers = length(model.layertypes);model.type = 'classification';
psize = model.layersizes(1,2:(model.numlayers+1))*model.layersizes(1,1:model.numlayers)' + sum(model.layersizes(2:(model.numlayers+1)));
model.psize = psize;

str = '%11s';
stra = ['\n',str,str,str,str,str,'\n'];
str_head = sprintf(stra, 'Dataset', 'input_d', 'hidden_d', 'output_d', 'param_d');
fprintf('--------------------------------------------------------\n');
fprintf('%s',str_head);
str_num = ['%11s %9d %10d %9d %11d\n'];
fprintf(str_num, dataset_name, inputd, hiddend, outputd, psize);
fprintf('--------------------------------------------------------\n\n');
%% Initialize the Model

if init == 0
    initial_guess = zeros(psize,1); sub_dir = ['/zeros_', num2str(seed)];
    fprintf('\n\nZero Initialization! \n\n');
elseif init == 1 % Xavier initial
    initial_guess = sqrt(3/(inputd+outputd)).*randn(psize,1); initial_guess = initial_guess/norm(initial_guess); sub_dir = ['/randn_normalized_', num2str(seed)];
    fprintf('\n\nNormalized Random Initialization! \n\n');
else
    initial_guess = randn(psize,1); sub_dir = ['/randn_', num2str(seed)];
    fprintf('\n\nRandom Initialization! \n\n');
end
options.params = initial_guess;

%% Specify the algorithm
options.name = strcat(dataset_name,'_classification');
options.inner_iters = 250;
options.cur_iter = 0;
options.maxNoProps = maxNP;
options.record = record;
options.max_iters = 20000; 
options.lambda = 1e-4; % l1 regularization     
 
%%
dir_name = ['./results/',options.name,sub_dir];
if ~exist(dir_name, 'dir')
    mkdir(dir_name);
end

file_name = [dir_name,'/',options.name,'_options.lambda_', num2str(options.lambda)];

%% Start Training
switch method
        
    case 'SPG-SARAH'
        fprintf('\n\n------------------- SPG_SARAH ----------------\n\n');
        file_name_spg_sarah = [file_name,'_spg_sarah.mat'];
        if exist(file_name_spg_sarah, 'file') && resume_train
            load(file_name_spg_sarah,'options'); % resume training
        end
        [params, options] = SPG_SARAH(model,X,y,X_test,y_test,options);
        
        parsave(file_name_spg_sarah, options);
        
   case 'SPG-SAGA'
        fprintf('\n\n------------------- SPG_SAGA ----------------\n\n');
        file_name_spg_saga = [file_name,'_spg_saga.mat'];
        if exist(file_name_spg_saga, 'file') && resume_train
            load(file_name_spg_saga,'options'); % resume training
        end
        [params, options] = SPG_SAGA(model,X,y,X_test,y_test,options);
        
        parsave(file_name_spg_saga, options);   

    case 'SPG-SVRG'
        fprintf('\n\n------------------- SPG_SVRG ----------------\n\n');
        file_name_spg_svrg = [file_name,'_spg_svrg.mat'];
        if exist(file_name_spg_svrg, 'file') && resume_train
            load(file_name_spg_svrg,'options'); % resume training
        end
        [params, options] = SPG_SVRG(model,X,y,X_test,y_test,options);
        parsave(file_name_spg_svrg, options); 
    case 'SPPDG-SVRG'
        fprintf('\n\n------------------- PPDG_SVRG ----------------\n\n');
        file_name_ppdg_svrg = [file_name,'_ppdg_svrg.mat'];
        if exist(file_name_ppdg_svrg, 'file') && resume_train
            load(file_name_ppdg_svrg,'options'); % resume training
        end
        [params, options] = PPDG_SVRG(model,X,y,X_test,y_test,options);
        parsave(file_name_ppdg_svrg, options);
    case 'SPPDG-SARAH'
        fprintf('\n\n------------------- PPDG_SARAH ----------------\n\n');
        file_name_ppdg_sarah = [file_name,'_ppdg_sarah.mat'];
        if exist(file_name_ppdg_sarah, 'file') && resume_train
            load(file_name_ppdg_sarah,'options'); % resume training
        end
        [params, options] = PPDG_SARAH(model,X,y,X_test,y_test,options);
        parsave(file_name_ppdg_sarah, options);
    case 'SPPDG-SAGA'
        fprintf('\n\n------------------- PPDG_SAGA ----------------\n\n');
        file_name_ppdg_saga = [file_name,'_ppdg_saga.mat'];
        if exist(file_name_ppdg_saga, 'file') && resume_train
            load(file_name_ppdg_saga,'options'); % resume training
        end
        [params, options] = PPDG_SAGA(model,X,y,X_test,y_test,options);
        parsave(file_name_ppdg_saga, options);
        
     case 'SADMM-SAGA'
        fprintf('\n\n------------------- ADMM_SAGA ----------------\n\n');
        file_name_admm_saga = [file_name,'_admm_saga.mat'];
        if exist(file_name_admm_saga, 'file') && resume_train
            load(file_name_admm_saga,'options'); % resume training
        end
        [params, options] = ADMM_SAGA(model,X,y,X_test,y_test,options);
        parsave(file_name_admm_saga, options);
     case 'SADMM-SVRG'
        fprintf('\n\n------------------- ADMM_SVRG ----------------\n\n');
        file_name_admm_svrg = [file_name,'_admm_svrg.mat'];
        if exist(file_name_admm_svrg, 'file') && resume_train
            load(file_name_admm_svrg,'options'); % resume training
        end
        [params, options] = ADMM_SVRG(model,X,y,X_test,y_test,options);
        parsave(file_name_admm_svrg, options);   
    case 'SADMM-SARAH'
        fprintf('\n\n------------------- ADMM_SARAH ----------------\n\n');
        file_name_admm_sarah = [file_name,'_admm_sarah.mat'];
        if exist(file_name_admm_sarah, 'file') && resume_train
            load(file_name_admm_sarah,'options'); % resume training
        end
        [params, options] = ADMM_SARAH(model,X,y,X_test,y_test,options);
        parsave(file_name_admm_sarah, options);
        
            
end

end


