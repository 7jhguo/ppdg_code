function [paramx, options] = SPG_SVRG(model,X, y,X_test, y_test, options)
% SPG_SVRG is a code for image classification on 1-hidden layer network
% using SPG with SVRG.
%
%
% input & ouput: 
%       model           ---- neural network model
%       X,y             ---- training data: input (d x n), output (c x n)
%       X_test,y_test   ---- test data
%       lambda          ---- l1 regularization
% .     options:
%           options.params:     weights of the model
%           options.tr_times:   training timestamp at each iteration
%           options.tr_losses:  training loss at each iteration
%           options.tr_grads:   training gradient norm at each iteration
%           opitons.tr_errs:    training error at each iteration
%           options.te_errs:    test error at each iteration
%           options.cur_ter:    current itertion number(if not 0, resume
%                               training
%           options.maxNoProps: maximum propagations for training
%           options.max_iters:  maximum iterations for training
%
%
%  2024/1/25

layersizes = model.layersizes;
numlayers = model.numlayers;
noProps = 1;
n = size(X,2);
sz = floor(0.05*n);
psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1)));
alpha = 0.05;

cur = 0;
cur_time = 0;


if isfield(options, 'lambda')
    lambda  = options.lambda;
end 

if isfield(options, 'maxNoProps')
    maxNoProps  = options.maxNoProps;
end 


if isfield(options,'max_iters')
  max_iters = options.max_iters;
end

if isfield(options,'cur_iter') && options.cur_iter >= 1
    cur = options.cur_iter;
    cur_time = options.tr_times(cur);
    options.tr_times = [options.tr_times(1:cur); zeros(max_iters,1)];
    options.tr_losses = [options.tr_losses(1:cur); zeros(max_iters,1)];
    options.tr_grad = [options.tr_grad(1:cur); zeros(max_iters,1)];
    options.tr_errs = [options.tr_errs(1:cur); zeros(max_iters,1)];
    options.tr_noProps = [options.tr_noProps(1:cur); zeros(max_iters,1)];
    
    options.te_losses = [options.te_losses(1:cur); zeros(max_iters,1)];
    options.te_errs = [options.te_errs(1:cur); zeros(max_iters,1)];
    noProps = options.tr_noProps(cur);
    %maxNoProps = maxNoProps + noProps;
    
else
    options.tr_errs = zeros(max_iters,1);
    options.tr_losses = zeros(max_iters, 1);
    options.tr_grad = zeros(max_iters, 1);
    options.tr_times = zeros(max_iters, 1);
    options.tr_noProps = zeros(max_iters, 1);
    
    options.te_errs = zeros(max_iters,1);
    options.te_losses = zeros(max_iters, 1);    
end

if ~isfield(options,'record');      options.record       = 1;           end
record    = options.record;
print_it  = 50;


% initialize parameters
fprintf('initial setup:\n');
if isfield(options,'params')
  paramx = options.params;
else
  paramx = sprandn(psize,1,0.1)*0.5; 
end

grad = zeros(size(paramx));
fprintf(' batch size: %d\n step size: %f\n  max props: %d\n\n',...
    sz, alpha,  maxNoProps);

extra_time = 0;
tic;

% training
fprintf('\n start training...\n');

m   =  2;


%% main loop
for iter = cur+1: cur + max_iters

   if noProps > maxNoProps
        iter = iter - 1;
        break;
   end  
   
   if record
   temp_time = toc;
   ll_err = compute_model(model, paramx, X,y);
   ll = ll_err(1); tr_err = ll_err(2);
   tr_loss = ll+ lambda * norm(paramx,1);
   
   te_loss_err = compute_model(model, paramx, X_test, y_test);
   te_loss = te_loss_err(1); te_err = te_loss_err(2);
   
   options.tr_losses(iter) = tr_loss;
   options.tr_errs(iter) = tr_err;
   options.te_losses(iter) = te_loss;
   options.te_errs(iter) = te_err;
   options.tr_grad(iter) = norm(grad,Inf);

   options.tr_noProps(iter) = noProps;
   if iter == cur + 1 || mod(iter,print_it) == 0
   fprintf('training loss + reg: %f, grad: %f(max), %f(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
   fprintf('training err: %f\n', tr_err);
   fprintf('test loss: %f, test err: %f\n', te_loss, te_err);
   fprintf(' total Props: %g\n', noProps);
   end
   extra_time = extra_time + toc - temp_time;
   end
   options.tr_times(iter) = toc - extra_time + cur_time ;
   if mod(iter,print_it) == 0
   fprintf('\nIter: %d, time = %f s\n', iter, options.tr_times(iter));
   end
   
   paramx_ =  paramx;
   
   % full gradient 
   [~, grad0] = compute_model(model, paramx_, X, y); 
   fullgrad  = grad0;
   noProps = noProps + size(X,2);
   paramx_0    =  paramx_;
   
  %% inner loop   
  for t =1:m  
   idx   = randsample(n, sz);  
   x_sample = X(:,idx);
   y_sample = y(:,idx); 

   % sample gradient
   [~, grad1] = compute_model(model, paramx_0, x_sample, y_sample);
   
   noProps = noProps + size(x_sample,2);
   [~, grad2] = compute_model(model, paramx_, x_sample, y_sample);
   
   noProps = noProps + size(x_sample,2);
   grad  = grad1-grad2+fullgrad;
   
   
   %% 
   paramy = paramx_0- alpha *grad; 
   paramx_0 = 1/max(1,norm(paramy,inf)/lambda)*paramy;
 
  
 end   
  paramx = paramx_0;
  
end

if record
options.params = paramx;
options.cur_iter = iter;
options.tr_times = options.tr_times(1:iter);
options.tr_losses = options.tr_losses(1:iter);
options.tr_errs = options.tr_errs(1:iter);
options.tr_grad = options.tr_grad(1:iter);
options.te_losses = options.te_losses(1:iter);
options.te_errs = options.te_errs(1:iter);
options.tr_noProps = options.tr_noProps(1:iter);
options.method = 'SPG-SVRG';
else
   ll_err = compute_model(model, paramx, X,y);
   ll = ll_err(1); tr_err = ll_err(2);
   tr_loss = ll + lambda * norm(paramx,1);
   
   te_loss_err = compute_model(model, paramx, X_test, y_test);
   te_loss = te_loss_err(1); te_err = te_loss_err(2);
   
   fprintf('training loss + reg: %f, grad: %f(max), %f(norm)\n', tr_loss, norm(grad,Inf), norm(grad,2));
   fprintf('training err: %f\n', tr_err);
   fprintf('test loss: %f, test err: %f\n', te_loss, te_err);
   fprintf('total Props: %g\n', noProps);
end
end
