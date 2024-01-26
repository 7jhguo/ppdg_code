function [x,out] = SPPDG_SVRG_f(opts,L,A,b,K)
%
% SPPDG_SVRG_f is the code of SPPDG with SVRG proposed in the paper "Preconditioned Primal-Dual 
% Gradient Methods for Nonconvex Composite and Finite-Sum Optimization" for solving 
% Nonconvex graph-guided fused lasso.
%
%
%           L:                   Lipschitz constant
%           A,b:                 data 
 %          K:                   linear operator       
%       opts: 
%           opts.maxit:          maximum iterations
%           opts.ssize_g:        number of sample
%           opts.alpha:          step size
%           opts.lambda:         regularization parameter   
%
%
%  2024/1/25


tic;
[N,p]          = size(A);
 
 
if nargin < 5
    opts        = []; 
end
opts            = set_options_nonc_f(opts,N,p,L);

% initial point
x                = opts.x0;
y                = opts.y0;
alpha            = opts.alpha;

% maximum number iter
maxit           = opts.maxit;

% set step size
stepsize          = alpha;
lambda        = opts.lambda;

% prepare trace in output
if opts.trace
    [trace.obj,trace.time,trace.err,trace.iter] = deal(zeros(maxit,1));
end

% parameters for sample set generation
 ssize           = opts.ssize_g; 

 itPrint         = opts.itPrint ;
 record           = opts.record;  
if record >= 1
    if opts.trace
    % set up print format
    stra = ['%6s', '%7s', '%9s', '%8s', '\n', ...
        repmat( '-', 1, 50 ), '\n'];
    str_head = sprintf(stra, 'iter', 'obj', 'err','time');
    str_num = ('%4d |  %2.1e  %+2.1e  %4.1f  \n'); 
    else
    % set up print format
    stra = ['%6s', '%9s', '\n', ...
        repmat( '-', 1, 30 ), '\n'];
    str_head = sprintf(stra, 'iter','err');
    str_num = ('%4d |  %+2.1e \n');         
    end
end

mat = normest(K,2)^2;    
 
time_temp = toc;
m   =  2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:maxit        
     x_    =  x; 
     
     %  full gradient of loss
     cache1 = (tanh(b.*(A*x_)).^2-1).*b;  
     g1        = (cache1'*A)'/N;
      
        x_0      =  x_;
    for t =1:m

        % sample
          rand_perm = randperm(N);
          A_perm_g = A(rand_perm,:);
          b_perm_g = b(rand_perm);
         
          A_sample_g    = A_perm_g(1:ssize,:);
          b_sample_g    = b_perm_g(1:ssize);

          
          % compute sample gradient 
          cache2    =  (tanh(b_sample_g.*(A_sample_g*x_0)).^2-1).*b_sample_g;  
          g2        = (cache2'*A_sample_g)'/ssize;         
          cache3    = (tanh(b_sample_g.*(A_sample_g*x_)).^2-1).*b_sample_g;  
          g3        =  (cache3'*A_sample_g)'/ssize;
          % svrg gradient
          grad      = g2 - g3 + g1;
          
          % get new iterate
           z       =  x_0 - stepsize*(grad+K'*y);  
          y_temp   =  y+K*( 2*z-x )/(stepsize*mat); 
    for i = 1:p
       if  abs(y_temp(i))>lambda+1/(alpha*mat)
           y(i) =  y_temp(i)-sign( y_temp(i))/(alpha*mat);
       elseif  (abs(y_temp(i))<lambda+1/(alpha*mat))&&(abs(y_temp(i))>=lambda)
               y(i) = sign( y_temp(i) )*lambda;          
       else 
           y(i) = y_temp(i);           
       end   

    end
          x_0         =  z;
    end
        x             =  z ;
        % objective value
        tmp        = b.*(A*x);
        value_f   = 1 - mean(tanh(tmp));
        value_h  = lambda*sum(sqrt(K*x));
      
        % save information for graphic output
    if opts.trace 
        obj_full        = value_f+ value_h;  
        trace.obj(iter)     = obj_full;
        trace.iter(iter)     = iter;  
        trace.time(iter)   = toc - time_temp;
    end  
              
    %----------------------------------------------------------------------
    % print iteration info
    %---------------------------------------------------------------------- 
     if (record>=1 && ( ...
            iter == 1 || iter==maxit || mod(iter,itPrint)==0) && iter >= 1)
        if (iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit) 
            fprintf('\n%s\n', str_head);
        end
        if opts.trace
            fprintf(str_num, iter,trace.obj(iter), trace.err(iter),trace.time(iter));
        end
     end
     
end


if opts.trace
    trace.obj       = trace.obj(1:iter);
    trace.time      = trace.time(1:iter);
    trace.iter      = trace.iter(1:iter);
    out.trace       = trace;
end

out.iter        = iter;
out.obj     = obj_full ;

    
end
