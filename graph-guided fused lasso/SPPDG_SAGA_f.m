function [x,out] = SPPDG_SAGA_f(opts,L,A,b,K)
%
% SPPDG_SAGA_f is the code of SPPDG with SAGA proposed in the paper "Preconditioned Primal-Dual 
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

% set and check options
if nargin < 5
    opts        = []; 
end
opts            = set_options_nonc_f(opts,N,p,L);

%--------------------------------------------
% set some initial values and parameters 
%---------------------------------------------

% initial point
x               = opts.x0;
y               = opts.y0;

alpha        = opts.alpha;
ssize        = opts.ssize_g;

% maximum number iter
maxit           = opts.maxit;

% set step size
stepsize          = alpha;
lambda        = opts.lambda;

% prepare trace in output
if opts.trace
    [trace.obj,trace.time,trace.err,trace.iter] = deal(zeros(maxit,1));
end


itPrint          = opts.itPrint ;
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

% compute the initial gradient 

Grad_shifty = zeros(N,p);
G1=zeros(ssize,p);
for  s=1:N
     cache = (tanh(b(s)*(A(s,:)*x))^2-1)*b(s);  
     grad_shifty        = (cache*A(s,:));  
     Grad_shifty(s,:)  =  grad_shifty ;
end
 grad_shifty = mean(Grad_shifty)';

 mat = normest(K,2)^2;
 time_temp = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for  iter = 1:maxit

        rand_perm  = randperm(N, ssize);
          
   % compute sample gradient of loss   
        g1 = zeros(p,1);

   for  index  = 1:ssize
        index1 = rand_perm(index);
        cache1 =  (tanh(b(index1)*(A(index1,:)*x))^2-1)*b(index1);  
        g_temp = (cache1'*A(index1,:)) ;
        G1(index,:)     =    g_temp ;
        g1     =  g1+  g_temp'/ssize;  
   end       

       grad_shifty_sample = zeros(p,1);
       
   for index = 1:ssize
       index1 = rand_perm(index);
       grad_shifty_sample  = grad_shifty_sample+ Grad_shifty(index1,:)'/ssize;
   end 
    
       grad  =   g1 -  grad_shifty_sample +  grad_shifty;
   
    for  index = 1:ssize
         index1 = rand_perm(index);
         Grad_shifty(index1,:)  =  G1(index,:) ;             
    end
        grad_shifty = mean(Grad_shifty)';

  % get new iterate
    
    z        =  x - stepsize*(grad+K'*y);  

    y_temp   =  y+K*( 2*z-x )/(stepsize*mat); 
    for i = 1:p
       if  abs(y_temp(i))>lambda+1/(alpha*mat)
           y(i) =( y_temp(i)-sign( y_temp(i))/(alpha*mat));
       elseif  (abs(y_temp(i))<lambda+1/(alpha*mat))&&(abs(y_temp(i))>=lambda)
               y(i) =(sign( y_temp(i) )*lambda);          
       else 
           y(i) = y_temp(i);           
       end   

    end
   
    x             =  z;
      
    %---------------------------------------------------------------
    % get the value of objective
    %---------------------------------------------------------------
        tmp       = b.*(A*x);
        value_f   = 1 - mean(tanh(tmp));
        value_h   = lambda*sum(sqrt(K*x));

 % save information for graphic output
    if opts.trace 
        obj_full        = value_f+ value_h; 
        trace.obj(iter) = obj_full;
        trace.iter(iter)= iter;  
        trace.time(iter)= toc - time_temp;
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

   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GENERATE OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if opts.trace
    trace.obj       = trace.obj(1:iter);
    trace.time      = trace.time(1:iter);
    trace.iter      = trace.iter(1:iter);
    out.trace       = trace;
end

out.iter        = iter;
out.obj         = obj_full ;

    
end


