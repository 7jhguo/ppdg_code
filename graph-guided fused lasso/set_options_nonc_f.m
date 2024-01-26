function opts = set_options_nonc_f(opts,N,p,L)

% set initial point
if ~isfield(opts,'x0')       
    opts.x0  =zeros(p,1);  
end

if ~isfield(opts,'y0')       
    opts.y0  =zeros(N,1);  
end

if ~isfield(opts,'z0')       
    opts.z0  = zeros(N,1);  
end

% set maximum number of iterations
if ~isfield(opts,'maxit')    
    opts.maxit = 500;       
end


% step size parameters
kappa=opts.kappa;
if ~isfield(opts,'alpha')  
opts.alpha = (-(3+7*L+6*kappa)+sqrt((3+7*L+6*kappa)^2+32*(14*L^2+51*kappa)))/(10*(14*L^2+51*kappa)); 
end 

if ~isfield(opts,'lambda')
opts.lambda =1e-5; 
end 


% print output    
if ~isfield(opts,'itPrint')
    opts.itPrint = 1;      
end
if ~isfield(opts,'trace')     
    opts.trace  = 0;        
end
if ~isfield(opts,'record')
    opts.record = 1;        
end

end