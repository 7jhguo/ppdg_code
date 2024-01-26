% This code amis to show the performance of SPPDG with SAGA, SVRG, SARAH 
% for solving Nonconvex graph-guided fused lasso.
% 
%
% version: 2024/1/25

clear;
clc;
trace      = 1;
record     = 1;
fprintf    = 0;

% dataset
% prob_name       = {'gisette','MNIST','CINA'};
prob_name       = {'CINA'};

 if strcmp(prob_name,'CINA')   
    M  =4000;   
 end
 if strcmp(prob_name,'gisette')   
    M  =5000;   
 end
 if strcmp(prob_name,'MNIST')   
    M  =5000;   
 end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALGORITHMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


algorithm1.name       = 'SPPDG-SAGA';
algorithm1.print_name = 'SPPDG-SAGA';
algorithm1.opts       = struct('maxit',2*M,'itPrint',10,'trace',trace,'record',1);


algorithm2.name       = 'SPPDG-SVRG';
algorithm2.print_name = 'SPPDG-SVRG';
algorithm2.opts       = struct('maxit',M,'itPrint',10,'trace',trace,'record',1);


algorithm3.name       = 'SPPDG-SARAH';
algorithm3.print_name = 'SPPDG-SARAH';
algorithm3.opts       = struct('maxit',M,'itPrint',10,'trace',trace,'record',1);

% algorithm4.name       = 'ADMM-SAGA';
% algorithm4.print_name = 'SADMM-SAGA';
% algorithm4.opts       = struct('maxit',M,'itPrint',10,'trace',trace,'record',1);
% 
% algorithm8.name       = 'ADMM-SVRG';
% algorithm8.print_name = 'SADMM-SVRG';
% algorithm8.opts       = struct('maxit',M,'itPrint',10,'trace',trace,'record',1);
% 
% algorithm10.name       = 'ADMM-SARAH';
% algorithm10.print_name = 'SADMM-SARAH';
% algorithm10.opts       = struct('maxit',M,'itPrint',10,'trace',trace,'record',1);


algorithms = {algorithm1, algorithm2, algorithm3};  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of test repetitions
nr_test         =  1;

% test id
id_test         = 12;

% save data and results
add_str         = 'SPPDG_BASE';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TEST FRAMEWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(prob_name)
    
    % load dataset
    dataset_name    = prob_name{i};
    load(strcat('./datasets/',dataset_name,'/',dataset_name,'_train_scale.mat'));
    load(strcat('./datasets/',dataset_name,'/',dataset_name,'_labels.mat'));    

    [N,p]          = size(A);
 
    A               = [ones(N, 1) A];  %[N,p+1]
    p               = p + 1;
    L               = 4/(3*sqrt(3)); %Lipschitz constant
    
    %compute linear operator
    B = sum(A);
    B= repmat(B, N, 1);
    temp_metrix = A-1/N*B;
    K   =  1/N*(temp_metrix')*temp_metrix ;
 

    C   =  eye(N-p,p);
    K   =   [K;C];
    PER = 0.01; % sample size
       
    % generate and save graphic output
    if trace
        results = cell(length(algorithms),1);
    end
    
     %----------------------------------------------------------------------
     % Loop for different algorithms
     %----------------------------------------------------------------------
    for j = 1:length(algorithms) 
        
        %------------------------------------------------------------------
        % print algorithmic information and set specific parameters
        %------------------------------------------------------------------
        
      
        switch algorithms{j}.name
            case 'SPPDG-SAGA'
                
                 temp_nr_test    = nr_test; 
                 algorithms{j}.opts.ssize_g  = floor(PER*N); 
                 % kappa of SPPDG
                 algorithms{j}.opts.kappa   = (N^2)*(L^2)/(floor(PER*N)^2*N-floor(PER*N));             
       
             case 'SPPDG-SVRG'
                
                 temp_nr_test    = nr_test;                  
                 algorithms{j}.opts.ssize_g  = floor(PER*N); 
                 algorithms{j}.opts.kappa  = max(2*L^2/floor(PER*N), 4*L^2/(floor(PER*N)*N));
               
             case 'SPPDG-SARAH'
                 
                 temp_nr_test    = nr_test;              
                 algorithms{j}.opts.ssize_g  = floor(PER*N);
                 algorithms{j}.opts.kappa  = L^2/ floor(PER*N) + 2*L^2/ floor(PER*N);
               
             % case 'ADMM-SAGA'                                                                               
             % 
             % 
             %    temp_nr_test    = nr_test;
             % 
             %    if strcmp(algorithms{j}.print_name,'SADMM-SAGA')                      
             %    algorithms{j}.opts.ssize_g  = floor(PER*N); % 0.1N
             %    algorithms{j}.opts.kappa   = (N^2)*(L^2)/(floor(PER*N)^2*N-floor(PER*N));         
             %    end   
             % 
             %  case 'ADMM-SVRG'                                                                               
             % 
             % 
             %    temp_nr_test    = nr_test;
             % 
             %    if strcmp(algorithms{j}.print_name,'SADMM-SVRG')                      
             %    algorithms{j}.opts.ssize_g  = floor(PER*N); 
             %    algorithms{j}.opts.kappa  = max(2*L^2/floor(PER*N), 4*L^2/(floor(PER*N)*N));
             %    end  
             % 
             %   case 'ADMM-SARAH' 
             % 
             %    temp_nr_test    = nr_test;
             % 
             %     if strcmp(algorithms{j}.print_name,'SADMM-SARAH')                      
             %      algorithms{j}.opts.ssize_g  = floor(PER*N); 
             %      algorithms{j}.opts.kappa  = L^2/ floor(PER*N) + 2*L^2/ floor(PER*N);
             % 
             %     end   
                                  
        end
        
    
       % save algorithmic information
        if trace
            results{j}.name     = algorithms{j}.print_name;
           
        end 
        
        %------------------------------------------------------------------
        % Loop for number of tests and application of algorithms
        %------------------------------------------------------------------
        
        for k = 1:temp_nr_test 
           
            switch algorithms{j}.name
                case 'SPPDG-SAGA'
                    [x,out]  = SPPDG_SAGA_f(algorithms{j}.opts,L,A,b,K);
            
                case 'SPPDG-SVRG'
                    [x,out]  = SPPDG_SVRG_f(algorithms{j}.opts,L,A,b,K);
                    
                case 'SPPDG-SARAH'
                    [x,out]  = SPPDG_SARAH_f(algorithms{j}.opts,L,A,b,K); 
                    
                % case 'ADMM-SAGA'
                %     [x,out]  = ADMM_SAGA_f(algorithms{j}.opts,L,A,b,K);
                % 
                % case 'ADMM-SVRG'
                %     [x,out]  = ADMM_SVRG_f(algorithms{j}.opts,L,A,b,K);
                % 
                % case 'ADMM-SARAH'
                %     [x,out]  = ADMM_SARAH_f(algorithms{j}.opts,L,A,b,K);
     
            end
            
            % save results for graphic output
            
             if (trace && k == 1)
                results{j}.rel_err = out.trace.obj; 
                results{j}.time    = out.trace.time; 
                %results{j}.err    = out.trace.err;
            elseif trace
                results{j}.rel_err = results{j}.rel_err + out.trace.obj; 
                results{j}.time    = results{j}.time + out.trace.time;  
               % results{j}.err    = results{j}.err + out.trace.err;  
             end          
        end
        
        % Compute average of the results
        if trace
            results{j}.obj     = results{j}.rel_err/temp_nr_test;  
            results{j}.time    = results{j}.time/temp_nr_test;
            %results{j}.err     = results{j}.err/temp_nr_test;
        end

        
         if trace
            results{j}.iter    = out.trace.iter;
         end
         
     
    end
        
    if trace                   
        if isempty(add_str)
            save(strcat('results/results_nonc_sigmoid_',dataset_name,'_id_',num2str(id_test),'.mat'),'results');
        else
            save(strcat('results/results_nonc_sigmoid_',add_str,'_',dataset_name,'_id_',num2str(id_test),'.mat'),'results');
        end  
    end
    
end


%% Draw figures
if trace

if length(algorithms) == 3
    add_str         = 'SPPDG_BASE';
    dataset_name    = prob_name{i};
    id              = num2str(id_test);    
    draw_figures_results_m(add_str,dataset_name,id)
end
end

