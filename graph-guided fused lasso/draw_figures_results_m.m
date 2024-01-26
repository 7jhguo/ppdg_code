function  draw_figures_results_m(add_str,dataset_name,id)

% read data
temp_results        = cell(3,1);
load(strcat('results/','results_','nonc_sigmoid_',add_str,'_',dataset_name,'_id_',id,'.mat'));
temp_results{1}     = results{1}; 
temp_results{2}     = results{2};
temp_results{3}     = results{3}; 


res_name            = dataset_name;
res_prob            = 'nonconv_sigmoid_BASE';
res_id              = 12;

switch res_name
    case 'gisette'
        y_min_trel   = 0.05;    y_max_trel   = 0.15;
        y_min_erel   = 0.05;    y_max_erel   = 0.15; 
        x_max_trel   = 5000;      
        x_max_erel   = 1600;         
        intervals   = [50000000,5000000,5000000,200000,500000,500000];
    case 'MNIST'
       y_min_trel   = 0.3;    y_max_trel   = 0.5;
       y_min_erel   = 0.3;    y_max_erel   = 0.5; 
       x_max_trel   = 3500;       
       x_max_erel   = 1500;
       intervals   = [500000,2000000,2000000,1000000,1000000,1000000];
    case 'covtype'
       y_min_trel   = 0.55;   y_max_trel   = 0.85;
       y_min_erel   = 0.55;      y_max_erel   = 0.85;
       x_max_trel   = 10000;       
       x_max_erel   = 4000;
       intervals   = [100000,100000,100000,100000,1000000,100000];        
    case 'CINA'
        y_min_trel   = 0.16;    y_max_trel   = 0.24;
        y_min_erel   = 0.16;    y_max_erel   =0.24; 
        x_max_trel   =4000;      
        x_max_erel   =30;         
        intervals   = [40000,20000,400000,400000,400000,4000000];
end

        
markers     = {'-','-','-','--','--','--','>-','s-','v-'};
names = cell(3,1);

for i = 1:3
    names{i}    = temp_results{i}.name;
end

colors = {[0,0,0]/255,...     
          [255,71,71]/255,...  
          [17,140,17]/255,... 
          [67,20,97]/255, ...       
          [119,55,0]/255,...       
          [17,140,17]/255,...      
          [0.9,0.7,0.0], ...       
          [0,101,189]/255,...      
          [218,215,203]/255,...     
          [255,71,71]/255,...   
          [0,0,0]/255};           



%--------------------------------------------------------------------------
% plot: iter// value
%--------------------------------------------------------------------------

figure;
clf
for i = 1:3
    semilogy_marker(temp_results{i}.iter,temp_results{i}.obj,markers{i},intervals(i),10,colors{i});
end

legend(names);
ylim([y_min_erel y_max_erel]);
xlim([1 x_max_trel]);

xlabel('Iteration');
ylabel('Objective value');

set(gca,'FontSize',12);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');

print('-depsc',strcat('./results/results_',res_prob,'_',res_name,'_iter_value_id_',num2str(res_id),'.eps'));

%--------------------------------------------------------------------------
% plot: time// value
%--------------------------------------------------------------------------
figure;
clf
for i = 1:3
    semilogy_marker(temp_results{i}.time,temp_results{i}.obj,markers{i},intervals(i),10,colors{i});
end

legend(names);
ylim([y_min_trel y_max_trel]);
xlim([0 x_max_erel])

xlabel('Time');
ylabel('Objective value');

set(gca,'FontSize',12);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');

print('-depsc',strcat('./results/results_',res_prob,'_',res_name,'_time_value_id_',num2str(res_id),'.eps'));

