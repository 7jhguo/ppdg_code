function  draw_figures_results(file_name,results)

nr_algors           = length(results);
temp_results        = cell(nr_algors,1);
for i = 1:nr_algors
   temp_results{i}  = results{i}; 
end
       
interval     = 100000; markersize = 6; 

names = cell(nr_algors,1);

for i = 1:nr_algors
    names{i}    = temp_results{i}.method;
end

markers_t = {'-','-','-','--','--','--','-.','-.','-.'};
colors_t = {  [0,0,0]/255,... 
          [255,71,71]/255,...  
          [17,140,17]/255,... 
          [0.9,0.7,0.0], ...   
          [119,55,0]/255,... 
            [0,101,189]/255,...       
          [0.4940 0.1840 0.5560], ...  
          [67,20,97]/255, ...           
         [0.8500 0.3250 0.0980], ...            
          [218,215,203]/255};            
colors   = cell(1,nr_algors); markers   = cell(nr_algors,1);
if nr_algors > 10; error('The number of algorithms is too many!'); end

for i = 1:nr_algors
  colors{i} = colors_t{i}; markers{i} = markers_t{i};
end

%%{
%% print seperate pics

%% plot: Props // train_loss
figure;
clf
for i = 1:nr_algors
semilogx_marker(temp_results{i}.tr_noProps,temp_results{i}.tr_losses,markers{i},interval,markersize,colors{i});
end
xlim([0 1e8]);
%ylim([1.5 3]);
legend(names,'Location','SouthWest');
xlabel('# of Props'); ylabel('Training Loss');
set(gca,'FontSize',12);
grid on;
%set(gca,'YMinorGrid','off','YMniorTick','off');
print('-depsc',strcat(file_name,'_train_loss_props.eps'));
%% plot: Props // test_error
figure;
clf
for i = 1:nr_algors
semilogx_marker(temp_results{i}.tr_noProps,temp_results{i}.te_errs,markers{i},interval,markersize,colors{i});
end
xlim([0 1e8]);
legend(names,'Location','SouthWest');
xlabel('# of Props'); ylabel('Test Error');
set(gca,'FontSize',12);
grid on;
%set(gca,'YMinorGrid','off','YMniorTick','off');
print('-depsc',strcat(file_name,'_test_err_props.eps'));


%% plot:  Props-- train error

figure;
clf
for i = 1:nr_algors
semilogx_marker(temp_results{i}.tr_noProps,temp_results{i}.tr_errs,markers{i},interval,markersize,colors{i});
end
xlim([0 1e8]);
legend(names,'Location','SouthWest');
xlabel('# of Props'); ylabel('Training Error');
set(gca,'FontSize',12);
grid on;
%set(gca,'YMinorGrid','off','YMniorTick','off');
print('-depsc',strcat(file_name,'_train_err_props.eps'));



