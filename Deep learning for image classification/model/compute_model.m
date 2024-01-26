function [loss,grad,perr] = compute_model(model,params,X,y)
%% Evaluting neural network models: loss, grad, error
% Input:
%       model       ---- specification neural network model
%         model.numlayers: number of layers
%         model.layersizes: number of neurons of each layer
%         model.layertypes: type of each layer (currently only support
%                           logisitic(sigmoid),tanh,linear,softmax)
%       params      ---- weights
%       X,y         ---- input(d x n),output(c x n), d,c should be
%                        consistent with model.layersizes.
%
% Output:
%       loss        ---- loss 
%       grad        ---- gradient, same size as params
%       perr        ---- learning error,e.g. classification error or mse.
%    if only returning 1 output, it returns [loss, learning error]
%     i.e. res = compute_model(model,params,X,y) --> res = [loss, err] 
%    if returning 2 outputs,  it returns [[loss,err], gradient]
%     i.e. [res,grad] = compute_model(model,params,X,y) -> res = [loss, err]
%    
% written by Peng Xu, Fred Roosta, 6/8/2017, updated(2/8/2018)

n = size(X,2);
numlayers = model.numlayers;
layertypes = model.layertypes;
layersizes = model.layersizes;
psize = length(params);
dW = cell(numlayers, 1);
db = cell(numlayers, 1);



[W,b] = unpack(params, numlayers, layersizes);% allocation

xi = X;
z = cell(numlayers+1,1);
dx = cell(numlayers,1);
z{1} = X;

ll = 0;

%% loss
for k = 1:numlayers
    xi = bsxfun(@plus, W{k}*z{k}, b{k});
    if strcmp(layertypes{k}, 'logistic')
        zi = 1./(1 + exp(-xi));
    elseif strcmp(layertypes{k}, 'tanh')
        zi = tanh(xi);
    elseif strcmp(layertypes{k}, 'linear')
        zi = xi; 
    elseif strcmp(layertypes{k}, 'softmax')
        zi = softmax(xi);
     %   disp(size(zi))
    else
        error('Unknow layer type');
    end
    z{k+1} = zi;
end
%fprintf('%d\n', zi(:,1)) 

if strcmp(layertypes{numlayers}, 'linear')
   
    ll = ll - sum(sum((y - zi).^2));
    err = 2*(y-zi);
elseif strcmp( layertypes{numlayers}, 'logistic' )
%     a = 5;
%     fprintf('%g', a);
%     ll = ll + double( sum(sum(y.*log(xi + (y==0)) + (1-y).*log(1-xi + (y==1)))) );
    % more stable:
     ll = ll + sum(sum(zi.*(y - (zi >= 0)) - log(1+exp(zi - 2*zi.*(zi>=0)))));                
     err = (y - zi);
elseif strcmp( layertypes{numlayers}, 'softmax' )
    ll = ll + double(sum(sum(y.*log(zi)))); %use Cross-entropy lossd
    %fprintf('%d\n',ll);
    err = y - zi;

end

loss = -ll/n;

if nargout <=2 
    % compute error
    if strcmp(model.type, 'mse')
        perr = sum(sum((y - zi).^2))/n;
    elseif strcmp(model.type, 'classification')
        [~,labels] = max(zi);
        [~,truelabels] = max(y);
        perr = mean(truelabels ~= labels);
    end
    loss = [loss; perr];
    if nargout == 1
        return;
    end
end

%% gradient
db{numlayers} = sum(err,2);     %sum of row
dW{numlayers} = err * z{numlayers}';
dx{numlayers} = err;
err = W{numlayers}' * err;
for k = numlayers-1:-1:1
    xi = z{k+1};
    if strcmp(layertypes{k}, 'logistic')
        err = err.* (1-xi).*xi;
    elseif strcmp(layertypes{k}, 'tanh')
        err = err .* (1 - xi) .* (1 + xi);
    elseif strcmp(layertypes{k}, 'linear')
        % err = err;
%    elseif strcmp(layertypes{k}, 'softmax')
%         xi = exp(xi);
  %      xi = bsxfun(@rdivide, xi, sum(xi,2));
    else
        error('Unknow layer type');
    end
%     z{i+1} = [];
    db{k} = sum(err,2);
    dW{k} = err * z{k}';
    dx{k} = err;
    err = W{k}'*err;
end
grad = pack(dW, db, psize, numlayers, layersizes);
grad = -grad/n;



if nargout < 3
    return
end
% compute error
if strcmp(model.type, 'mse')
    perr = sum(sum((y - zi).^2))/n;
elseif strcmp(model.type, 'classification')
    [~,labels] = max(zi);
    [~,truelabels] = max(y);
    perr = mean(truelabels ~= labels);
end
end


function M = pack(W,b, psize, numlayers, layersizes)
    
    M = zeros( psize, 1 );
    
    cur = 0;
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
    end
    
end

function [W,b] = unpack(M, numlayers, layersizes)

    W = cell( numlayers, 1 );
    b = cell( numlayers, 1 );
    
    cur = 0;
    for i = 1:numlayers
        W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );

        cur = cur + layersizes(i)*layersizes(i+1);
        
        b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );

        cur = cur + layersizes(i+1);
    end
    
end

function v = vec(A)

v = A(:);
end
