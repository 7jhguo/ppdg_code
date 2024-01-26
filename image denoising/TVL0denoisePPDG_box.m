function newim = TVL0denoisePPDG_box(ori, im, lambda, niter)
%
% TVL0denoisePPDG_box is the code of PPDG, proposed in the paper "Preconditioned Primal-Dual Gradient Methods for Nonconvex
% Composite and Finite-Sum Optimization", for image denoising using the l0 gradient minimization.
%
% Arguments:
%    ori         -  original image
%    im          -  noisey image
%    lambda      -  Regularization parameter
%    niter       -  Number of iterations
%
% Returns:
%    newim       -  Denoised image with pixel values in [0 1].
%
%
%  2024/1/26

tic;
  if(nargin<3) || isempty(niter)
    niter=100;
  end

  L2=8.0; % norm 
  alpha=0.2; % step size
  sigma=L2*alpha; % 1/beta
  a    =-1;
  b    = 1;

  [height, width]=size(im);   
  p=zeros(height, width, 2);
  d=zeros(height, width);
  ux=zeros(height, width);
  uy=zeros(height, width);

  mx=max(im(:));
  mx_ori = max(ori(:));
  if(mx>1.0)
    nim=double(im)/double(mx); % normalize
    ori = double(ori)/double(mx_ori);
  else
    nim=double(im); % leave intact
    ori=double(ori);
  end
   u=nim;
   u_ori = ori;

  p(:, :, 1)=u(:, [2:width, width]) - u;
  p(:, :, 2)=u([2:height, height], :) - u; 

  time_temp = toc;

  %% main loop
  for k=1:niter
   
    ux=u(:, [2:width, width]) - u; 
    uy=u([2:height, height], :) - u;
    % proximal step
    p =p + cat(3, ux, uy)/sigma;    
    for j=1:width
        for i=1:height
            for s=1:2
            if p(i, j, s)>lambda/b+b/sigma
                p(i, j, s) =   p(i, j, s)-b/sigma;
            end
            if  (p(i, j, s)<=lambda/b+b/sigma) && (p(i, j, s)>lambda/b)
                    p(i, j, s) = lambda/b;
            end
            if  (p(i, j, s)<=lambda/b) && (p(i, j, s)>lambda/a)
                p(i, j, s) = p(i, j, s);
            end
            if  (p(i, j, s)<=lambda/a) && (p(i, j, s)>lambda/a+a/sigma)
                p(i, j, s)=lambda/a;
            end 
            if p(i, j, s)<=lambda/a+a/sigma          
                    p(i, j, s) = p(i, j, s)-a/sigma;
            end
            end
         end
    end 
   
    
    % compute divergence in div  
    div=[p([1:height-1], :, 2); zeros(1, width)] - [zeros(1, width); p([1:height-1], :, 2)];
    div=[p(:, [1:width-1], 1)  zeros(height, 1)] - [zeros(height, 1)  p(:, [1:width-1], 1)] + div;% +p(:, :, 3);
    
    %computing gradient step   
    unew = u-alpha*(u-nim-div); 
    u=2*unew -u;  %2x-x
    
    %objective function
      ux=u(:, [2:width, width]) - u;
      uy=u([2:height, height], :) - u;
      uxy=ux.^2+uy.^2;
      g =  uxy~=0;
     
     v=(u-nim).^2;
     obj_full= lambda*sum(g(:))+ 0.5*sum( v( : ) );

     fprintf('Iteration %d: energy %g \n', k, obj_full);

  end
  out_time = toc-time_temp; 
  disp( out_time)
  tempu=(u-u_ori).^2;
  psnr = 10*log10(height*width*(max(u(:)))^2/sum(tempu(:)));
  disp(psnr)
  newim=u;
