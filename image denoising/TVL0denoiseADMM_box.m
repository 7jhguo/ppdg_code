function newim = TVL0denoiseADMM_box(ori, im, lambda, niter)
%
% TVL0denoiseADMM_box is the code of ADMM for image denoising using the l0 gradient minimization.
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

  alpha=0.02; % step size
  a    =-1;
  b    = 1;
  rho  = 1;


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
  z(:, :, 1)=p(:, :, 1);
  z(:, :, 2)=p(:, :, 2);
  
  
  time_temp = toc;

%% main loop
for k=1:niter
  
     ux=u(:, [2:width, width]) - u; 
     uy=u([2:height, height], :) - u;
     p(:, :, 1) = ux+z(:, :, 1)/rho; 
     p(:, :, 2) = uy+z(:, :, 2)/rho;
     
    % compute project onto [a_i,b_i]
    for j=1:width
        for i=1:height
            for s=1:2
            if p(i, j, s)>b
                pi(i, j, s) =b;
            elseif p(i, j, s)<=a
                pi(i, j, s) =a;
            else 
                pi(i, j, s) = p(i, j, s);
            end
            end
         end
    end 

    for j=1:width
        for i=1:height
            for s=1:2
            if 2*p(i, j, s)*pi(i, j, s)-pi(i, j, s)^2>=lambda/rho
                p(i, j, s) =   pi(i, j, s);
            else 
                p(i, j, s) = 0;
            end
            end
         end
    end 
      z(:, :, 1) =z(:, :, 1) + rho*(ux- p(:, :, 1));  
      z(:, :, 2) =z(:, :, 2) + rho*(uy- p(:, :, 2));  
      
    div=[z([1:height-1], :, 2); zeros(1, width)] - [zeros(1, width); z([1:height-1], :, 2)];
    div=[z(:, [1:width-1], 1)  zeros(height, 1)] - [zeros(height, 1)  z(:, [1:width-1], 1)] + div;
    
    wx= ux- p(:, :, 1);
    wy= uy- p(:, :, 2);
    div2 = [wy([1:height-1], :); zeros(1, width)] - [zeros(1, width); wy([1:height-1], :)];
    div2 = [wx(:, [1:width-1])  zeros(height, 1)] - [zeros(height, 1)  wx(:, [1:width-1])] + div2;
    
     % gradient step
      u  = u-alpha*(u-nim-div-rho*div2);

     % objective function
      ux=u(:, [2:width, width]) - u;
      uy=u([2:height, height], :) - u;
      uxy=ux.^2+uy.^2;
      g =  uxy~=0;
      v=(u-nim).^2;
      obj_full=lambda* sum(g(:))+ 0.5*sum( v( : ) );
      fprintf('Iteration %d: energy %g \n', k, obj_full);
    
end
  out_time = toc-time_temp; 
  disp( out_time)
  tempu=(u-u_ori).^2;
  psnr = 10*log10(height*width*(max(u(:)))^2/sum(tempu(:)));
  disp(psnr)

  newim  =u;
  
