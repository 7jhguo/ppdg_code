function newim = TVL0denoisePDHG_box(ori, im, lambda, niter)
% TVL0denoisePDHG_box is the code of PDHG for image denoising using the l0 gradient minimization.
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

  alpha = 20; % step size 
  a     =-1;
  b     = 1;

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
 
  w(:, :, 1) = p(:, :, 1);
  w(:, :, 2) = p(:, :, 2);
  z   = u;
  
  time_temp = toc;


  %% main loop
  for k=1:niter

    zx=z(:, [2:width, width]) - z; 
    zy=z([2:height, height], :) - z;

     % proximal step
      p_temp(:, :, 1)  = alpha*w(:, :, 1) +zx;
      p_temp(:, :, 2)  =  alpha*w(:, :, 2) + zy;
  
      % compute project onto [a_i,b_i]
     for j=1:width
        for i=1:height
            for s=1:2
            if p_temp(i, j, s)>b
                pi(i, j, s) =b;
            elseif p_temp(i, j, s)<=a
                pi(i, j, s) =a;
            else 
                pi(i, j, s) = p_temp(i, j, s);
            end
            end
         end
     end 

   for j=1:width
        for i=1:height
            for s=1:2
            if 2*p_temp(i, j, s)*pi(i, j, s)-pi(i, j, s)^2>=lambda*alpha
                p_temp(i, j, s) =   pi(i, j, s);
            else 
                p_temp(i, j, s) = 0;
            end
            end
         end
    end 
   w(:, :, 1) = w(:, :, 1)+1/alpha*(zx-p(i, j, 1));
   w(:, :, 2) = w(:, :, 2)+1/alpha*(zy-p(i, j, 2)); 
   
    % compute divergence in div  % -A'w
    div=[w([1:height-1], :, 2); zeros(1, width)] - [zeros(1, width); w([1:height-1], :, 2)];
    div=[w(:, [1:width-1], 1)  zeros(height, 1)] - [zeros(height, 1)  w(:, [1:width-1], 1)] + div;% +p(:, :, 3);
    
    
    %computing gradient step   
    unew = 1/(alpha+1)*u+(nim+div)/(1+1/alpha); 
    
    z = 2*unew - u;  %2x-x
    u = unew;
    
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
