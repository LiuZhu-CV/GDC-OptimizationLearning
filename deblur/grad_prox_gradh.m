function Yk1 = grad_prox_gradh(Yk,Bx,Sbig,W,WT,L,lambda)


D=Sbig.*fft2(Yk)-fft2(Bx);
Y=Yk-2/L*(real(ifft2(conj(Sbig).*D )));

WY=W(Y);
Wtilde = WY;
t = Wtilde.^2  < lambda/(L);
Wtilde(t) = 0;
WY = Wtilde;
clear t;
Yk1 = WT(WY);