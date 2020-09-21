clear all, 
close all, 
clc,
%%
P=load('Iden_IA.txt');

%Fe=P.Fe
%% Filtrage des données 
%   Design a 8th-order butterworth band-pass filter which passes frequencies between 0.15 and 0.3.

%On calcule et trace la representation fréquencielle
Fs=1/0.016; %fréquence d'échantillonage (hz)
nfft= length(P);% Nombre de point de la fft 
S=fft(P(:,2),nfft);
S=S(1:nfft/2);
ms=abs(S);
%Vecteur des frequence 
f=(0:(nfft/2)-1)*(Fs/nfft);
figure;
plot(f,20*log10(ms))
title('Tracé de la FFT du signal des positions bruité')
xlabel('Frequence en Hz');% Nombre de point de la fft 
%La frequence au dela de laquelle  l'energie du bruit  est superieure à celle du signal utile est 4Hz

fd = 10;
fa =17;
N = 10000;
Wp = 2*fd/Fs;
Ws = 2*fa/Fs;
Rp = 0.1;
Rs = 50;
[n,Wn] = buttord(Wp,Ws,3,Rs);   % 3dB attenuation à fd
[b,a] = butter(n,Wn);
figure;

freqz(b,a,N,Fs);
%%
%[b,a]=butter(2,[10/62.5,20/62.5]); 

 %Wn = 5/62.5;                   % Normalized cutoff frequency
%[a,b,k] = butter(8,Wn,'high');  % Butterworth filter

%Affichage des résultats
%2 et 3 ===> q1 et q2
% 4 et 5 ====> dq1 et dq2
% 6 et 7 ====> ddq1 et ddq2
% 8 tau
% 9 Vitesse
%%
%plot(P(:,1),filter(b,a,P(:,4)),'b-');
%hold on
%plot(P(:,1),P(:,9),'r-');
%%
for j=2:3
    figure
    cb1=plot(P(:,1),(P(:,j))*180/pi,'b-');
    set(cb1,'LineWidth',2);
    Xl=xlabel('t   (en secondes)');
    tl=title(['Tracé de la composante numéro q',num2str(j-1)]);
    set(tl,'FontSize',18);
    set(tl,'FontWeight','bold');
    set(Xl,'FontSize',12);
    set(gca,'FontSize',14);
end;
%%
for j=4:5
    figure
    cb1=plot(P(:,1),(P(:,j)),'b-');
    set(cb1,'LineWidth',2);
    Xl=xlabel('t   (en secondes)');
    tl=title(['Tracé de la composante numéro dq',num2str(j-1)]);
    set(tl,'FontSize',18);
    set(tl,'FontWeight','bold');
    set(Xl,'FontSize',12);
    set(gca,'FontSize',14);
end;
%%
for j=6:7
    figure
    cb1=plot(P(:,1),(P(:,j)),'b-');
    set(cb1,'LineWidth',2);
    Xl=xlabel('t   (en secondes)');
    tl=title(['Tracé de la composante numéro ddq',num2str(j-1)]);
    set(tl,'FontSize',18);
    set(tl,'FontWeight','bold');
    set(Xl,'FontSize',12);
    set(gca,'FontSize',14);
end;
%%
w0=2250;
for i=1:length(P)
    tauu(i)=((w0-(P(i,4))*60*5/2*pi))/34;
end

figure
    cb1=plot(P(:,1),(tauu),'b-');
    set(cb1,'LineWidth',2);
    Xl=xlabel('t   (en secondes)');
    tl=title(['couple moteur']);
    set(tl,'FontSize',18);
    set(tl,'FontWeight','bold');
    set(Xl,'FontSize',12);
    set(gca,'FontSize',14);
    
%%
figure
    cb1=plot(P(:,1),P(:,8),'b-');
    set(cb1,'LineWidth',2);
    Xl=xlabel('t   (en secondes)');
    tl=title(['couple moteur']);
    set(tl,'FontSize',18);
    set(tl,'FontWeight','bold');
    set(Xl,'FontSize',12);
    set(gca,'FontSize',14);
%% Identification

W=zeros(2,9);
Y=zeros(2,1);

for i=1: length(P)
    %q1=filter(b,a,P(i,2));
    %q2=filter(b,a,P(i,3))+pi;
    %dq1=filter(b,a,P(i,4));
    %dq2=filter(b,a,P(i,5));
    %ddq1=filter(b,a,P(i,6));
    %ddq2=filter(b,a,P(i,7));
    %tau=filter(b,a,P(i,8));
    
    q1=P(i,2);
    q2=P(i,3)+pi;
    dq1=P(i,4);
    dq2=P(i,5);
    ddq1=P(i,6);
    ddq2=P(i,7);
    tau=P(i,8);
    D=[ddq1*(sin(q2)^2)+2*dq1*dq2*cos(q2)*sin(q2)  ddq1   ddq2*cos(q2)-(dq2^2)*sin(q2) 0        0           dq1 sign(q1) 0      0;
                 -cos(q2)*sin(q2)*(dq1^2)             0   ddq1*cos(q2)               ddq2     sin(q2)      0    0     dq2 sign(q2)];
    y=[tau;0];
    W=[W;D];
    Y=[Y;y];
end
W=W(3:length(W),:);
Y=Y(3:length(Y),:);

%%
p=pinv(W)*Y

sprintf('%.15f',p')
    
%% Validation 
Tau=W*p;
T1= zeros(length(P),1);
k=1;
for j=1:length(Tau)
   if mod(j,2)== 1
       T1(k,1)=Tau(j,1);
       k=k+1;
   end
end;

%%
figure
    cb1=plot(P(:,1),P(:,8),'b-');
    hold on
    cb2=plot(P(:,1),T1(:,1),'r-');
    set(cb1,'LineWidth',2);
     set(cb2,'LineWidth',2);
    Xl=xlabel('t   (en secondes)');
    tl=title(['couple moteur']);
    set(tl,'FontSize',18);
    set(tl,'FontWeight','bold');
    set(Xl,'FontSize',12);
    set(gca,'FontSize',14);