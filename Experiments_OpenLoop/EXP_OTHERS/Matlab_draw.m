clear all, 
close all, 
clc,
%%
X=load('Coordonees1_100.txt');


plot(X(:,1)-X(1,1),X(:,2),'r-');
hold on;
plot(X(:,1)-X(1,1),X(:,3),'b-');