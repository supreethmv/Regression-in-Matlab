clear all
close all
clc
addpath('/home/danish/Desktop/Courses 2019 WS/Machine Learning/ml_ex04/DataEx4/cvx-a64/cvx/')


load('Part2-TrainingData.mat')
Xtrain2= zscore(Xtrain);
Xtrain2(Xtrain2>2.0) = 2.0;
Xtrain2(Xtrain2<-2.0) = -2.0;


Xtrain = [Xtrain, Xtrain2,  ones(size(Xtrain,1),1)];
LossLSR=[];
LossR=[];
LossLS=[];
for lambda = 1:5:16
    
    
    W = L1LossPlusL2Regularization(Xtrain,Ytrain,lambda); % Least Squares with L1-Regularization lambda = 10
    yLSR = Xtrain*W;
    LossLSR= [LossLSR mean((Ytrain - yLSR).^2)];  % Mean Squared Error on training data
    
    
    
    wR = RidgeRegression(Xtrain,Ytrain,lambda); % Least Squares with L2-Regularization lambda = 10
    yR = Xtrain*wR;
    LossR= [LossR mean((Ytrain - yR).^2)];   % Mean Squared Error on training data
    
    
    
    wLS = LeastSquares(Xtrain,Ytrain);
    yLS =Xtrain*wLS;
    LossLS= [LossLS mean((Ytrain - yLS).^2)];  % Mean Squared Error on training data
    
end

Losses(:,1) = LossLSR;
Losses(:,2) = LossR;
Losses(:,3) = LossLS;


figure1 = figure('WindowState','maximized');
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot1 = plot(Losses,'Marker','o','LineWidth',3,'Parent',axes1);
set(plot1(1),'DisplayName','L1Loss+L2Reg');
set(plot1(2),'DisplayName','Ridge Reg');
set(plot1(3),'DisplayName','LLS Reg');
ylabel({'Loss'});
xlabel({'Lambda'});
title({'Loss vs Lambda'});
box(axes1,'on');
grid(axes1,'on');
set(axes1,'FontSize',12,'FontWeight','bold');
legend(axes1,'show');

LossLeastSqReg = LossLS(1);

save LossSecondExtrainloss 'LossLeastSqReg'


% wL1_10 = L1LossPlusL2Regularization(Xtrain,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10

% p=polyfit(Xtrain,Ytrain,1) %fits data to a linear, 1st degree polynomial
% yFit = X*p(1)+p(2); %Calculate fitted regression line


figure,hold on, grid on
scatter(1:1794,Ytrain,'o');
plot(1:1794,yLS,'s');
axis on
grid on
legend('Actual Response','Predicted Response')


