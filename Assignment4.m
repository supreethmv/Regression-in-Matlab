clear all
close all
clc
lambda0=0;
lambda10=10;

addpath('/home/danish/Desktop/Courses 2019 WS/Machine Learning/ml_ex04/DataEx4/cvx-a64/cvx/')

load('Part1-TrainingData.mat') % load data



%% Question 1: (1-a) Least Square
w = LeastSquares(Xtrain,Ytrain);

%% Question 1: (1-b) Ridge Regression

%% Question 1: (2) Basis Function

DesignMatrix1 = Basis(Xtrain,1);
DesignMatrix2 = Basis(Xtrain,2);
DesignMatrix3 = Basis(Xtrain,3);
DesignMatrix5 = Basis(Xtrain,5);
DesignMatrix10 = Basis(Xtrain,10);
DesignMatrix15 = Basis(Xtrain,15);
DesignMatrix20 = Basis(Xtrain,20);

%% Question 1(3) Part 1: L1 vs L2 regularization

% In the first example we have only one feature, thus we want to learn a function f : R → R. First
% plot the training data (plot(Xtrain, Ytrain,);


figure1 = figure('WindowState','maximized');
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(Xtrain,Ytrain,'MarkerSize',8,'Marker','.','LineStyle','none');
ylabel('Y','FontWeight','bold');
xlabel('X','FontWeight','bold');
title({'Data Plotting'});
box(axes1,'on');
grid(axes1,'on');
set(axes1,'FontSize',12,'FontWeight','bold');


% Which loss function (L1 or L2) is more appropriate for this kind of data ?
% Justify this by checking the data plot. Use in the next part only the regression method with
% your chosen loss (that is either ridge regression or L1-loss with L2-regularizer).

wRidge0 = RidgeRegression(DesignMatrix10,Ytrain,lambda0); % Least Squares with L2-Regularization lambda = 0
wRidge10 = RidgeRegression(DesignMatrix10,Ytrain,lambda10); % Least Squares with L2-Regularization lambda = 10

yCalcRidge0 = DesignMatrix10*wRidge0;
yCalcRidge10 = DesignMatrix10*wRidge10;

plot(Xtrain,yCalcRidge0,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcRidge10,'MarkerSize',8,'Marker','.','LineStyle','none');


wL1_0 = L1LossPlusL2Regularization(DesignMatrix10,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_10_10 = L1LossPlusL2Regularization(DesignMatrix10,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10

yCalcL1_0 = DesignMatrix10*wL1_0;
yCalcL1_10 = DesignMatrix10*wL1_10_10;


plot(Xtrain,yCalcL1_0,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL1_10,'MarkerSize',8,'Marker','.','LineStyle','none');


legend('Data','RidgeLambda0','RidgeLambda10','LSL1-Lambda0','LSL1-Lambda10')


%% Question 1(3) Part 2: k = 1, 2, 3, 5, 10, 15, 20 with L1

% Use the basis functions with k = 1, 2, 3, 5, 10, 15, 20 from part b) to fit the
% regularized version of the loss chosen in the previous part. Use regularization parameter
% λ = 10. Plot the resulting functions (use x = 0 : 0.01 : 1) for all values of k together with
% the training data,

figure1 = figure('WindowState','maximized');
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot(Xtrain,Ytrain,'MarkerSize',8,'Marker','.','LineStyle','none');
ylabel('Y','FontWeight','bold');
xlabel('X','FontWeight','bold');
title({'Data Plotting -L1 Loss (median of p(y|x))'});
box(axes1,'on');
grid(axes1,'on');
set(axes1,'FontSize',12,'FontWeight','bold');

wL1_10_1 = L1LossPlusL2Regularization(DesignMatrix1,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10
wL1_10_2 = L1LossPlusL2Regularization(DesignMatrix2,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10
wL1_10_3 = L1LossPlusL2Regularization(DesignMatrix3,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10
wL1_10_5 = L1LossPlusL2Regularization(DesignMatrix5,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10
wL1_10_10 = L1LossPlusL2Regularization(DesignMatrix10,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10
wL1_10_15 = L1LossPlusL2Regularization(DesignMatrix15,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10
wL1_10_20 = L1LossPlusL2Regularization(DesignMatrix20,Ytrain,lambda10); % Least Squares with L1-Regularization lambda = 10

wL1_0_1 = L1LossPlusL2Regularization(DesignMatrix1,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_0_2 = L1LossPlusL2Regularization(DesignMatrix2,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_0_3 = L1LossPlusL2Regularization(DesignMatrix3,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_0_5 = L1LossPlusL2Regularization(DesignMatrix5,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_0_10 = L1LossPlusL2Regularization(DesignMatrix10,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_0_15 = L1LossPlusL2Regularization(DesignMatrix15,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0
wL1_0_20 = L1LossPlusL2Regularization(DesignMatrix20,Ytrain,lambda0); % Least Squares with L1-Regularization lambda = 0

yCalcL10_1 = DesignMatrix1*wL1_10_1;
yCalcL10_2 = DesignMatrix2*wL1_10_2;
yCalcL10_3 = DesignMatrix3*wL1_10_3;
yCalcL10_5 = DesignMatrix5*wL1_10_5;
yCalcL10_10 = DesignMatrix10*wL1_10_10;
yCalcL10_15 = DesignMatrix15*wL1_10_15;
yCalcL10_20 = DesignMatrix20*wL1_10_20;

yCalcL0_1 = DesignMatrix1*wL1_0_1;
yCalcL0_2 = DesignMatrix2*wL1_0_2;
yCalcL0_3 = DesignMatrix3*wL1_0_3;
yCalcL0_5 = DesignMatrix5*wL1_0_5;
yCalcL0_10 = DesignMatrix10*wL1_0_10;
yCalcL0_15 = DesignMatrix15*wL1_0_15;
yCalcL0_20 = DesignMatrix20*wL1_0_20;

plot(Xtrain,yCalcL10_1,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL10_2,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL10_3,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL10_5,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL10_10,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL10_15,'MarkerSize',8,'Marker','.','LineStyle','none');
plot(Xtrain,yCalcL10_20,'MarkerSize',8,'Marker','.','LineStyle','none');

legend('Data','k1','k2','k3','k5','k10','k15','k20')


%% Compute loss for training and test data for Lambda = 10
load('Part1-TestData.mat')
yCalcTrain = [yCalcL10_1 yCalcL10_2 yCalcL10_3 yCalcL10_5 yCalcL10_10 yCalcL10_15 yCalcL10_20];

DesignMatrixTest1 = Basis(Xtest,1);
DesignMatrixTest2 = Basis(Xtest,2);
DesignMatrixTest3 = Basis(Xtest,3);
DesignMatrixTest5 = Basis(Xtest,5);
DesignMatrixTest10 = Basis(Xtest,10);
DesignMatrixTest15 = Basis(Xtest,15);
DesignMatrixTest20 = Basis(Xtest,20);


yCalcL1 = DesignMatrixTest1*wL1_10_1;
yCalcL2 = DesignMatrixTest2*wL1_10_2;
yCalcL3 = DesignMatrixTest3*wL1_10_3;
yCalcL5 = DesignMatrixTest5*wL1_10_5;
yCalcL10 = DesignMatrixTest10*wL1_10_10;
yCalcL15 = DesignMatrixTest15*wL1_10_15;
yCalcL20 = DesignMatrixTest20*wL1_10_20;

yCalcTest = [yCalcL1 yCalcL2 yCalcL3 yCalcL5 yCalcL10 yCalcL15 yCalcL20];

LossTrainTest = zeros(7,1);
for kk = 1:7
    LossTrainTest(kk,1) = mean((Ytrain - yCalcTrain(:,kk)).^2);   % Mean Squared Error on training data
    LossTrainTest(kk,2) = mean((Ytest - yCalcTest(:,kk)).^2);   % Mean Squared Error on test data
    
end

trainloss = LossTrainTest(:,1);
testloss = LossTrainTest(:,2);
k_array = [1 2 3 5 10 15 20]; 



figure1 = figure('WindowState','maximized');
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot1=plot(k_array, LossTrainTest,'Marker','o','LineWidth',3);
set(plot1(1),'DisplayName','LossTrain');
set(plot1(2),'DisplayName','LossTest');
ylabel({'Loss'});
xlabel({'K (Basis)'});
title({'Loss w.r.t # of Basis with lambda = 10'});
box(axes1,'on');
grid(axes1,'on');
set(axes1,'FontSize',12,'FontWeight','bold');
legend1 = legend(axes1,'show');
set(legend1,'FontSize',12);



save LossFirstEx trainloss testloss

%% Compute loss for training and test data for Lambda = 0

yCalcTrain = [yCalcL0_1 yCalcL0_2 yCalcL0_3 yCalcL0_5 yCalcL0_10 yCalcL0_15 yCalcL0_20];



yCalcL1 = DesignMatrixTest1*wL1_0_1;
yCalcL2 = DesignMatrixTest2*wL1_0_2;
yCalcL3 = DesignMatrixTest3*wL1_0_3;
yCalcL5 = DesignMatrixTest5*wL1_0_5;
yCalcL10 = DesignMatrixTest10*wL1_0_10;
yCalcL15 = DesignMatrixTest15*wL1_0_15;
yCalcL20 = DesignMatrixTest20*wL1_0_20;

yCalcTest = [yCalcL1 yCalcL2 yCalcL3 yCalcL5 yCalcL10 yCalcL15 yCalcL20];

LossTrainTest = zeros(7,1);
for kk = 1:7
    LossTrainTest(kk,1) = mean((Ytrain - yCalcTrain(:,kk)).^2);   % Mean Squared Error on training data
    LossTrainTest(kk,2) = mean((Ytest - yCalcTest(:,kk)).^2);   % Mean Squared Error on test data
    
end

trainloss0 = LossTrainTest(:,1);
testloss0 = LossTrainTest(:,2);



figure1 = figure('WindowState','maximized');
axes1 = axes('Parent',figure1);
hold(axes1,'on');
plot1=plot(k_array, LossTrainTest,'Marker','o','LineWidth',3);
set(plot1(1),'DisplayName','LossTrain');
set(plot1(2),'DisplayName','LossTest');
ylabel({'Loss'});
xlabel({'K (Basis)'});
title({'Loss w.r.t # of Basis with lambda = 0'});
box(axes1,'on');
grid(axes1,'on');
set(axes1,'FontSize',12,'FontWeight','bold');
legend1 = legend(axes1,'show');
set(legend1,'FontSize',12);



save LossFirstEx0 trainloss0 testloss0