% load ./MATLAB_data/ofdata.mat

% Split data into training and test set
rng(5000);
holdoutCVP = cvpartition(labelVector, 'holdout', 0.3);
train_label = labelVector(holdoutCVP.training,:);
train_label2 = strcmp(train_label,'deviate');
train_feature = featureVector(holdoutCVP.training,:);
test_label = labelVector(holdoutCVP.test,:);
test_label2 = strcmp(test_label,'deviate');
test_feature = featureVector(holdoutCVP.test,:);

label_vector2 = strcmp(labelVector,'deviate');

c = cvpartition(label_vector2,'k',10);
opts = statset('display','final');
fun = @(XT,yT,Xt,yt)...
      (sum(~strcmp(yt, predict(fitcsvm(XT, yT, 'Standardize', true, 'KernelFunction', 'rbf', ...
      'KernelScale', 'auto'), Xt))));

[fs,history] = sequentialfs(fun, featureVector, label_vector2, 'cv',c,'options',opts)

% %Filtering feature selection: criteria --> Chernoff Bound
% I = rankfeatures(featureVector', label_vector2,'Criterion', 'bhattacharyya','NumberOfIndices', 4000);

%SVM on the selected train-test data%
% SVMModel = fitcsvm(train_feature(:,I), train_label2, 'Standardize', true, 'KernelFunction', 'rbf', ...
%     'KernelScale', 'auto');
% [predictedLabels, scores] = predict(SVMModel, test_feature(:,I));
% 
% %k-fold CV%
% classifier = fitcsvm(featureVector(:,I), label_vector2, 'Standardize', true, 'KernelFunction', 'rbf', ...
%      'KernelScale', 'auto');
% CVSVMModel = crossval(classifier);
% classLoss = kfoldLoss(CVSVMModel);
% % yfit = kfoldPredict(CVSVMModel);
% 
% %[avgConf, ord] = confusionmat(label_vector2, yfit)
% 
% [confMat,order] = confusionmat(test_label2, predictedLabels)
% accuracy = (confMat(1,1) + confMat(2,2)) / (sum(sum(confMat))) %percentage of true positive and true neg%
% precision = confMat(2,2)/(confMat(1,2)+confMat(2,2))
% recall = confMat(2,2)/(confMat(2,1)+confMat(2,2))
% fscore = 2 * (precision * recall) / (precision + recall)
% error = 0;
% for j = 1:size(test_label2, 1)
%  if (test_label2(j) ~= predictedLabels(j))
%    error = error + 1;
%  end
% end
% test_error = error / size(test_label2, 1)
%sensitivity = confMat(1,1)/(confMat(1,1)+confMat(2,1))
%specificity = confMat(2,2)/(confMat(2,2)+confMat(1,2))


% hold on;
% plot(classLoss, 'r');
% plot(test_error, 'r--');
% plot(accuracy, 'b'); 
% plot(precision, 'g'); 
% plot(recall, 'c'); 
% plot(fscore, 'b--');

% legend('general_err','train_err','accuracy','precision', 'recall', 'fscore', 'Location', 'northwest');