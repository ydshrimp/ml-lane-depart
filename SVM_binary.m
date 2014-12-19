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
numFeatures = 1100;

%Filtering feature selection: criteria --> Chernoff Bound
I = rankfeatures(featureVector', label_vector2,'Criterion', 'bhattacharyya','NumberOfIndices', numFeatures);

%k-fold CV: finding the lowest generalization error possible%
classifier = fitcsvm(featureVector(:,I), label_vector2, 'Standardize', true, 'KernelFunction', 'rbf', ...
'KernelScale', 'auto');
CVSVMModel = crossval(classifier);
classLoss = kfoldLoss(CVSVMModel)

%test error on given training set%
SVMModel = fitcsvm(train_feature(:,I), train_label2, 'Standardize', true, 'KernelFunction', 'rbf', ...
     'KernelScale', 'auto');
[predictedLabels, scores] = predict(SVMModel, train_feature(:,I));

error = 0;
for j = 1:size(train_label2, 1)
 if (train_label2(j) ~= predictedLabels(j))
   error = error + 1;
 end
end
train_error = error / size(train_label2, 1)

[confMat,order] = confusionmat(test_label2, predictedLabels)
accuracy = (confMat(1,1) + confMat(2,2)) / (sum(sum(confMat))) %percentage of true positive and true neg%
precision = confMat(2,2)/(confMat(1,2)+confMat(2,2))
recall = confMat(2,2)/(confMat(2,1)+confMat(2,2))
fscore = 2 * (precision * recall) / (precision + recall)

% hold on;
% plot(classLoss, 'r-x');
% plot(test_error, 'b-o');
% legend('test err','train err','Location', 'northwest');

% hold on;
% plot(test_error, 'r--');
% plot(accuracy, 'b'); 
% plot(precision, 'g'); 
% plot(recall, 'c'); 
% plot(fscore, 'b--');
% 
% legend('general_err','train_err','accuracy','precision', 'recall', 'fscore');
