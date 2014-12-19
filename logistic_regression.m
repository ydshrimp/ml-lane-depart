% fitting logistic regression
%X: num_train_examples x num_features matrix
%y: num_train_examples label vector (0: go / 1: deviate)
rng(5000);
holdoutCVP = cvpartition(labelVector, 'holdout', 0.3);
train_label = labelVector(holdoutCVP.training,:);
train_label2 = strcmp(train_label,'deviate');
train_feature = featureVector(holdoutCVP.training,:);
test_label = labelVector(holdoutCVP.test,:);
test_label2 = strcmp(test_label,'deviate');
test_feature = featureVector(holdoutCVP.test,:);

label_vector2 = strcmp(labelVector,'deviate');
numFeatures = size(featureVector, 2);

%Filtering feature selection: criteria --> Chernoff Bound
I = rankfeatures(featureVector', label_vector2,'Criterion', 'bhattacharyya','NumberOfIndices', numFeatures);


%Using glmfit
coeffs = glmfit(train_feature(:, I), train_label2, 'binomial', 'link', 'logit');
ypredict = glmval(coeffs, train_feature(:, I), 'logit') > 0.5;

%Compute cross validation error (10-fold CV)
classf = @(XTRAIN, ytrain,XTEST)(+(glmval(glmfit(XTRAIN, ytrain, 'binomial', 'link', 'logit'), ...
    XTEST, 'logit') > 0.5));
mse = crossval('mse', featureVector(:,I), +label_vector2,'Predfun', classf, 'kfold', 10)

% Generate confusion matrix 
[confMat,order] = confusionmat(double(train_label2), double(ypredict))
accuracy = (confMat(1,1) + confMat(2,2)) / (sum(sum(confMat))) %percentage of true positive and true neg%
precision = confMat(2,2)/(confMat(1,2)+confMat(2,2))
recall = confMat(2,2)/(confMat(2,1)+confMat(2,2))
fscore = 2 * (precision * recall) / (precision + recall)
error = 0;
for j = 1:size(train_label2, 1)
 if (train_label2(j) ~= ypredict(j))
   error = error + 1;
 end
end
train_error = error / size(train_label2, 1)

% hold on;
% plot(mse, 'r');
% plot(test_error, 'r--');
% plot(accuracy, 'b'); 
% plot(precision, 'g'); 
% plot(recall, 'c'); 
% plot(fscore, 'b--');
% 
% legend('general_err','train_err','accuracy','precision', 'recall', 'fscore', 'Location', 'northwest');