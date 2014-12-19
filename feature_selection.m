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

i = 1;
for numFeatures = 100:200:size(featureVector, 2)
    numFeaturesArray(i) = numFeatures;
    %Filtering feature selection: criteria --> Chernoff Bound
    I = rankfeatures(featureVector', label_vector2,'Criterion','bhattacharyya','NumberOfIndices', numFeatures);

%     %Using glmfit
%     coeffs = glmfit(train_feature(:, I), train_label2, 'binomial', 'link', 'logit');
%     predictedLabels = glmval(coeffs, train_feature(:, I), 'logit') > 0.5;

%     %Compute cross validation error (10-fold CV)
%     classf = @(XTRAIN, ytrain,XTEST)(+(glmval(glmfit(XTRAIN, ytrain, 'binomial', 'link', 'logit'), ...
%         XTEST, 'logit') > 0.5));
%     classLoss(i) = crossval('mse', featureVector(:,I), +label_vector2,'Predfun', classf, 'kfold', 10)

    
    %SVM on the selected train-test data%
    SVMModel = fitcsvm(train_feature(:,I), train_label2, 'Standardize', true, 'KernelFunction', 'rbf', ...
        'KernelScale', 'auto');
    [predictedLabels, scores] = predict(SVMModel, train_feature(:,I));

    %k-fold CV%
    classifier = fitcsvm(featureVector(:,I), label_vector2, 'Standardize', true, 'KernelFunction', 'rbf', ...
         'KernelScale', 'auto');
    CVSVMModel = crossval(classifier);
    classLoss(i) = kfoldLoss(CVSVMModel)
    % yfit = kfoldPredict(CVSVMModel);

    %[avgConf, ord] = confusionmat(label_vector2, yfit)

    %predictedTestLabs = glmval(coeffs, test_feature(:, I), 'logit') > 0.5;
    predictedTestLabs = predict(SVMModel, test_feature(:,I));
    [confMat,order] = confusionmat(test_label2, predictedTestLabs)
    accuracy(i) = (confMat(1,1) + confMat(2,2)) / (sum(sum(confMat))) %percentage of true positive and true neg%
    precision(i) = confMat(2,2)/(confMat(1,2)+confMat(2,2))
    recall(i) = confMat(2,2)/(confMat(2,1)+confMat(2,2))
    fscore(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i))
    
    error = 0;
    for j = 1:size(train_label2, 1)
     if (train_label2(j) ~= predictedLabels(j))
       error = error + 1;
     end
    end
    train_error(i) = error / size(train_label2, 1)
    i = i + 1;
    
   % predictedLabels = glmval(coeffs, train_feature(:, I), 'logit') > 0.5;
    %train_error(i) = sum(train_label2~=predictedLabels)/size(train_label2,1)
    
    %sensitivity = confMat(1,1)/(confMat(1,1)+confMat(2,1))
    %specificity = confMat(2,2)/(confMat(2,2)+confMat(1,2))
end

hold on;
plot(numFeaturesArray, classLoss, 'r');
plot(numFeaturesArray, train_error, 'b');
plot(numFeaturesArray, accuracy, 'b'); 
plot(numFeaturesArray, precision, 'g-'); 
plot(numFeaturesArray, recall, 'c'); 
plot(numFeaturesArray, fscore, 'b--');

xlabel('Number of Features', 'FontSize', 14);
title('Filter Feature Selection (SVM)', 'FontSize', 18)
legend('Cross validation Error','Train Error','Accuracy','Precision', 'Recall', 'F1-score');
