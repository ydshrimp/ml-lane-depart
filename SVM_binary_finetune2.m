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

%k-fold CV%
classifier = fitcsvm(featureVector(:,I), label_vector2, 'Standardize', true, 'KernelFunction', 'rbf', ...
     'KernelScale', 'auto');

init_bc = 1;
init_ks = classifier.KernelParameters.Scale;

for i = 1:11
    for k = 1:11
        bc = init_bc * (1e-5)*10^i;
        ks = init_ks * (1e-5)*10^k;
        %SVM on the selected train-test data%
        svm_classifier = fitcsvm(featureVector(:,I), label_vector2, 'Standardize', true, 'KernelFunction', 'rbf', ...
     'KernelScale', ks, 'BoxConstraint', bc);
        CVSVMModel = crossval(svm_classifier);
        classLoss(i, k) = kfoldLoss(CVSVMModel)

        SVMModel = fitcsvm(train_feature(:,I), train_label2, 'Standardize', true, 'KernelFunction', 'rbf', ...
             'KernelScale', ks, 'BoxConstraint', bc);
        [predictedLabels, scores] = predict(SVMModel, train_feature(:,I));
           
        error = 0;
        for j = 1:size(train_label2, 1)
         if (train_label2(j) ~= predictedLabels(j))
           error = error + 1;
         end
        end
        train_error(i, k) = error / size(train_label2, 1)
    end
end

%sensitivity = confMat(1,1)/(confMat(1,1)+confMat(2,1))
%specificity = confMat(2,2)/(confMat(2,2)+confMat(1,2))

% hold on;
% plot(classLoss, 'r');
% plot(test_error, 'r--');
% plot(accuracy, 'b'); 
% plot(precision, 'g'); 
% plot(recall, 'c'); 
% plot(fscore, 'b--');
% 
% legend('general_err','train_err','accuracy','precision', 'recall', 'fscore', 'Location', 'northwest');