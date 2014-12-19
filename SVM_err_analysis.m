% load ./MATLAB_data/ofdata.mat

% Split data into training and test set
for i = 1:9
    p = 0.1 * (10 - i); %To simulate test-train error, we treat p*m = m'%
    
    rng(5000);
    holdoutCVP = cvpartition(labelVector, 'holdout', p);
    subset_label = labelVector(holdoutCVP.training,:);
    subset_label2 = strcmp(subset_label,'deviate');
    subset_feature = featureVector(holdoutCVP.training,:);  
    test_label = labelVector(holdoutCVP.test,:);
    test_label2 = strcmp(test_label,'deviate');
    test_feature = featureVector(holdoutCVP.test,:);
    m(i) = size(subset_feature, 1);
    
    label_vector2 = strcmp(labelVector,'deviate');
    numFeatures = 1100;
     %Filtering feature selection: criteria --> Chernoff Bound
    I = rankfeatures(featureVector', label_vector2,'Criterion', 'bhattacharyya','NumberOfIndices', numFeatures);

    %k-fold CV: finding the lowest generalization error possible%
    classifier = fitcsvm(subset_feature(:,I), subset_label2, 'Standardize', true, 'KernelFunction', 'rbf', ...
    'KernelScale', 'auto');
    CVSVMModel = crossval(classifier);
    classLoss(i) = kfoldLoss(CVSVMModel)
    [predictedLabels, scores] = predict(classifier, subset_feature(:,I));

    error = 0;
    for j = 1:size(subset_label2, 1)
     if (subset_label2(j) ~= predictedLabels(j))
       error = error + 1;
     end
    end
    test_error(i) = error / size(subset_label2, 1)
end

hold on;
plot(m, classLoss, 'r-x');
plot(m, test_error, 'b-o');
legend('Cross Validation Error','Testing Error');
xlabel('Number of Training Examples', 'FontSize', 14); ylabel('Error', 'FontSize', 14);
title('Learning Curve', 'FontSize', 18);

% plot(accuracy, 'b'); 
% plot(precision, 'g'); 
% plot(recall, 'c'); 
% plot(fscore, 'b--');
% 
% legend('general_err','train_err','accuracy','precision', 'recall', 'fscore', 'Location', 'northwest');