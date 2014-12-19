% Output vectors are labelVector and featureVector
% You can run runSVM.m with these variables in the workspace and 
% the script will train/test using SVM churn out the confusion matrix
%
% The labels are "deviate", i.e. left/right lane changes, and "straight"
%
% As of 11/23, this script only uses "end action" as "straight", it
% ignores those cases where everything from the start to the first action
% label is "straight"

% Set storing of figures
printFig = 0;

% Set interval size
interval = 200;

% Set HOG parameters
cellSize = 16;
numBins = 9; % default 9

% Set crop window size
cropImg = 1;
cropWinSize = 200;

% Apply filtering to img
morphFilter = 0;
medFilter = 0;
medKernel = [2 2];

% Set input directories
train_labels = '/Users/ellakim/Documents/cs229proj/data/labels/HIGHWAY';
train_ofdata = '/Users/ellakim/Documents/cs229proj/data/ofdata_2/HIGHWAY';

% For each video clip, extract features and labels
ofdataFiles = dir(strcat(train_ofdata,'/*.csv'));
labelsFiles = dir(strcat(train_labels,'/*.csv'));

if (size(ofdataFiles,1) ~= size(labelsFiles,1))
    disp(strcat(datestr(now),': ERROR: missing data or labels'));
end

numFiles = size(ofdataFiles,1);

featureVector = [];
labelVector = [];
validVideoIdx = 1;

for i=1:numFiles % For each video
    ofdataFile = strcat(strcat(train_ofdata,'/'), ofdataFiles(i).name)
    labelFile = strcat(strcat(train_labels,'/'), labelsFiles(i).name);
    
    % Check if there is mismatched label and data files
    if ~strcmp(ofdataFiles(i).name(1:end-13), labelsFiles(i).name(1:end-11))
        % log error
        fid = fopen('errorLogFile','a+');
        fprintf(fid,'%s: mismatched label and data: %s\n',strcat(datestr(now)),labelsFiles(i).name(1:end-11));
        fclose(fid);
    end
    
    ofTable = readtable(ofdataFile);
    labelTable = readtable(labelFile);
    
    % Parse table data
    idx = ofTable.FrameIdx;
    Xpos = ofTable.Xpos;
    Ypos = ofTable.Ypos;
    labelIdx = labelTable.FrameIdx;
    action = labelTable.Action;
   
    % For each label, extract features
    for t=1:size(labelIdx,1)
        if strcmp(action{t},'delete')
            disp(strcat('Deleting: ', ofdataFile));
            break
        end
            
        if ~strcmp(action{t},'eof')
            startIdx = labelIdx(t);
            endIdx = startIdx + interval;
            
            % Construct data matrix
            % If start or end index does not contain a valid point, pick next
            % closest
            if sum(idx==startIdx)==0
                [NA, closest] = min(abs(idx-startIdx));
                startIdx = idx(closest);
            end
            if sum(idx==endIdx)==0
                [NA, closest] = min(abs(idx-endIdx));
                endIdx = idx(closest);
            end
            
%             txMatrix = [idx(find(idx==startIdx,1):find(idx==endIdx,1,'last')) ...
%                 Xpos(find(idx==startIdx,1):find(idx==endIdx,1,'last'))];
            txMatrix = [idx Xpos];
            
%             % If first action is not at frame 1, then make indices 1 to
%             % first action label "go"
%             if size(action,1) >= 1 && ~strcmp(action{1},'rchange')      
%             end
        

            % Shift points that have negative x positions
            % I have no idea why this happens, probably a bug in the
            % optical flow script
            negXpos = txMatrix(:,2)<0;
            if sum(negXpos) > 0
                txMatrix(negXpos,2) = 1;
            end
            
            if strcmp(action{t},'rchange') || strcmp(action{t},'lchange') 

%               plot(txMatrix(find(idx==startIdx,1):find(idx==endIdx,1),1),txMatrix(find(idx==startIdx,1):find(idx==endIdx,1),2), '.');
                
                % DBSCAN
%                 txMatrixInt = txMatrix(find(idx==startIdx,1):find(idx==endIdx,1),:);
%                 k = 8;
%                 Eps = 2;
%                 [labels labelType] = dbscan(txMatrixInt,k, Eps);
%                 clr = lines(max(labels));
%                 scatter(txMatrixInt((labels>0),1),txMatrixInt((labels>0),2), 48, clr(labels(labels>0),:), 'Marker', '.');

                % Extract HOG features
                % Contruct image representation of data for HOG extraction
                img = zeros(720,interval);
                for f=1:interval
                    curIdx = startIdx+f-1;
                    img(ceil(txMatrix(idx==curIdx,2)),f) = 1;
                end
                
                img = flipud(img);
                
                if morphFilter
                    se = strel('square',2);
                    img = imopen(img,se);
                end
                
                if medFilter
                    img = medfilt2(img, medKernel);
                end
                
                % Center image and crop image to reduce feature vector size
                if cropImg
                    meanXpos = round(mean(txMatrix(:,2)));
                    img = img((meanXpos-cropWinSize:meanXpos+cropWinSize),:);
                end

                if ~printFig
                    hog1 = extractHOGFeatures(img,'CellSize',[cellSize cellSize], 'NumBins', numBins);
                else
                    [hog1, visualization] = extractHOGFeatures(img,'CellSize',[cellSize cellSize], 'NumBins', numBins);
                    subplot(1,2,1);
                    imshow(img);
                    a = subplot(1,2,2);
                    plot(visualization)
                    title(a,'deviate')
                    print('-dpng', strcat('./images/deviate/',ofdataFiles(i).name(1:end-13)));
                end
                for numIterate = 1:2
                  featureVector(end+1,:) = hog1;
                  labelVector{end+1,1} = 'deviate';
                end
            end
            
            if strcmp(action{t},'end action')
%                 figure
%                 plot(txMatrix(:,1),txMatrix(:,2), '.');
                
                startIdx = startIdx + 50; % delay go action
                
                % Extract HOG features
                % Contruct image representation of data for HOG extraction
                img = zeros(720,interval);
                for f=1:interval
                    curIdx = startIdx+f-1;
                    img(ceil(txMatrix(idx==curIdx,2)),f) = 1;
                end
                
                img = flipud(img);
                
                if morphFilter
                    se = strel('square',2);
                    img = imopen(img,se);
                end
                
                if medFilter
                    img = medfilt2(img, medKernel);
                end
                
                % Center image and crop image to reduce feature vector size
                if cropImg
                    meanXpos = round(mean(txMatrix(:,2)));
                    img = img((meanXpos-cropWinSize:meanXpos+cropWinSize),:);
                end
                
                if ~printFig
                    hog1 = extractHOGFeatures(img,'CellSize',[cellSize cellSize], 'NumBins',numBins);
                else
                    [hog1, visualization] = extractHOGFeatures(img,'CellSize',[cellSize cellSize], 'NumBins', numBins);
                    subplot(1,2,1);
                    imshow(img);
                    a = subplot(1,2,2);
                    plot(visualization)
                    title(a,'straight');
                    print('-dpng', strcat('./images/straight/',ofdataFiles(i).name(1:end-13)));
                end
                
                featureVector(end+1,:) = hog1;
                labelVector{end+1,1} = 'straight';
            end
        end
    end % end of each label
end % end of each video

