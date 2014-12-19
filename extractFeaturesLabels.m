function [features, labels, deleteFlag] = extractFeaturesLabels(ofdataFile, labelFile)

%% Initialization
ofTable = readtable(ofdataFile);
labelTable = readtable(labelFile);

% Parse table data
idx = ofTable.FrameIdx;
Xpos = ofTable.Xpos;
Ypos = ofTable.Ypos;

labelIdx = labelTable.FrameIdx;
action = labelTable.Action;

% Frame parameters
interval = 100;
skipIdx = 1;

%% Extract features and labels for each video clip
% Check if video clip is longer than interval, o/w discard
numFramesTotal = idx(end) - idx(1) + 1;
startFrame = idx(1);
numInt = ceil(numFramesTotal/interval)-1;

f_xposDiff = zeros(numInt,1);
f_stdMean = zeros(numInt,1);
f_cornerDensity = zeros(numInt,1);
labels = zeros(numInt,1);

if (idx(end) >= interval)
    for intIdx=1:numInt
        endFrame = min(idx(end),startFrame + interval - 1);
        disp(['startFrame: ' num2str(startFrame) '; endFrame: ' num2str(endFrame)])
        
        numFramesInt = endFrame-startFrame+1;
        % Construct data matrix
        txyMatrix = [idx(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')) Xpos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')) Ypos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last'))];
        
        % Find number of corners at each frame index
        numCorners = zeros(numFramesInt,1);
        for t=1:numFramesInt
            numCorners(t) = sum(txyMatrix((txyMatrix(:,1)==(startFrame+t-1)),1));
        end
        
        %     % DBSCAN
        %     k = floor(mean(numCorners));
        %     Eps = 5;
        %     [labels labelType] = dbscan(txyMatrix(:,1:2),k,Eps);
        %     clr = lines(max(labels));
        %     scatter(txyMatrix((labels>0),1),txyMatrix((labels>0),2), 48, clr(labels(labels>0),:), 'Marker', '.');
        
        % Feature 1: Diff in mean x-position at start and end of interval
        xposMean = zeros(numFramesInt,1);
        for t=1:numFramesInt % For each time index
            xposMean(t) = mean(txyMatrix((txyMatrix(:,1)==(startFrame+t-1)),2)); % calculate mean of all the x-position
        end
        f_xposDiff(intIdx) = xposMean(end) - xposMean(1);
        
        % Feature 2: Mean of standard deviation of x-position in interval
        xposStd = zeros(numFramesInt,1);
        for t=1:numFramesInt % For each time index
            xposStd(t) = std(txyMatrix((txyMatrix(:,1)==(startFrame+t-1)),2)); % calculate std of all the x-position
        end
        f_stdMean(intIdx) = mean(xposStd);
        
        % Feature 3: Number of detected corners in interval
        f_cornerDensity(intIdx) = sum(numCorners);
        
        % Advance frame interval
        startFrame = startFrame + interval;
    end
    
    % Assign labels to intervals
    deleteFlag = 0;
    for l=1:size(labelIdx,1)
        if ((ceil(labelIdx(l)/interval)) <= numInt)
            if strcmp(action(l), 'lturn') || strcmp(action(l), 'rturn')
                labels(ceil(labelIdx(l)/interval),1) = 1;
            elseif strcmp(action(l), 'stop') ... 
                    || strcmp(action(l), 'ignore') ...
                    || strcmp(action(l), 'rturn') || strcmp(action(l), 'lturn')
                labels(ceil(labelIdx(l)/interval),1) = 2; %ignore
            elseif strcmp(action(l), 'delete')
                deleteFlag = 1;
            end
        end
    end
    
%     % Assign label to interval
%     for l=1:size(labelIdx,1)
%         %     labels{floor(labelIdx(l)/interval)} = action(l);
%         if ((ceil(labelIdx(l)/interval)) <= numInt)
%             if strcmp(action(l), 'lchange')
%                 labels(ceil(labelIdx(l)/interval),1) = 1;
%             elseif strcmp(action(l), 'rchange')
%                 labels(ceil(labelIdx(l)/interval),1) = 2;
%             elseif strcmp(action(l), 'lturn')
%                 labels(ceil(labelIdx(l)/interval),1) = 3;
%             elseif strcmp(action(l), 'rturn')
%                 labels(ceil(labelIdx(l)/interval),1) = 4;
%             elseif strcmp(action(l), 'stop') || strcmp(action(l), 'eof') ... 
%                     || strcmp(action(l), 'ignore') || strcmp(action(l), 'end action')
%                 % do nothing
%             end
%         end
%     end
end

% %% Plot for sanity check
% subplot(2,2,1); scatter(idx, Xpos, 'Marker', '.');
% title('Data');
% subplot(2,2,2); plot(interval/2:interval:endFrame, f_xposDiff,'rx');
% title('Diff in mean xpos');
% subplot(2,2,3); plot(interval/2:interval:endFrame, f_stdDiff,'gx');
% title('Diff in std');
% subplot(2,2,4); plot(interval/2:interval:endFrame, f_cornerDensity,'bx');
% title('Num of corners');

%% Output Data
features = [f_xposDiff f_stdMean f_cornerDensity];