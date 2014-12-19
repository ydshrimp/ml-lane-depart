clear all
close all
ofTable = readtable('./train_features/ajunction_features.csv');
atTable = readtable('annotation.csv');

% Parse table data
idx = ofTable.FrameIdx;
Xpos = ofTable.Xpos;
Ypos = ofTable.Ypos;

actions = zeros(size(Xpos));
atIdx = atTable.FrameIdx;
atAction = atTable.Action;

% Start and end data indices
startIdx = 1;
endIdx = size(ofTable.Xpos,1);
% endIdx = 2500;
skipIdx = 1;

% Start and end frame indices
interval = 200;
startFrame = idx(3000);
% endFrame = idx(end);
endFrame = startFrame + interval -1 ;

% Construct data matrix
xytMatrix = [idx(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')) Xpos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')) Ypos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last'))];

% hold on
% scatter3(idx(startIdx:skipIdx:endIdx),Xpos(startIdx:skipIdx:endIdx),Ypos(startIdx:skipIdx:endIdx),'filled')
plot(idx(startIdx:skipIdx:endIdx),Xpos(startIdx:skipIdx:endIdx),'.')
% xlabel('frame idx'); ylabel('x-pos'); zlabel('y-pos');

% Plot Y-pos against X-pos
% plot(Xpos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')),Ypos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')),'.')
% xlabel('x-pos'); ylabel('y-pos');
% figure; scatter3(idx(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')),Xpos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')),Ypos(find(idx==startFrame,1):skipIdx:find(idx==endFrame,1,'last')),'filled')

% K-means clustering
K = 2;
[cidx, C] = kmeans(xytMatrix, K);

figure; hold on
clr = lines(K);
scatter3(xytMatrix(:,1),xytMatrix(:,2), xytMatrix(:,3), 48, clr(cidx,:), 'Marker','.')
scatter3(C(:,1), C(:,2), C(:,3), 100, clr, 'Marker','o', 'LineWidth',3)
hold off

% Plot annotation lines
% for i=1:size(atIdx,1)
%     hline = line([atIdx(i) atIdx(i)], ylim);
%     hline.Color = 'r';
% end

% hline = lsline;
% hline.Color = 'r';

% Sum points within an interval
