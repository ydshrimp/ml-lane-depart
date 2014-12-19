clear all
close all
table = readtable('opticalflow_meandisp_log.csv');

time = table.TimeStamp;
disp = table.MeanXDisplacement;

hold on
plot(time, zeros(size(disp,1)));
plot(time, disp);

ma_size = 20;
a = 1;
b = ones(1,ma_size); b = b./ma_size;
disp_smooth = filter(b,a,disp);
plot(time, disp_smooth, 'LineWidth', 3)