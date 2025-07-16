clear; clc; close all;
fformat = '.csv';
subjectName = {'S1-12'};
fileName{1,1} = strcat(subjectName{1,1},'_Dapi', fformat);
fileName{1,2} = strcat(subjectName{1,1},'_AC', fformat); % in green 
fileName{1,3} = strcat(subjectName{1,1},'_IC', fformat); % in red 
fileName{1,4} = strcat(subjectName{1,1},'_colocal', fformat); % in red 
path = pwd;

channel0{1,1} = csvread(fileName{1,1});

cc = [0 153 204; 164 211 238; 113 113 198; 255 215 0];
cc = cc/255; 
bbb = 200; 
sz = 5; 
%%
clearvars -except slide fformat fileName roi cc bbb sz ch_dist_stacked
for roi = 1:2
figure()
set(gcf,'Position',[200 200 1300 400]);
subplot(1,2,1)
for i = 1:size(fileName,2)
    channel0{1,i} = csvread(fileName{1,i});
    scatter(channel0{1,i}(:,1), channel0{1,i}(:,2), sz, cc(i,:),'filled') 
     hold on
    xlim([min(channel0{1,1}(:,1))-bbb max(channel0{1,1}(:,1))+bbb])
    ylim([min(channel0{1,1}(:,2))-bbb max(channel0{1,1}(:,2))+bbb])
    hold on
end
camroll(180)
%%%
h = impoly;
position = wait(h);
x_polygon = position(:, 1);
y_polygon = position(:, 2);
%%%
for i = 1:size(fileName,2)
    in_polygon{i} = inpolygon(channel0{1,i}(:,1), channel0{1,i}(:,2), x_polygon, y_polygon);
    channel{1,i} = channel0{1,i}(in_polygon{i},:);
end
%%%
subplot(1,2,2)
camroll(180) 
for i = 1:size(fileName,2)
    channel0{1,i} = csvread(fileName{1,i});
    scatter(channel0{1,i}(:,1), channel0{1,i}(:,2), sz, cc(i,:))
    scatter(channel{i}(:,1), channel{i}(:,2),sz, cc(i,:),'filled')
    hold on
    plot(x_polygon, y_polygon, 'k-', 'LineWidth', 2);
    xlim([min(channel0{1,1}(:,1))-bbb max(channel0{1,1}(:,1))+bbb])
    ylim([min(channel0{1,1}(:,2))-bbb max(channel0{1,1}(:,2))+bbb])
    hold on
end
%%%
threshold = 50;
degree = 2; 

x = channel0{1,1}(:,1);  y = channel0{1,1}(:,2);
%%%
figure()
scatter(x, y, 2, [0.6 0.6 0.6]);
camroll(180)
xlabel('Drawing ventral(red) and dorsal(blue) line')

%%% Denoising for Ventral line
h = imfreehand;
lineCoords = h.getPosition();
delete(h);
hold on
selected_Vent0 = [];
clear xPoint yPoint
for j = 1:length(x)
    xPoint = x(j);  yPoint = y(j);
    dist = sqrt((xPoint - lineCoords(:, 1)).^2 + (yPoint - lineCoords(:, 2)).^2);
    if min(dist) <= threshold
        selected_Vent0 = [selected_Vent0; xPoint, yPoint];
    end
end
scatter(selected_Vent0(:,1), selected_Vent0(:,2), 2, 'filled', 'r');
hold on
Outerline{1,2} = selected_Vent0;

%%% Denoising for Dorsal line
h2 = imfreehand;
lineCoords2 = h2.getPosition();
delete(h2);
hold on
selected_Dor0 = [];
clear xPoint2 yPoint2
for jj = 1:length(x)
    xPoint2 = x(jj);  yPoint2 = y(jj);
    dist = sqrt((xPoint2 - lineCoords2(:, 1)).^2 + (yPoint2 - lineCoords2(:, 2)).^2);
    if min(dist) <= threshold
        selected_Dor0 = [selected_Dor0; xPoint2, yPoint2];
    end
end
scatter(selected_Dor0(:,1), selected_Dor0(:,2), 2, 'filled', 'b');
hold on
Outerline{1,1} = selected_Dor0;

%%% regression line for Ventral line
clear xx yy coefficients x_vals y_vals
xx = selected_Vent0(:,1); yy = selected_Vent0(:,2);
coefficients = polyfit(xx, yy, degree);
x_vals = linspace(min(xx), max(xx), length(xx))';
y_vals = polyval(coefficients, x_vals);
plot(x_vals, y_vals, 'r', 'LineWidth', 2);
hold on

%%% regression line for Dorsal line
clear xx2 yy2 coefficient2s x_vals2 y_vals2
xx2 = selected_Dor0(:,1); yy2 = selected_Dor0(:,2);
coefficients2 = polyfit(xx2, yy2, degree);
x_vals2 = linspace(min(xx2), max(xx2), length(xx))';
y_vals2 = polyval(coefficients2, x_vals2);
plot(x_vals2, y_vals2, 'b', 'LineWidth', 2);
hold on

%%%
clear distance ch_distance
for k = 2:4
    if sum(isempty(channel{1,k})) ~= 1
        for s = 1:size(channel{1,k},1)
            temp_dot = channel{1,k}(s,:);
            for ii = 1:size(x_vals,1)
                distance(ii,1) = sqrt((x_vals(ii) - temp_dot(1,1))^2 + (y_vals(ii) - temp_dot(1,2))^2);
                distance(ii,2) = sqrt((x_vals2(ii) - temp_dot(1,1))^2 + (y_vals2(ii) - temp_dot(1,2))^2);
                distance(ii,3) = distance(ii,1) + distance(ii,2);
            end
            ch_distance{1,k-1}(s,1:3) = min(distance);
            ch_distance{1,k-1}(s,4) = ch_distance{1,k-1}(s,1)/ch_distance{1,k-1}(s,3);
        end
    else
        ch_distance{1,k-1} = [];
    end
    ch_dist_stacked{roi,k-1} = ch_distance{1,k-1};
end
%%%

figure(1000)
subplot(2,2,roi*2-1)
edges = [0:0.1:1];
for i = 1:size(ch_distance,2)
    h = histogram(ch_distance{1,i}, edges, 'FaceColor', cc(i+1,:))
    hold on
    counts{roi,i} = h.Values;
end
xlim([-0.05 1.05])
ylim([0 30])

xlabel('Normalized depth')
ylabel('cell number')
camroll(90)
legend('PPCAC','PPCIC','Location','northwest')
legend('boxoff')
% 
subplot(2,2,roi*2)
edges2 = [0.1:0.1:1]
total_Cellnum = size(ch_distance{1,1},1) + size(ch_distance{1,2},1); 
for i = 1:size(counts,2)
    counts_perc{roi,i} = counts{roi,i}/total_Cellnum;
    bar(edges2, counts_perc{roi,i}, 'FaceColor', cc(i+1,:),'FaceAlpha',.7,'EdgeAlpha',.7)
    hold on
    ylim([0 0.5])
    hold on
    xlabel('Normalized depth')
ylabel('cell fraction')
end
camroll(90)
legend('PPCAC','PPCIC','Location','northwest')
legend('boxoff')

end