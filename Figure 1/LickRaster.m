%%%%%%%%%%%%%% Jung & Lee et al, Parietal top-down projections balance flexibility and stability in adaptive learning 
%%%%%%%%%%%%%% Eunji Jung, 2025-May-02

clear; clc; close all
    prompt = 'log file name?'; 
    path = inputdlg(prompt);
    filename = fopen(strcat(path{1,1}, '.log'));
    %filename = fopen('Eg_LickRaster_AudRL_naive_delay500ms.log')
fromtxt=textscan(filename,'%s %s %s %s %s %s %s %s %s %s %s %s %s','Delimiter','\t');
%%%
%%% reading txt
num_col4 = str2double(fromtxt{1,4});  
num_col5 = str2double(fromtxt{1,5});  
sum_col3 = cellfun(@(x) sum(double(x)), fromtxt{1,3}); 

mask1 = (num_col4 > 1) | (num_col4 == 1 & sum_col3 == 732);
reftxt1 = find(mask1); 
temp_behav = [num_col4(mask1), num_col5(mask1)];
temp_ms = temp_behav(:,1);
temp_ms(:,2) = floor(temp_behav(:,2)/10);

mask2 = (sum_col3 == 847);
lick = [num_col4(mask2), num_col5(mask2)];
lick_time_temp = floor(lick(:,2)/10); % in to ms
deterror = diff(lick_time_temp);
idx = find(deterror > 100) + 1;
lick_time_ms = [lick_time_temp(1); lick_time_temp(idx)];

%%%% parameter settings : change the duration (ms) here for the lickraster plot 
stim = 500;     rw = 2000;      pre = 1000;      delay = 1000;      post = 6000; 
Antperiod = stim+delay;
stimperiod = 6000; %stim+delay+rw; 

%%% port code
Tra_R1_5kHz = 211;      Tra_R1_10kHz = 212;     Cond_R1_5kHz = 1211;    Cond_R1_10kHz = 1212;
Tra_R2_5kHz = 2112;     Tra_R2_10kHz = 2122;    Cond_R2_5kHz = 12112;   Cond_R2_10kHz =12122;   BlockChangeCode = 888;
CS_code = [1211 1212 12122 12112];      Tra_code = [211 212 2122 2112];
Correct_code = [351 411 352 412];
if sum(find(temp_ms(:,1) == Cond_R2_5kHz)) %% if task includes classical conditioning stage (CS): positive reinforcer regardless of the Go-lick 
    CS_type = 1;
    target_5kHz = [211 1211 2112 12112];        target_10kHz = [212 1212 2122 12122];
    target_stim = [211 1211 2112 12112 212 1212 2122 12122];
else
    CS_type = 0; 
    target_5kHz = [211 2112];                   target_10kHz = [212 2122];
    target_stim = [211 2112 212 2122];
end
perform_code = [351 451 311 411 352 452 312 412];
target = horzcat(target_stim, perform_code);

%%%
BlockChangeTime = temp_ms(temp_ms(:,1) == BlockChangeCode, 2);
if isempty(BlockChangeTime)
    if CS_type == 1
        temp_ms(:,1) = temp_ms(:,1) * 10;
    end
    ChangePoint = find(abs(diff(temp_ms(:,2))) > 5000);

    if ~isempty(ChangePoint)
        pairedPoints = [ChangePoint(1:2:end-1), ChangePoint(2:2:end)];
        BlockChangeTime = ceil(mean(temp_ms(pairedPoints + 1, 1), 2));
    end
end

BlockChangeTime(end+1) = max(temp_ms(:,1)) + 1e7;
BlockChangeNum = numel(BlockChangeTime);
BlockTime = BlockChangeTime(min(2, BlockChangeNum));

mask = ismember(temp_ms(:,1), target);
temp_rev = temp_ms(mask,:);
temp_1st = temp_rev(temp_rev(:,2) < BlockTime, :);

[Tra_rows, Tra_cols] = size(Tra_code);
Stim_Tra = arrayfun(@(x) temp_1st(temp_1st(:,1) == x, 2), Tra_code, 'UniformOutput', false);
if CS_type ~= 0
    Stim_CS = arrayfun(@(x) temp_1st(temp_1st(:,1) == x, 2), CS_code, 'UniformOutput', false);
else
    Stim_CS = {};
end

% window setting 
antWindow = @(o) lick_time_ms > o & lick_time_ms < o + Antperiod;
stimWindow = @(o) lick_time_ms > o - pre & lick_time_ms < o + stimperiod;

if CS_type == 1 
    for s = [1 3 2 4]
        onsets = Stim_CS{1,s}(:,1);
        numTrials = numel(onsets);

        if ismember(s, [1 3]) % Go stim 
            condMask = arrayfun(@(o) any(antWindow(o)), onsets); % anticipatory lick 
            correctVal = condMask; % Hit(1) / Miss(0)
        else % No-go stim 
            condMask = arrayfun(@(o) ~any(antWindow(o)), onsets); % no anticipatory lick 
            correctVal = condMask; % Correct rejection(1) / FA(0)
        end
        
        Stim_CS_perform{1,s*2-1} = cell(sum(condMask),2);
        Stim_CS_perform{1,s*2} = cell(sum(~condMask),2);
        allLicks = arrayfun(@(o) lick_time_ms(stimWindow(o)) - o, onsets, 'UniformOutput', false);
        
        Stim_CS_perform{1,s*2-1}(:,1:2) = [num2cell(onsets(condMask)), allLicks(condMask)];
        Stim_CS_perform{1,s*2}(:,1:2) = [num2cell(onsets(~condMask)), allLicks(~condMask)];

        Stim_CS_lick{1,s} = [onsets, double(condMask)]; 
        Stim_CS_correct{1,s} = [onsets, double(correctVal)]; 
    end
end

for s = [1 3 2 4]
    for i = 1:size(Stim_Tra,1)
        onsets = Stim_Tra{i,s}(:,1);
        numTrials = numel(onsets);

        if ismember(s, [1 3]) % Go stim 
            condMask = arrayfun(@(o) any(antWindow(o)), onsets); % lick during response window 
            correctVal = condMask;
        else % No-go stim 
            condMask = arrayfun(@(o) ~any(antWindow(o)), onsets);
            correctVal = condMask;
        end
        
        Stim_Tra_perform{i,s*2-1} = cell(sum(condMask),3);
        Stim_Tra_perform{i,s*2} = cell(sum(~condMask),3);

        allLicks = arrayfun(@(o) lick_time_ms(stimWindow(o)) - o, onsets, 'UniformOutput', false);
        traInfo = repmat(i, numTrials, 1);

        Stim_Tra_perform{i,s*2-1}(:,1:3) = [num2cell(onsets(condMask)), allLicks(condMask), num2cell(traInfo(condMask))];
        Stim_Tra_perform{i,s*2}(:,1:3) = [num2cell(onsets(~condMask)), allLicks(~condMask), num2cell(traInfo(~condMask))];
        
        Stim_Tra_lick{i,s} = [onsets, double(condMask)];
        Stim_Tra_correct{i,s} = [onsets, double(correctVal)];
    end
end

[numRows, numCols] = size(Stim_Tra_perform);
num_perform(1:numRows, :) = cellfun(@(x) size(x,1), Stim_Tra_perform);

Stim_All_perform = Stim_Tra_perform;

if CS_type == 1 
    validCells = cellfun(@(c,t) ~isempty(c) && ~isempty(t), Stim_CS_perform, Stim_Tra_perform);
    Stim_All_perform(validCells) = arrayfun(@(i) ...
        sortrows([Stim_Tra_perform{i}(:,1); Stim_CS_perform{i}(:,1)]), ...
        find(validCells), 'UniformOutput', false);
    num_CS_perform = cellfun(@(x) size(x,1), Stim_CS_perform);
    num_perform(4,:) = num_CS_perform;
    num_perform(5,:) = num_perform(1,:) + num_CS_perform;
end


tnum_Tra = size(Stim_Tra_correct{1,1},1)*2;
tnum_CS = length(find(temp_1st(:,1) == CS_code));
tnum_total = length(find(temp_1st(:,1) == target_stim));
BlockSize = [tnum_Tra tnum_CS tnum_total-(tnum_Tra+tnum_CS)];
BlockSize2 = cumsum(BlockSize);

correct_TS_bi = cell(1,2);
correct_TS_bi{1} = sortrows([Stim_Tra_correct{1,1}; Stim_Tra_correct{1,2}]);
correct_TS_bi{2} = sortrows([Stim_Tra_correct{1,3}; Stim_Tra_correct{1,4}]);

if CS_type == 1
    correct_CS_bi = cell(1,2);
    correct_CS_bi{1} = sortrows([Stim_CS_correct{1,1}; Stim_CS_correct{1,2}]);
    correct_CS_bi{2} = sortrows([Stim_CS_correct{1,3}; Stim_CS_correct{1,4}]);
    correct_All = cellfun(@(cs,ts) sortrows([cs; ts]), correct_CS_bi, correct_TS_bi,'UniformOutput', false);
else
    correct_All = correct_TS_bi;
end

smth = 11;
kernel = ones(smth,1)/smth;

for i = 1:size(correct_All,2)
    if ~isempty(correct_All{i})
        correct_All{i}(:,4) = conv2(correct_All{i}(:,2), kernel, 'same');
    end
    if ~isempty(correct_TS_bi{i})
        correct_TS_bi{i}(:,4) = conv2(correct_TS_bi{i}(:,2), kernel, 'same');
    end
end


%%% TTR : Trial number To Reversal 
performcri = 0.75; consec = 2; consec_tnum = 3; Rev_point0 = []; Rev_cri_Tra =[];

%%% Finding reversal point: Correct rate over 0.7 for Consec 3 trial
for i = 1:size(correct_All,2)
    tnum = 0;
    for t = 1:size(correct_All{1,i})-consec
        if sum(correct_All{1,i}(t:t+consec,4) >= performcri) == consec_tnum
            tnum = tnum+1;
            Rev_point0{1,i}(tnum,1) = t+floor(smth/2)-1;
            Rev_point0{1,i}(tnum,2) = correct_All{1,i}(t+consec,1);
        end
    end
end
%%%
clear Rev_point1 Rev_point
if size(Rev_point0,2) > 1
    for i = 1:size(Rev_point0,2)
        Rev_point{1,i} = Rev_point0{1,i}(find(diff(Rev_point0{1,i}(:,1))>1)+1,:);
        Rev_point1{1,i} = Rev_point0{1,i}(1,:);
        Rev_point{1,i} = sortrows(vertcat(Rev_point{1,i}, Rev_point1{1,i}));
    end
else % Inflexible sessions
    Rev_point{1,i}(1,1) = BlockSize2(1,2);
    Rev_point{1,i}(1,2) = correct_All{1,2}(end,1);
end



%%% calculate performance before and after the TTR 
[~, numCols] = size(Stim_All_perform);
Stim_All_perform_num = zeros(3, numCols);

for i = 1:numCols
    if ~isempty(Stim_All_perform{1,i})
        onsets = cell2mat(Stim_All_perform{1,i}(:,1));
        revPoint = Rev_point{2}(1,2);
       
        mask = onsets < revPoint;
        Stim_All_perform{2,i} = Stim_All_perform{1,i}(mask,:);
        Stim_All_perform{3,i} = Stim_All_perform{1,i}(~mask,:);
        Stim_All_perform_num(:,i) = [sum(~mask); sum(mask); size(Stim_All_perform{1,i},1)]';
    end
end


%%% collecting lick event according to the stim type 
stimtype_lick = cell(1,2);
[is5kHz, is10kHz] = deal(ismember(temp_ms(:,1), target_5kHz), ismember(temp_ms(:,1), target_10kHz));
timeMask = temp_ms(:,2) < BlockTime + 1000000;
stimData = {temp_ms(is5kHz & timeMask,2), temp_ms(is10kHz & timeMask,2)};

for i = 1:2
    onsets = stimData{i};
    lickMask = @(o) lick_time_ms > o - pre & lick_time_ms < o + stimperiod;
    delayLick = @(o) any(lick_time_ms > o + stim + delay & lick_time_ms < o + stimperiod);
    stimtype_lick{i} = arrayfun(@(o) {...
        lick_time_ms(lickMask(o)) - o, ...   % lick time
        find(onsets == o), ...               % trial num
        1 + delayLick(o) ...                 # whether lick 
    }, onsets, 'UniformOutput', false);
end

if CS_type == 1
    [csTrials, traTrials] = deal(size(Stim_CS{1,4},1)*2, size(Stim_Tra{1,1},1)*2);
    Stage = repmat(traTrials, BlockChangeNum*2, 1);
    Stage(2:2:end) = csTrials;
else
    traTrials = size(Stim_Tra{1,1},1)*2;
    Stage = repmat(traTrials, BlockChangeNum, 1);
end
Stage_num = cumsum(Stage);



%%% plotting lick raster plot 
sz = 3; sz2 = 7; whetherlick = 500;
prelick = 1000;
f = figure('Position',[400 100 400 500], 'Renderer','painters');
title(path);
x_limits = [-prelick stimperiod+whetherlick];
stage_y = Stage_num(:,1)/2;
rev_y = Stage_num(1,1)/2 + Rev_point{1,2}(1,1)/2;
tnum = 200; 
for s = 1:size(stimtype_lick,2)
    ax = subplot(1,numel(stimtype_lick),s);
    hold(ax, 'on');
    plot(ax, x_limits, [stage_y stage_y], 'k-');
    plot(ax, x_limits, [rev_y rev_y], 'g-', 'LineWidth',1.5); % TTR

    currData = stimtype_lick{s};
    [lick_times, trial_nums, lick_types] = deal([]);
    
    for i = 1:numel(currData)
        trialLicks = currData{i}{1};
        trialNum = currData{i}{2};
        lickType = currData{i}{3};
        lick_times = [lick_times; trialLicks(:)];
        trial_nums = [trial_nums; repmat(trialNum, numel(trialLicks),1)];
        lick_types = [lick_types; repmat(lickType, numel(trialLicks),1)];
    end

    if ~isempty(lick_times)
        scatter(ax, lick_times, trial_nums, sz, 'k', 'filled');
    end

    binary_x = stimperiod + whetherlick;
    trial_ids = cellfun(@(x) x{2}, currData);
    is_type1 = cellfun(@(x) x{3} == 1, currData);
    
    if any(is_type1)
        scatter(ax, binary_x*ones(nnz(is_type1),1), trial_ids(is_type1), sz, 'c', 'filled');
    end
    if any(~is_type1)
        scatter(ax, binary_x*ones(nnz(~is_type1),1), trial_ids(~is_type1), sz, 'm', 'filled');
    end
    
    xlim(ax, x_limits);
    y_max = max([tnum, trial_ids(:)'+5]);
    ylim(ax, [0 y_max]);
    xlabel('Time (ms)');
    ylabel('Trials');

    if s == 1
        line(ax, [0 0], ylim, 'Color', 'b', 'LineWidth',1.5);
        title('5kHz');
    else
        line(ax, [0 0], ylim, 'Color', 'r', 'LineWidth',1.5);
        title('10kHz');
    end
end
