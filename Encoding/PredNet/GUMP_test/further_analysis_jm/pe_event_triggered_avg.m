function [avg_traces, time_vector] = pe_event_triggered_avg(pe_signal, scenes_csv_file, event_types, win_pre_sec, win_post_sec, TR_sec)

% pe_signal: [T x 1] vector (BOLD TR-aligned)
% scenes_csv_file: path to scenes.csv
% event_types: cell array of event labels to test (matching scene names in csv)
% win_pre_sec, win_post_sec: pre/post window in seconds
% TR_sec: TR duration in sec (2.47 sec)

% Load scenes.csv
tbl = readtable(scenes_csv_file, 'ReadVariableNames', false);
frame_idx = tbl.Var1;
scene_name = tbl.Var2;

% Convert frame_idx to TR indices
fps = 25;
frame_sec = frame_idx / fps;
TR_idx = round(frame_sec / TR_sec);

n_TR = length(pe_signal);
win_pre_TR = round(win_pre_sec / TR_sec);
win_post_TR = round(win_post_sec / TR_sec);

time_vector = (-win_pre_TR:win_post_TR) * TR_sec;

% Loop over event types
avg_traces = zeros(length(event_types), length(time_vector));

for e = 1:length(event_types)
    evt = event_types{e};
    % Find events of this type
    evt_idx = find(strcmp(scene_name, evt));
    evt_TR_idx = TR_idx(evt_idx);
    
    % Aggregate windowed signals
    n_valid = 0;
    evt_matrix = [];
    
    for i = 1:length(evt_TR_idx)
        center = evt_TR_idx(i);
        win_idx = (center - win_pre_TR):(center + win_post_TR);
        if all(win_idx >= 1 & win_idx <= n_TR)
            evt_matrix = [evt_matrix; pe_signal(win_idx)'];
            n_valid = n_valid + 1;
        end
    end
    
    if n_valid > 0
        avg_traces(e,:) = mean(evt_matrix,1);
    else
        warning(['No valid events for ', evt]);
    end
end

% Plot
figure;
plot(time_vector, avg_traces', 'LineWidth',2);
legend(event_types, 'Interpreter','none');
xlabel('Time (s) relative to event');
ylabel('PE signal');
title('Event-triggered PE signal');
grid on;

end
