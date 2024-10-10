%% What is happening in IsaacSim
clear; clc;
set(groot, 'DefaultAxesFontSize', 17); % Set default font size for axes labels and ticks
set(groot, 'DefaultTextFontSize', 17); % Set default font size for text objects
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex') % Set interpreter for axis tick labels
set(groot, 'DefaultTextInterpreter', 'latex'); % Set interpreter for text objects (e.g., titles, labels)
set(groot, 'DefaultLegendInterpreter', 'latex');
set(groot, 'DefaultFigureRenderer', 'painters');
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'DefaultLineMarkerSize', 15);

dt = 0.005;                            % dt for 200Hz sim
Iw = 0.00111;                          % Wheel inertia about z
It = 0.02210 + 2 * 0.00056 + 0.00032 + 0.00111;  % Torso + 2 wheel + foot about z

%% Plot

% Armature sweep
% files = {'output_t1_a0_001.csv', 'output_t1_a0_01.csv', 'output_t1_a0_1.csv'};
% tags = {'Torque: 1.0, Armature: 0.001', 'Torque: 1.0, Armature: 0.01', 'Torque: 1.0, Armature: 0.1'};
% arma = [0.001, 0.01, 0.1];

% Torque sweep
% files = {'output_t0_5_a0_01.csv', 'output_t1_a0_01.csv', 'output_t2_a0_01.csv'};
% tags = {'Torque: 0.5, Armature: 0.01', 'Torque: 1.0, Armature: 0.01', 'Torque: 2.0, Armature: 0.01'};
% arma = [0.01, 0.01, 0.01];

% Recent
% files = {'output_t1_a0_01.csv', 'output1.csv', 'output.csv'};
% tags = {'v_{lim} = 1000', 'v_{lim} = 10000', 'v_{lim} = 100000'};
% arma = [0.01, 0.01, 0.01];

% Recent
files = {'output.csv'};
tags = {'v_{lim} = 100000'};
arma = [0.01];
torque = [2];
fn = 1;

k_max = 250;
t = [1:k_max] * dt;

data = cell(size(files));
for ii = 1:size(files, 2)
    data{ii} = readtable(files{ii});
end

figure(fn);
clf;

subplot(2, 2, 1)
hold on
for ii = 1:size(data, 2)
    d = data{ii};
    plot(t, d{1:k_max, 13}, DisplayName=tags{ii});
end
legend
title("Body Angular Velocity, z")

subplot(2, 2, 2)
hold on
for ii = 1:size(data, 2)
    d = data{ii};
    plot(t, d{1:k_max, 21}, DisplayName=tags{ii});
end
title("Wheel Angular Velocity, z")

subplot(2, 2, 3)
hold on
for ii = 1:size(data, 2)
    d = data{ii};
    plot(t(1:end-1), diff(d{1:k_max, 13}) / dt, DisplayName=tags{ii});
end
% set(gca, 'ColorOrderIndex', 1)
% for ii = 1:size(data, 2)
%     a = arma(ii);
%     plot(t(1:end-1), -ones(1, size(t, 2) - 1) / It, '-.')
% end
set(gca, 'ColorOrderIndex', 1)
for ii = 1:size(data, 2)
    a = arma(ii);
    plot(t(1:end-1), -torque(ii) * ones(1, size(t, 2) - 1) / (It), '-.')
end
if size(data{1}, 2) > 35
    for ii = 1:size(data, 2)
        d = data{ii};
        plot(t, d{1:k_max, end - 4}, '--', DisplayName=tags{ii});
    end
end
title("Body Angular Acceleration, z")

subplot(2, 2, 4)
hold on
for ii = 1:size(data, 2)
    d = data{ii};
    plot(t(1:end-1), diff(d{1:k_max, 21}) / dt, DisplayName=tags{ii});
end
% set(gca, 'ColorOrderIndex', 1)
%     a = arma(ii);
%     plot(t(1:end-1), -ones(1, size(t, 2) - 1) / Iw, '-.')
% end
set(gca, 'ColorOrderIndex', 1)
for ii = 1:size(data, 2)
    a = arma(ii);
    plot(t(1:end-1), torque(ii) * ones(1, size(t, 2) - 1) / (Iw + a), '-.')
end
if size(data{1}, 2) > 35
    for ii = 1:size(data, 2)
        d = data{ii};
        plot(t, d{1:k_max, end}, '--', DisplayName=tags{ii});
    end
end
title("Wheel Angular Acceleration, z")