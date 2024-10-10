% Plotting

data = readtable('output.csv');
wheel_torque = readtable('wheel_torque.csv');

p = data{:, 1:3};
q = data{:, 4:7};
v = data{:, 8:10};
w = data{:, 11:13};
jp = data{:, 14:17};
jv = data{:, 18:21};
cmd = data{:, 22:26};
f = data{:, 27};
qd = data{:, 28:31};


ct_inds = find(diff(f > 0.1));
figure(1);
clf
subplot(2,1,1)
plot(p)
legend('x', 'y', 'z')
subplot(2,1,2)
plot(v)
legend('vx', 'vy', 'vz')

figure(2)
clf
plot(q)
legend('qw', 'qx', 'qy', 'qz')

figure(3)
clf
plot(qd)
legend('qdw', 'qdx', 'qdy', 'qdz')

figure(4)
clf
plot(jp(:, 1))
hold on
plot(jv(:, 1))
legend('foot pos', 'foot vel')

figure(5)
clf
plot(jv(:, 2:4))
xline(ct_inds);
legend('w1', 'w2', 'w3')

figure(6)
clf
plot(f)
legend('contact')

figure(7)
clf
plot(w)
legend('wx', 'wy', 'wz')

figure(8)
plot(wheel_torque{1:4:end, :})
xline(ct_inds)
legend('tau1', 'tau2', 'tau3')

figure(9)
clf
hold on
plot(diff(w(:, 3)))
plot(wheel_torque{:, 3} / -150 / 3)
plot(jv(:, 4) / 50, '--')

bad = abs(diff(w(:, 3)) - (wheel_torque{1:size(w, 1) - 1, 3} / -150 / 3)) > 0.01;
bad_inds = find(diff(bad));
xline(bad_inds, 'k')

%% 
bad_state = data{447, :};
