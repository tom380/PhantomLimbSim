clear; clc; clf;

% Load data
implicit     = load('../outputs/solver_implicit.mat');
implicitfast = load('../outputs/solver_implicitfast.mat');
euler        = load('../outputs/solver_euler.mat');
rk4          = load('../outputs/solver_rk4.mat');
reference    = load('C:/Users/Tom/OneDrive - University of Twente/000 Thesis/MATLAB Data Analysis/paper/barrutia.mat').export;

% Original moment signals
m_r  = mean(reference{4,3}, 2);        % reference (mean over trials)
m_i  = implicit.moment_actuator;
m_if = implicitfast.moment_actuator;
m_e  = euler.moment_actuator;
m_rk = rk4.moment_actuator;

% Make sure everything is column vectors
m_r  = m_r(:);
m_i  = m_i(:);
m_if = m_if(:);
m_e  = m_e(:);
m_rk = m_rk(:);

% Normalized "time"/phase vectors for each signal
phase_r  = linspace(0, 1, length(m_r))';               % reference
phase_i  = implicit.time(:)     / 2 - 1;               % implicit
phase_if = implicitfast.time(:) / 2 - 1;               % implicitfast
phase_e  = euler.time(:)        / 2 - 1;               % euler
phase_rk = rk4.time(:)          / 2 - 1;               % rk4

% Target common phase grid (2000 samples)
Nsamp = 2000;
phase = linspace(0, 1, Nsamp)';

% Resample all moment signals to the common phase grid
m_r_rs  = interp1(phase_r,  m_r,  phase, 'pchip');
m_i_rs  = interp1(phase_i,  m_i,  phase, 'pchip');
m_if_rs = interp1(phase_if, m_if, phase, 'pchip');
m_e_rs  = interp1(phase_e,  m_e,  phase, 'pchip');
m_rk_rs = interp1(phase_rk, m_rk, phase, 'pchip');

%% ---- Metrics: correlation and RMSE vs reference ----
% Helper inline for RMSE
rmse = @(x, y) sqrt(mean((x - y).^2));

% Define phase ranges
stance_idx = phase < 0.35;   % 0–35%
swing_idx  = phase >= 0.35;  % 35–100%

% ---- Full gait cycle metrics ----
R_full_implicit     = corr(m_r_rs,  m_i_rs);
R_full_implicitfast = corr(m_r_rs,  m_if_rs);
R_full_euler        = corr(m_r_rs,  m_e_rs);
R_full_rk4          = corr(m_r_rs,  m_rk_rs);

RMSE_full_implicit     = rmse(m_r_rs, m_i_rs);
RMSE_full_implicitfast = rmse(m_r_rs, m_if_rs);
RMSE_full_euler        = rmse(m_r_rs, m_e_rs);
RMSE_full_rk4          = rmse(m_r_rs, m_rk_rs);

% ---- Stance phase metrics (0–35%) ----
R_stance_implicit     = corr(m_r_rs(stance_idx),  m_i_rs(stance_idx));
R_stance_implicitfast = corr(m_r_rs(stance_idx),  m_if_rs(stance_idx));
R_stance_euler        = corr(m_r_rs(stance_idx),  m_e_rs(stance_idx));
R_stance_rk4          = corr(m_r_rs(stance_idx),  m_rk_rs(stance_idx));

RMSE_stance_implicit     = rmse(m_r_rs(stance_idx), m_i_rs(stance_idx));
RMSE_stance_implicitfast = rmse(m_r_rs(stance_idx), m_if_rs(stance_idx));
RMSE_stance_euler        = rmse(m_r_rs(stance_idx), m_e_rs(stance_idx));
RMSE_stance_rk4          = rmse(m_r_rs(stance_idx), m_rk_rs(stance_idx));

% ---- Swing phase metrics (35–100%) ----
R_swing_implicit     = corr(m_r_rs(swing_idx),  m_i_rs(swing_idx));
R_swing_implicitfast = corr(m_r_rs(swing_idx),  m_if_rs(swing_idx));
R_swing_euler        = corr(m_r_rs(swing_idx),  m_e_rs(swing_idx));
R_swing_rk4          = corr(m_r_rs(swing_idx),  m_rk_rs(swing_idx));

RMSE_swing_implicit     = rmse(m_r_rs(swing_idx), m_i_rs(swing_idx));
RMSE_swing_implicitfast = rmse(m_r_rs(swing_idx), m_if_rs(swing_idx));
RMSE_swing_euler        = rmse(m_r_rs(swing_idx), m_e_rs(swing_idx));
RMSE_swing_rk4          = rmse(m_r_rs(swing_idx), m_rk_rs(swing_idx));

% Collect everything in a single table
metrics_tbl = table( ...
    {'implicit'; 'implicitfast'; 'euler'; 'rk4'}, ...
    [R_full_implicit; R_full_implicitfast; R_full_euler; R_full_rk4], ...
    [RMSE_full_implicit; RMSE_full_implicitfast; RMSE_full_euler; RMSE_full_rk4], ...
    [R_stance_implicit; R_stance_implicitfast; R_stance_euler; R_stance_rk4], ...
    [RMSE_stance_implicit; RMSE_stance_implicitfast; RMSE_stance_euler; RMSE_stance_rk4], ...
    [R_swing_implicit; R_swing_implicitfast; R_swing_euler; R_swing_rk4], ...
    [RMSE_swing_implicit; RMSE_swing_implicitfast; RMSE_swing_euler; RMSE_swing_rk4], ...
    'VariableNames', { ...
        'solver', ...
        'corr_full',  'rmse_full', ...
        'corr_stance','rmse_stance', ...
        'corr_swing', 'rmse_swing' ...
    } ...
);

% Show metrics in command window
disp(metrics_tbl);

% Put everything in a table and export to CSV
moments_tbl = table(phase, m_r_rs, m_i_rs, m_if_rs, m_e_rs, m_rk_rs, ...
    'VariableNames', {'phase', 'reference', 'implicit', 'implicitfast', 'euler', 'rk4'});

writetable(moments_tbl, 'solvers.csv');

% Optional: plot the resampled signals
figure; hold on;
plot(phase, m_r_rs,  'DisplayName', 'reference');
plot(phase, m_i_rs,  'DisplayName', 'implicit');
plot(phase, m_if_rs, 'DisplayName', 'implicitfast');
plot(phase, m_e_rs,  'DisplayName', 'euler');
plot(phase, m_rk_rs, 'DisplayName', 'rk4');
xlim([0 1]);
legend('Location','best');
xlabel('Gait phase (-)');
ylabel('Moment (units)');
title('Resampled moments (2000 samples)');
grid on;
