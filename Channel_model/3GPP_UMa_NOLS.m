%% ================= Khởi tạo môi trường =================
clear all; close all; clc;

addpath(genpath('E:\QuaDRiGa-main'));  
savepath;  % Lưu đường dẫn để MATLAB nhớ

%% ================= Tham số mô phỏng =================
s = qd_simulation_parameters;      % Đối tượng tham số mô phỏng
s.center_frequency = 3.5e9;        % Tần số trung tâm = 3.5 GHz (band C 5G)

%% ================= Tạo layout =================
l = qd_layout(s);                          % Layout mô phỏng (BS + UE)
l.set_scenario('3GPP_38.901_UMa_NLOS');    % Kịch bản kênh: Urban Macro NLOS

%% ================= Antenna Array cho BS =================
% Tạo ma trận toạ độ anten (4x4 = 16 phần tử)
[x_bs, y_bs] = meshgrid(0:0.5:1.5, 0:0.5:1.5);  % spacing 0.5 λ
z_bs = zeros(size(x_bs));                       % nằm trên mặt phẳng z=0
pos_tx = [x_bs(:), y_bs(:), z_bs(:)]';          % [3 x 16]

% Khởi tạo anten BS
tx_ant = qd_arrayant();               
tx_ant.element_position = pos_tx;               % gán toạ độ phần tử
tx_ant.no_elements = size(pos_tx,2);            % số phần tử = 16
l.tx_array = tx_ant;                            % gán anten cho layout (BS)

%% ================= Antenna Array cho UE =================
% Tạo ma trận toạ độ anten (2x2 = 4 phần tử)
[x_ue, y_ue] = meshgrid(0:0.5:0.5, 0:0.5:0.5);  
z_ue = zeros(size(x_ue));                       
pos_rx = [x_ue(:), y_ue(:), z_ue(:)]';          % [3 x 4]

% Khởi tạo anten UE
rx_ant = qd_arrayant();
rx_ant.element_position = pos_rx;
rx_ant.no_elements = size(pos_rx,2);            % số phần tử = 4
l.rx_array = rx_ant;                            % gán anten cho layout (UE)

%% ================= Cấu hình BS =================
l.no_tx = 4;     % số BS = 4
l.tx_position = [150  150 -150 -150;    % x
                 150 -150  150 -150;    % y
                 25   25   25   25];    % z (chiều cao BS = 25 m)

%% ================= Cấu hình UE =================
l.no_rx = 50;                                     % số UE = 50
l.rx_position(1,:) = rand(1,l.no_rx)*500 - 250;   % x: phân bố ngẫu nhiên [-250,250]
l.rx_position(2,:) = rand(1,l.no_rx)*500 - 250;   % y: tương tự
l.rx_position(3,:) = 1.5;                         % z: cao 1.5 m (người dùng)

%% ================= Sinh kênh truyền =================
l.set_pairing;         % Gán cặp BS–UE
c = l.get_channels;    % Sinh kênh (output = mảng qd_channel)
Nsc = 672;                % số subcarrier
BW  = 20e6;               % băng thông
for i = 1:length(c)
    fprintf('Channel %d:\n', i);
    H = c(i).fr(BW,Nsc);
    disp(H(:,:,1));  % In ma trận H tại thời điểm đầu
end

%% ================= Vẽ PDP (Power Delay Profile) =================
% Function: Sử dụng linear index (idx) thay vì (ue_id, bs_id)
% Để lấy specific: idx = (bs_id-1)*l.no_rx + ue_id; e.g., UE1-BS1: idx=1
function plot_PDP(c, idx, Nsc, BW)
    % Lấy channel theo linear index
    if idx > numel(c) || idx < 1
        error('Invalid channel index: %d (total %d channels)', idx, numel(c));
    end
    h = c(idx);  

    % Lấy impulse response theo snapshot
    Ht = h.fr(BW, Nsc);            % freq response (Nrx x Ntx x Nsc x Nsnap=1)
    h_imp = ifft(Ht, [], 3);       % chuyển sang miền thời gian (delay domain: Nrx x Ntx x Ndelays x 1)

    % Tính PDP cho từng delay và snapshot (trung bình over rx/tx)
    PDP = squeeze(mean(mean(abs(h_imp).^2, 1), 2));   % [Ndelays x Nsnap] (đúng thứ tự dim)
    
    % Handle empty/zero case
    if isempty(PDP) || all(PDP(:) == 0)
        warning('PDP is empty or zero; nothing to plot for channel %d.', idx);
        return;
    end
    
    % Chuyển sang dB (normalized)
    PDP_dB = 10 * log10(PDP ./ max(PDP(:) + eps));   % tránh log(0)
    
    % Get dimensions
    [num_delays, Nsnap] = size(PDP_dB);  % num_delays ≈ Nsc (delay bins)
    
    % Tạo trục delay và snapshot
    Ts = 1 / BW;                          
    delay_us = (0 : num_delays - 1) * Ts * 1e6;  % µs (row vector)
    time_axis = 1 : Nsnap;                   % snapshot index (row vector)
    % Multiple delays, 1 snapshot: Plot power vs. delay (1D)
    figure;
    plot(delay_us, PDP_dB, 'r-o', 'LineWidth', 1.5);
    xlabel('Delay [\mus]');
    ylabel('Normalized Power [dB]');
    title(sprintf('PDP vs. Delay (Channel %d, Single Snapshot)', idx));
    grid on;
    return;
end

% Lời gọi: PDP cho UE1-BS1 (idx=1)
plot_PDP(c, 1, Nsc, BW);

% Optional: PDP cho UE1-BS2 (idx=51)
% plot_PDP(c, 51, Nsc, BW);

%% ================= Large-Scale Parameters (LSP) =================
% Sử dụng full PDP từ fr(BW, Nsc) để tránh mismatch coeff vs delay
nLinks = numel(c);  % 200
nProcess = nLinks;

LSP_table = struct([]);
u = 0;
skipped_count = 0;

for idx = 1:nProcess
    ch = c(idx); 
    
    % Tính full PDP từ frequency response (robust, size luôn Nsc)
    try
        Ht = ch.fr(BW, Nsc);  % [Nrx x Ntx x Nsc x Nsnap=1]
        h_imp = ifft(Ht, [], 3);  % [Nrx x Ntx x Nsc x 1] (delay domain)
        
        % PDP: Trung bình power per delay bin (over rx/tx)
        PDP = squeeze(mean(mean(abs(h_imp).^2, 1), 2));  % [Nsc x 1] (column)
        pwr_tap = PDP(:).';  % Row [1 x Nsc]
        
        % Delays tương ứng (full grid, seconds)
        delays = (0 : Nsc - 1) / BW;  % [1 x Nsc], unit: s
        
        % Check size (luôn khớp!)
        if length(pwr_tap) ~= length(delays)
            error('Unexpected size mismatch in PDP (should not happen).');
        end
        
    catch ME
        % Nếu lỗi fr/ifft (hiếm), skip
        warning('Channel %d: Error computing PDP (%s). Skipping.', idx, ME.message);
        skipped_count = skipped_count + 1;
        u = u + 1;
        LSP_table(u).P_total_dB = NaN;
        LSP_table(u).rms_delay_s = NaN;
        LSP_table(u).Kfactor_dB = NaN;
        LSP_table(u).channel_idx = idx;
        continue;
    end
    
    P_total = sum(pwr_tap);
    
    % Edge case: Kênh zero
    if P_total <= 0
        u = u + 1;
        LSP_table(u).P_total_dB = NaN;
        LSP_table(u).rms_delay_s = NaN;
        LSP_table(u).Kfactor_dB = NaN;
        LSP_table(u).channel_idx = idx;
        skipped_count = skipped_count + 1;
        continue;
    end
    
    Nbins = length(pwr_tap);  % = Nsc=672
    
    % Mean và RMS delay (trên full PDP, unit: s)
    if Nbins == 1
        mean_delay = delays(1);
        rms_delay = 0;
    else
        mean_delay = sum(pwr_tap .* delays) / P_total;
        rms_delay  = sqrt( sum(pwr_tap .* (delays - mean_delay).^2) / P_total );
    end

    % K-factor trên PDP (power của bin mạnh nhất / power các bin khác)
    [p_max, ~] = max(pwr_tap);
    P_others = P_total - p_max;
    if P_others <= 0 || Nbins == 1
        Kfactor_dB = Inf;
    else
        Kfactor_dB = 10 * log10(p_max / P_others);
    end

    Pr_dB = 10 * log10(P_total + eps);

    u = u + 1;
    LSP_table(u).P_total_dB = Pr_dB;
    LSP_table(u).rms_delay_s = rms_delay;
    LSP_table(u).Kfactor_dB = Kfactor_dB;
    LSP_table(u).channel_idx = idx;
end

% ---- Vẽ histogram (sử dụng bins=20 numeric, an toàn) ----
figure('Name', 'LSP Histograms', 'Position', [100 100 1200 400]);
data_power = [LSP_table.P_total_dB];
data_rms = [LSP_table.rms_delay_s] * 1e9;  % Chuyển sang ns
data_k = [LSP_table.Kfactor_dB];

subplot(1,3,1);
valid_power = data_power(isfinite(data_power));
if ~isempty(valid_power)
    histogram(valid_power, 20);  % 20 bins numeric
    xlabel('Rx Power [dB]'); ylabel('Số links');
    title('Histogram Rx Power');
    grid on;
    xlim([-120, -40]);
else
    axis off;
    text(0.5, 0.5, 'No valid data', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 12);
end

subplot(1,3,2);
valid_rms = data_rms(isfinite(data_rms));
if ~isempty(valid_rms)
    histogram(valid_rms, 20);
    xlabel('RMS Delay [ns]'); ylabel('Số links');
    title('Histogram RMS Delay Spread');
    grid on;
    xlim([0, 2000]);
else
    axis off;
    text(0.5, 0.5, 'No valid data', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 12);
end

subplot(1,3,3);
valid_k = data_k(isfinite(data_k));
if ~isempty(valid_k)
    histogram(valid_k, 20);
    xlabel('K-factor [dB]'); ylabel('Số links');
    title('Histogram K-factor');
    grid on;
    xlim([-10, 20]);
else
    axis off;
    text(0.5, 0.5, 'No valid data', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 12);
end

% In thống kê
fprintf('Số links processed: %d (từ %d total)\n', u, nLinks);
fprintf('Skipped %d channels (errors only).\n', skipped_count);
fprintf('Mean Rx Power: %.2f dB\n', nanmean(data_power));
fprintf('Mean RMS Delay: %.2f ns\n', nanmean(data_rms));
if ~isempty(valid_k)
    fprintf('Mean K-factor: %.2f dB (finite only)\n', nanmean(valid_k));
    fprintf('K-factor range: [%.2f, %.2f] dB\n', min(valid_k), max(valid_k));
else
    fprintf('No finite K-factors.\n');
end

%% ================= Vẽ sơ đồ mạng =================
l.visualize; 
