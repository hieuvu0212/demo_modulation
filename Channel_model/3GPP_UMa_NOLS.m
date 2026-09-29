%% =====================================================================
%  Mô phỏng kênh 3GPP TR 38.901 UMa NLOS bằng QuaDRiGa
%  - 4 BS (UPA 4x4, λ/2), 50 UE (UPA 2x2, λ/2), fc = 3.5 GHz
%  - Xuất: ma trận kênh H, PDP, LSP (path gain, RMS delay spread, K-factor),
%          path gain vs khoảng cách (so với công thức 38.901), sơ đồ mạng
% ======================================================================

%% ================= Khởi tạo môi trường =================
clear; close all; clc;

quadriga_path = 'E:\QuaDRiGa-main\quadriga_src';
if ~exist('qd_layout', 'class')
    addpath(genpath(quadriga_path));
end
rng(1);                                   % cố định seed để tái lập kết quả

%% ================= Tham số mô phỏng =================
fc      = 3.5e9;                          % tần số trung tâm 3.5 GHz (band n78)
lambda  = 299792458 / fc;                 % bước sóng [m]
BW      = 20e6;                           % băng thông
Nsc     = 672;                            % số subcarrier
no_bs   = 4;                              % số BS
no_ue   = 50;                             % số UE
h_bs    = 25;                             % chiều cao BS [m]
h_ue    = 1.5;                            % chiều cao UE [m]
area    = 250;                            % UE rải trong [-area, area]^2
d2d_min = 35;                             % khoảng cách 2D tối thiểu BS-UE (38.901 UMa)

s = qd_simulation_parameters;
s.center_frequency = fc;
s.show_progress_bars = 1;

%% ================= Tạo layout =================
l = qd_layout(s);

%% ================= Antenna Array cho BS: UPA 4x4 (mặt phẳng y-z) =================
tx_ant = qd_arrayant('omni');
tx_ant.element_position = upa_positions(4, 4, 0.5 * lambda);   % [3 x 16], đơn vị mét
l.tx_array = tx_ant;

%% ================= Antenna Array cho UE: UPA 2x2 =================
rx_ant = qd_arrayant('omni');
rx_ant.element_position = upa_positions(2, 2, 0.5 * lambda);   % [3 x 4]
l.rx_array = rx_ant;

%% ================= Cấu hình BS =================
l.no_tx = no_bs;
l.tx_position = [150  150 -150 -150;      % x
                 150 -150  150 -150;      % y
                 h_bs h_bs h_bs h_bs];    % z

%% ================= Cấu hình UE (đảm bảo d2D >= d2d_min tới mọi BS) =================
ue_pos = zeros(3, no_ue);
n = 0;
while n < no_ue
    p = (rand(2,1) * 2 - 1) * area;
    if min(vecnorm(l.tx_position(1:2,:) - p, 2, 1)) >= d2d_min
        n = n + 1;
        ue_pos(:, n) = [p; h_ue];
    end
end
l.no_rx = no_ue;
l.rx_position = ue_pos;

%% ================= Kịch bản & sinh kênh =================
l.set_scenario('3GPP_38.901_UMa_NLOS');   % tất cả link là UMa NLOS
l.set_pairing;                            % mọi cặp BS–UE
c = l.get_channels;                       % mảng qd_channel [no_ue x no_bs]
nLinks = numel(c);
fprintf('\nĐã sinh %d kênh (%d UE x %d BS), mỗi kênh %d Rx x %d Tx anten.\n', ...
    nLinks, no_ue, no_bs, c(1).no_rxant, c(1).no_txant);

% In ma trận H (subcarrier đầu) cho vài kênh đầu tiên
n_print = 2;
for i = 1:min(n_print, nLinks)
    H = c(i).fr(BW, Nsc);                 % [Nrx x Ntx x Nsc]
    fprintf('\n%s: H(:,:,1) =\n', c(i).name);
    disp(H(:,:,1));
end

%% ================= PDP =================
% Linear index: idx = (bs_id-1)*no_ue + ue_id  (vd. UE1-BS1: idx = 1)
plot_PDP(c, 1, Nsc, BW);
% plot_PDP(c, no_ue + 1, Nsc, BW);        % UE1-BS2

%% ================= Large-Scale Parameters (LSP) =================
% Tính trực tiếp từ các path (coeff/delay) của QuaDRiGa -> chính xác, không
% bị giới hạn độ phân giải 1/BW như khi IFFT từ đáp ứng tần số.
PG_dB  = nan(no_ue, no_bs);               % path gain trung bình theo anten [dB]
DS_ns  = nan(no_ue, no_bs);               % RMS delay spread [ns]
K_dB   = nan(no_ue, no_bs);               % tỉ số công suất path mạnh nhất / còn lại [dB]
d3D    = nan(no_ue, no_bs);               % khoảng cách 3D [m]

for ue = 1:no_ue
    for bs = 1:no_bs
        [PG_dB(ue,bs), DS_ns(ue,bs), K_dB(ue,bs)] = link_lsp(c(ue,bs));
        d3D(ue,bs) = norm(l.rx_position(:,ue) - l.tx_position(:,bs));
    end
end

% BS phục vụ = BS có path gain lớn nhất
[PG_serv_dB, serving_bs] = max(PG_dB, [], 2);

% ---- Histogram LSP ----
figure('Name', 'LSP Histograms', 'Position', [100 100 1200 400]);
subplot(1,3,1); plot_hist(PG_dB(:), 'Path gain [dB]', 'Histogram path gain');
subplot(1,3,2); plot_hist(DS_ns(:), 'RMS delay spread [ns]', 'Histogram RMS DS');
subplot(1,3,3); plot_hist(K_dB(:),  'K-factor [dB]', 'Histogram K-factor');

% ---- Path gain vs khoảng cách, so với 38.901 UMa NLOS ----
d_ax  = logspace(log10(d2d_min), log10(max(d3D(:))), 100);
PL_nl = pl_38901_uma_nlos(d_ax, fc, h_bs, h_ue);
figure('Name', 'Path gain vs distance');
semilogx(d3D(:), PG_dB(:), 'b.', 'MarkerSize', 10); hold on;
semilogx(d_ax, -PL_nl, 'r-', 'LineWidth', 1.5);
semilogx(d_ax, -PL_nl + 6, 'r--', d_ax, -PL_nl - 6, 'r--');   % ±σ_SF (6 dB)
xlabel('Khoảng cách 3D [m]'); ylabel('Path gain [dB]');
legend('QuaDRiGa', '38.901 UMa NLOS', '\pm\sigma_{SF}', 'Location', 'southwest');
title('Path gain vs khoảng cách'); grid on;

% ---- In thống kê ----
fprintf('\n================ Thống kê LSP (%d links) ================\n', nLinks);
fprintf('Path gain      : mean %.2f dB, range [%.2f, %.2f] dB\n', ...
    mean(PG_dB(:),'omitnan'), min(PG_dB(:)), max(PG_dB(:)));
fprintf('Serving PG     : mean %.2f dB\n', mean(PG_serv_dB,'omitnan'));
fprintf('RMS delay      : mean %.2f ns, median %.2f ns\n', ...
    mean(DS_ns(:),'omitnan'), median(DS_ns(:),'omitnan'));
fprintf('  (38.901 UMa NLOS @%.1f GHz: median DS ≈ %.0f ns)\n', fc/1e9, ...
    1e9 * 10^(-6.28 - 0.204*log10(fc/1e9)));
fprintf('K-factor       : mean %.2f dB, range [%.2f, %.2f] dB\n', ...
    mean(K_dB(:),'omitnan'), min(K_dB(:)), max(K_dB(:)));
fprintf('Số UE mỗi BS phục vụ: %s\n', mat2str(histcounts(serving_bs, 0.5:1:no_bs+0.5)));

%% ================= Vẽ sơ đồ mạng =================
l.visualize([], [], 0);
title('Layout UMa NLOS');

%% =====================================================================
%  Local functions (phải nằm cuối file script)
% ======================================================================
function pos = upa_positions(Nv, Nh, d)
% Toạ độ phần tử UPA Nv x Nh trên mặt phẳng y-z, spacing d [m], tâm ở gốc.
    [y, z] = meshgrid(((0:Nh-1) - (Nh-1)/2) * d, ((0:Nv-1) - (Nv-1)/2) * d);
    pos = [zeros(1, Nv*Nh); y(:).'; z(:).'];
end

function [P, tau] = path_powers(ch)
% Công suất (trung bình theo anten) và delay [s] của từng path, snapshot 1.
    coeff = ch.coeff(:,:,:,1);                             % [Nrx x Ntx x L]
    P = squeeze(mean(mean(abs(coeff).^2, 1), 2));          % [L x 1]
    if ndims(ch.delay) >= 3                                 % delay riêng từng anten
        tau = squeeze(mean(mean(ch.delay(:,:,:,1), 1), 2));
    else
        tau = ch.delay(:, 1);                               % [L x 1]
    end
    P = P(:); tau = tau(:);
end

function [PG_dB, DS_ns, K_dB] = link_lsp(ch)
% LSP của một link từ các path, công suất lấy trung bình theo anten.
    [P, tau] = path_powers(ch);
    Ptot = sum(P);
    if Ptot <= 0
        PG_dB = NaN; DS_ns = NaN; K_dB = NaN; return;
    end
    PG_dB = 10*log10(Ptot);
    tau_m = sum(P .* tau) / Ptot;
    DS_ns = 1e9 * sqrt(sum(P .* (tau - tau_m).^2) / Ptot);
    Pmax  = max(P);
    K_dB  = 10*log10(Pmax / max(Ptot - Pmax, eps));
end

function PL = pl_38901_uma_nlos(d3D, fc, h_bs, h_ut)
% Path loss 3GPP TR 38.901 Table 7.4.1-1, UMa NLOS (không kể shadow fading).
    fc_GHz = fc / 1e9;
    d2D  = sqrt(max(d3D.^2 - (h_bs - h_ut)^2, 1));
    dBP  = 4 * (h_bs - 1) * (h_ut - 1) * fc / 299792458;   % h_E = 1 m
    PL1  = 28 + 22*log10(d3D) + 20*log10(fc_GHz);
    PL2  = 28 + 40*log10(d3D) + 20*log10(fc_GHz) ...
           - 9*log10(dBP^2 + (h_bs - h_ut)^2);
    PL_LOS = PL1 .* (d2D <= dBP) + PL2 .* (d2D > dBP);
    PL_NL  = 13.54 + 39.08*log10(d3D) + 20*log10(fc_GHz) - 0.6*(h_ut - 1.5);
    PL = max(PL_LOS, PL_NL);
end

function plot_PDP(c, idx, Nsc, BW)
% PDP (trung bình theo anten) từ đáp ứng tần số, idx là linear index của c.
    if idx > numel(c) || idx < 1
        error('Invalid channel index: %d (total %d channels)', idx, numel(c));
    end
    Ht    = c(idx).fr(BW, Nsc);                        % [Nrx x Ntx x Nsc]
    h_imp = ifft(Ht, [], 3);                           % miền delay, bin = 1/BW
    PDP   = squeeze(mean(mean(abs(h_imp).^2, 1), 2));  % [Nsc x 1]
    if all(PDP == 0)
        warning('PDP is zero for channel %d.', idx); return;
    end
    PDP_dB   = 10*log10(PDP / max(PDP) + eps);
    delay_us = (0:Nsc-1).' / BW * 1e6;

    % Nửa sau trục delay là rò rỉ vòng của IFFT (delay "âm") -> bỏ qua.
    % Chỉ vẽ vùng có năng lượng (> -30 dB), tối thiểu 2 µs.
    half  = floor(Nsc/2);
    last  = find(PDP_dB(1:half) > -30, 1, 'last');
    nshow = min(half, max(last + 10, round(2e-6 * BW)));

    % Các path thật (trung bình theo anten) để so sánh
    [P, tau] = path_powers(c(idx));
    P_dB = 10*log10(P / max(P));

    figure('Name', sprintf('PDP %s', c(idx).name));
    plot(delay_us(1:nshow), PDP_dB(1:nshow), 'b-', 'LineWidth', 1.2); hold on;
    stem(tau*1e6, P_dB, 'r', 'filled', 'BaseValue', -40, 'MarkerSize', 4);
    ylim([-40 1]); xlim([0 delay_us(nshow)]);
    legend(sprintf('IFFT của H (%d SC, %.0f MHz)', Nsc, BW/1e6), 'Path (QuaDRiGa)');
    xlabel('Delay [\mus]'); ylabel('Normalized power [dB]');
    title(sprintf('PDP - %s', strrep(c(idx).name, '_', '\_')));
    grid on;
end

function plot_hist(x, xlab, ttl)
    x = x(isfinite(x));
    if isempty(x)
        axis off;
        text(0.5, 0.5, 'No valid data', 'HorizontalAlignment', 'center', ...
            'Units', 'normalized', 'FontSize', 12);
        return;
    end
    histogram(x, 20);
    xlabel(xlab); ylabel('Số links'); title(ttl); grid on;
end
