clc; clear; close all;

bit_count = 10000;              % số bit mô phỏng
SNR = 0:1:40;                   % dải SNR từ 0 → 40 dB
BER = zeros(1,length(SNR));     % mảng lưu BER mô phỏng

for aa = 1:length(SNR)
    T_Errors = 0;   % tổng số bit lỗi
    T_bits = 0;     % tổng số bit truyền
    
    while T_Errors < 100    % lặp cho tới khi đủ số lỗi
        % --- Tạo bit ngẫu nhiên ---
        uncoded_bits = round(rand(1, bit_count));
        
        % --- Điều chế BPSK ---
        tx = -2*(uncoded_bits - 0.5);  % 0→-1, 1→+1
        
        % --- Nhiễu AWGN ---
        N0 = 1/10^(SNR(aa)/10);
        
        % --- Kênh fading Rayleigh ---
        h = 1/sqrt(2) * (randn(1,length(tx)) + 1i*randn(1,length(tx)));
        noise = sqrt(N0/2) * (randn(1,length(tx)) + 1i*randn(1,length(tx)));
        
        % --- Tín hiệu nhận (qua kênh + nhiễu) ---
        rx = h .* tx + noise;
        
        % --- Cân bằng kênh ---
        rx = rx ./ h;
        
        % --- Giải điều chế ---
        rx2 = real(rx) < 0;   % quyết định bit (so sánh với 0)
        
        % --- Đếm lỗi ---
        diff = uncoded_bits - rx2;
        T_Errors = T_Errors + sum(abs(diff));
        T_bits   = T_bits + length(uncoded_bits);
    end
    
    % --- Tính BER ---
    BER(aa) = T_Errors / T_bits;
    disp(sprintf('SNR = %2d dB → BER = %.6f', SNR(aa), BER(aa)));
end

% --- BER lý thuyết ---
SNRLin = 10.^(SNR/10);

% Rayleigh lý thuyết
theoryBer = 0.5 .* (1 - sqrt(SNRLin ./ (SNRLin + 1)));

% AWGN lý thuyết
theoryBerAWGN = 0.5 .* erfc(sqrt(SNRLin));

% --- Vẽ ---
BER(BER==0) = 1e-6;
figure;
semilogy(SNR, theoryBer, 'k-', 'LineWidth', 2); hold on;
semilogy(SNR, BER, 'ro', 'LineWidth', 2);
semilogy(SNR, theoryBerAWGN, 'b-', 'LineWidth', 2);

xlabel('SNR (dB)');
ylabel('BER');
title('BER of BPSK in Rayleigh and AWGN Channel');
legend('Rayleigh Theoretical','Rayleigh Simulated','AWGN Theoretical');
axis([0 40 1e-5 1]);
grid on;
