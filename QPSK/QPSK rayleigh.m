clc; clear; close all;

bit_count = 1e4;                 % số bit mô phỏng trong 1 vòng
SNR_dB = 0:1:40;                 % dải SNR (dB)
BER = zeros(1, length(SNR_dB));  % lưu BER mô phỏng

for aa = 1:length(SNR_dB)
    T_Errors = 0;   % tổng lỗi
    T_bits = 0;     % tổng số bit truyền
    
    while T_Errors < 100          % lặp cho tới khi đủ 100 lỗi
        % --- Sinh bits ngẫu nhiên ---
        bits = randi([0 1], 1, bit_count);
        
        % --- Mapping QPSK ---
        bits_reshape = reshape(bits, 2, []).';
        symbols = (2*bits_reshape(:,1)-1) + 1i*(2*bits_reshape(:,2)-1);
        symbols = symbols.'/sqrt(2);    % hàng vector
        
        % --- Nhiễu AWGN ---
        SNR_lin = 10^(SNR_dB(aa)/10);
        Es = 1;
        N0 = Es/SNR_lin;
        noise = sqrt(N0/2) * (randn(size(symbols)) + 1i*randn(size(symbols)));
        
        % --- Rayleigh fading ---
        h = (randn(size(symbols)) + 1i*randn(size(symbols))) / sqrt(2);
        
        % --- Tín hiệu nhận ---
        rx = h .* symbols + noise;
        
        % --- Cân bằng kênh ---
        rx_eq = rx ./ h;
        
        % --- Giải điều chế ---
        b1_hat = real(rx_eq) >= 0;
        b2_hat = imag(rx_eq) >= 0;
        bits_rx = reshape([b1_hat; b2_hat], 1, []);
        
        % --- Đếm lỗi ---
        T_Errors = T_Errors + sum(bits ~= bits_rx);
        T_bits   = T_bits + length(bits);
    end
    
    % --- BER ---
    BER(aa) = T_Errors / T_bits;
    fprintf('SNR = %2d dB → BER = %.6f\n', SNR_dB(aa), BER(aa));
end

% --- BER lý thuyết ---
SNR_lin = 10.^(SNR_dB/10);
EbN0_lin = SNR_lin/2;   % QPSK: Es = 2*Eb

BER_theory_Rayleigh = 0.5*(1 - sqrt(EbN0_lin./(EbN0_lin+1)));
BER_theory_AWGN     = 0.5*erfc(sqrt(EbN0_lin));

% --- Vẽ ---
BER(BER==0) = 1e-6;                        % tránh log(0)
BER_theory_Rayleigh(BER_theory_Rayleigh==0) = 1e-6;
BER_theory_AWGN(BER_theory_AWGN==0)         = 1e-6;

figure;
semilogy(SNR_dB, BER_theory_Rayleigh, 'k-', 'LineWidth', 2); hold on;
semilogy(SNR_dB, BER, 'ro', 'LineWidth', 2);
semilogy(SNR_dB, BER_theory_AWGN, 'b-', 'LineWidth', 2);

xlabel('SNR (dB)');
ylabel('BER');
title('BER of QPSK in Rayleigh and AWGN Channel');
legend('Rayleigh Theoretical','Rayleigh Simulated','AWGN Theoretical');
axis([0 40 1e-5 1]);
grid on;

print('BER_QPSK.png','-dpng');   % lưu ra file PNG
