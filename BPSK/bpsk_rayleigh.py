import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Tham số
bit_count = 1000000
SNR_dB_range = np.arange(0, 41, 2)  # 0 → 40 dB
BER_sim = []

for SNR_dB in SNR_dB_range:
    # Tạo bit ngẫu nhiên
    bits = np.random.randint(0, 2, bit_count)
    symbols = 2*bits - 1  # BPSK: 0 -> -1, 1 -> +1

    # Nhiễu AWGN + kênh Rayleigh
    SNR = 10**(SNR_dB/10)
    Es = 1
    N0 = Es / SNR
    noise = np.sqrt(N0/2) * (np.random.randn(bit_count) + 1j*np.random.randn(bit_count))
    h = (np.random.randn(bit_count) + 1j*np.random.randn(bit_count)) / np.sqrt(2)

    # Nhận tín hiệu + equalization
    rx = h*symbols + noise
    rx_eq = rx / h

    # Giải điều chế
    bits_hat = (rx_eq.real > 0).astype(int)

    # BER mô phỏng
    BER = np.sum(bits != bits_hat) / bit_count
    BER_sim.append(BER)

# Lý thuyết Rayleigh & AWGN
SNR_lin = 10**(SNR_dB_range/10)
BER_theory_Rayleigh = 0.5*(1 - np.sqrt(SNR_lin/(SNR_lin+1)))
BER_theory_AWGN = 0.5*erfc(np.sqrt(SNR_lin))

# Vẽ
plt.figure(figsize=(12,5))

# (a) BER
plt.subplot(1,2,1)
plt.semilogy(SNR_dB_range, BER_theory_Rayleigh, 'k-', label="Rayleigh Theoretical")
plt.semilogy(SNR_dB_range, BER_sim, 'ro-', label="Rayleigh Simulated")
plt.semilogy(SNR_dB_range, BER_theory_AWGN, 'b-', label="AWGN Theoretical")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER of BPSK in Rayleigh Channel")
plt.legend()
plt.grid(True, which="both")
plt.ylim(1e-5, 1)

# (b) Constellation
plt.subplot(1,2,2)
sample = 200
plt.scatter(rx[:sample].real, rx[:sample].imag, color="r", alpha=0.5, label="Received")
plt.scatter(symbols[:sample].real, np.zeros(sample), color="b", marker="x", label="Ideal")
plt.title("Constellation of BPSK in Rayleigh Fading")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
