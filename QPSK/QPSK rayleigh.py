import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Tham số
bits_count = 2000000
SNR_dB_range = np.arange(0, 41, 2)  # 0 → 40 dB
BER_sim = []

for SNR_dB in SNR_dB_range:
    # Tạo bit ngẫu nhiên
    bits = np.random.randint(0, 2, bits_count)

    # Ánh xạ bits -> QPSK symbols
    symbols = []
    for i in range(0, len(bits), 2):
        b1, b2 = bits[i], bits[i+1]
        if b1 == 1 and b2 == 1: sym = (1 + 1j) / np.sqrt(2)
        elif b1 == 0 and b2 == 1: sym = (-1 + 1j) / np.sqrt(2)
        elif b1 == 0 and b2 == 0: sym = (-1 - 1j) / np.sqrt(2)
        else: sym = (1 - 1j) / np.sqrt(2)
        symbols.append(sym)
    symbols = np.array(symbols)

    # Nhiễu AWGN + kênh Rayleigh
    SNR = 10**(SNR_dB/10)          # đổi dB sang tuyến tính
    Es = 1                         # năng lượng symbol
    N0 = Es / SNR
    noise = np.sqrt(N0/2) * (np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols)))
    h = (np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols))) / np.sqrt(2)

    # Nhận tín hiệu + equalization
    rx = h*symbols + noise
    rx_eq = rx / h

    # Giải điều chế
    bits_rx = []
    for r in rx_eq:
        b1 = 1 if r.real >= 0 else 0
        b2 = 1 if r.imag >= 0 else 0
        bits_rx.extend([b1, b2])
    bits_rx = np.array(bits_rx)

    # BER
    BER = np.sum(bits != bits_rx) / bits_count
    BER_sim.append(BER)

# Lý thuyết Rayleigh & AWGN
SNR_lin = 10**(SNR_dB_range/10)
EbN0_lin = SNR_lin / 2
BER_theory_Rayleigh = 0.5 * (1 - np.sqrt(EbN0_lin / (EbN0_lin + 1)))
BER_theory_AWGN = 0.5*erfc(np.sqrt(EbN0_lin))   # QPSK lý thuyết trong AWGN

# Plot results
plt.figure(figsize=(12, 5))

# (a) BER
plt.subplot(1, 2, 1)
plt.semilogy(SNR_dB_range, BER_theory_Rayleigh, 'k-', label="Rayleigh Theoretical")
plt.semilogy(SNR_dB_range, BER_sim, 'ro-', label="Rayleigh Simulated")
plt.semilogy(SNR_dB_range, BER_theory_AWGN, 'b-', label="AWGN Theoretical")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER of QPSK in Rayleigh Channel")
plt.legend()
plt.grid(True, which="both")
plt.ylim(1e-5, 1)

# (b) Constellation
plt.subplot(1, 2, 2)
sample = 500
plt.scatter(rx_eq[:sample].real, rx_eq[:sample].imag, color="r", alpha=0.5, label="Received")
plt.scatter(symbols[:sample].real, symbols[:sample].imag, color="b", marker="x", label="Ideal")
plt.title("Constellation of QPSK in Rayleigh Fading")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
