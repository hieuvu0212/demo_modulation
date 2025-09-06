import numpy as np
import matplotlib.pyplot as plt

# Tạo bit
bits = np.random.randint(0, 2, 1000)

# Ánh xạ BPSK: 0 -> -1, 1 -> +1
symbols = 2*bits - 1  

# Thêm nhiễu và fading
SNR_dB = 5
SNR = 10**(SNR_dB/10)
Es = 1
N0 = Es / SNR

# Nhiễu Gaussian
noise = np.sqrt(N0/2) * (np.random.randn(1000) + 1j*np.random.randn(1000))

# Kênh Rayleigh (complex)
h = (np.random.randn(1000) + 1j*np.random.randn(1000)) / np.sqrt(2)

# Tín hiệu nhận
rx = h*symbols + noise

# Equalization (bù kênh)
rx_eq = rx / h  

# Giải điều chế
bits_hat = (rx_eq.real > 0).astype(int)

# BER
BER = np.sum(bits != bits_hat) / len(bits)
print(f"SNR = {SNR_dB} dB, BER = {BER:.4f}")

# Vẽ kết quả
plt.figure(figsize=(12,5))

# Plot tín hiệu
plt.subplot(1,2,1)
plt.plot(symbols[:50], 'bo-', label="Tx Symbols")
plt.plot(rx_eq[:50].real, 'rx-', label="Rx Symbols (equalized)")
plt.title("Tín hiệu BPSK qua Rayleigh fading channel")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# Chòm sao tín hiệu
plt.subplot(1,2,2)
plt.scatter(rx_eq.real, rx_eq.imag, color="r", alpha=0.5, label="Received")
plt.scatter(symbols, np.zeros_like(symbols), color="b", marker="x", label="Ideal")
plt.title("Chòm sao BPSK sau Rayleigh fading + AWGN")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
