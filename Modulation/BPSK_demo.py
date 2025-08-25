import numpy as np
import matplotlib.pyplot as plt

# 1. Sinh dữ liệu nhị phân
N = 1000  # số bit
bits = np.random.randint(0, 2, N)

# 2. Ánh xạ BPSK (0 -> -1, 1 -> +1)
symbols = 2*bits - 1

# 3. Thêm nhiễu AWGN theo SNR
SNR_dB = 5  # SNR (dB)
SNR = 10**(SNR_dB/10)  # đổi sang tỉ số
Es = 1  # năng lượng symbol
N0 = Es / SNR
noise = np.sqrt(N0/2) * np.random.randn(N)  # AWGN
rx = symbols + noise  # tín hiệu thu

# 4. Giải điều chế
bits_hat = (rx > 0).astype(int)

# 5. Tính BER
BER = np.sum(bits != bits_hat) / N
print(f"SNR = {SNR_dB} dB, BER = {BER:.4f}")

# 6. Vẽ kết quả
plt.figure(figsize=(12,5))

# (a) Vẽ một đoạn tín hiệu
plt.subplot(1,2,1)
plt.plot(symbols[:50], 'bo-', label="Tx symbols")
plt.plot(rx[:50], 'rx-', label="Rx symbols (with noise)")
plt.title("Tín hiệu BPSK (50 bit đầu)")
plt.xlabel("Index")
plt.ylabel("Giá trị")
plt.legend()
plt.grid(True)

# (b) Vẽ chòm sao
plt.subplot(1,2,2)
plt.scatter(rx, np.zeros_like(rx), color="r", alpha=0.5, label="Received")
plt.scatter(symbols, np.zeros_like(symbols), color="b", marker="x", label="Ideal")
plt.title("Sơ đồ chòm sao BPSK")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
