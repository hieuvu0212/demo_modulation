import numpy as np
import matplotlib.pyplot as plt
#tạo bit
bits = np.random.randint(0,2,1000)

#ánh xạ bits QPSK
symbols = []
for i in range(0,len(bits),2):
    b1 = bits[i]
    b2 = bits[i+1]
    if b1 == 1 and b2 == 1 : sym = (1 + 1j)/np.sqrt(2)
    elif b1 ==0 and b2 == 1: sym = (-1 + 1j)/np.sqrt(2)
    elif b1 == 0 and b2 == 0 : sym = (-1 - 1j)/np.sqrt(2)
    else : sym = (1 - 1j)/np.sqrt(2)
    symbols.append(sym)
symbols = np.array(symbols)

#AWGN
SNR_db = 5                       # SNR theo dB
SNR = 10**(SNR_db/10)            # đổi dB sang giá trị tuyến tính
Es = 1                           # năng lượng symbol trung bình
N0 = Es / SNR                    # mật độ phổ công suất nhiễu
noise = np.sqrt(N0/2) * (np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols)))
received = symbols + noise       # tín hiệu sau khi qua kênh

#giai dieu che
bits_rx = []
for r in received:
    b1 = 1 if r.real >= 0 else 0
    b2 = 1 if r.imag >= 0 else 0
    bits_rx.extend([b1, b2])
bits_rx = np.array(bits_rx)

#BER
BER = np.sum(bits != bits_rx) / len(bits)
print(f"SNR = {SNR} dB , BER = {BER:.4f}")

# Plot results
plt.figure(figsize=(10, 5))

# Time-domain representation (first 50 symbols)
plt.subplot(1, 2, 1)
plt.plot(symbols[:50], 'bo-', label="Transmitted symbols (Tx)")
plt.plot(received[:50], 'rx-', label="Received symbols (Rx)")
plt.title("QPSK signal (first 50 bits)")
plt.xlabel("Sample index")
plt.ylabel("Complex value")
plt.legend()
plt.grid(True)

# Constellation diagram
plt.subplot(1, 2, 2)
plt.scatter(received.real, received.imag, color="r", alpha=0.5, label="Received symbols (Rx)")
plt.scatter(symbols.real, symbols.imag, color="b", alpha=0.5, label="Transmitted symbols (Tx)", marker="x")
plt.title("QPSK Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()