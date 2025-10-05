
import numpy as np
# Khởi tạo Fully Connected Layer (FC)
class FC:
    # Khởi tạo lớp FC
    def __init__(self, n_in, n_out, activation):
        self.n_in = n_in                                # Số lượng neuron đầu vào
        self.n_out = n_out                              # Số lượng neuron đầu ra
        self.activation = activation                    # Hàm kích hoạt (activation function)

        # Khởi tạo trọng số và bias
        self.W = np.random.normal(size=(n_in, n_out))   # Ma trận trọng số (n_in, n_out)
        self.b = np.zeros((1, n_out))                   # Vector bias (1, n_out)

        # Khởi tạo gradient cho W và b (phục vụ lan truyền ngược)
        self.dW = np.zeros((n_in, n_out))
        self.db = np.zeros((1, n_out))

    # Hàm sigmoid
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Lan truyền thuận (Forward)
    def forward(self, x):
        self.x = x                                      # Lưu đầu vào (N, n_in)
        self.a = np.dot(self.x, self.W) + self.b        # Tính đầu ra tuyến tính a = xW + b

        # Áp dụng hàm kích hoạt (nếu có)
        if self.activation is None:
            self.f = self.a                             # Nếu không có activation → hàm tuyến tính
        elif self.activation == "sigmoid":
            self.f = self.sigmoid(self.a)               # Nếu có → áp dụng sigmoid
        else:
            raise NotImplementedError(f"Activation {self.activation} chưa hỗ trợ.")
        return self.f

    # Lan truyền ngược (Backward)
    def backward(self, df):
        # df: gradient của hàm mất mát theo đầu ra f

        # Tính đạo hàm theo a (da)
        if self.activation is None:
            da = df                                     # Nếu tuyến tính: da = df
        elif self.activation == "sigmoid":
            da = self.f * (1 - self.f) * df             # Nếu sigmoid: da = f'(a) * df
        else:
            raise NotImplementedError(f"Not implemented {self.activation}")

        # Tính gradient cho trọng số, bias và đầu vào
        self.dW = np.einsum('ij,ik->jk', self.x, da)    # dW = x^T * da
        self.db = np.sum(da, axis=0, keepdims=True)     # db = tổng(da)
        self.dx = np.dot(da, self.W.T)                  # dx = da * W^T

        # Lưu lại để debug hoặc kiểm thử
        self.df = df
        self.da = da

        return self.dx

    # Trả về các tham số (weights và bias)
    def parameters(self):
        return (self.W, self.b)

    # Trả về các gradient tương ứng
    def grads(self):
        return (self.dW, self.db)
# Multi-layer perceptron Model
class MLP :
    def __init__(self, n_input, hiddens):
        self.n_input = n_input  # Lưu số lượng neuron đầu vào của mạng (input layer)
        self.hiddens = hiddens  # Lưu danh sách số neuron của từng hidden layer
        self.layers = [  # Danh sách chứa các lớp FC được khởi tạo tự động
            # Dùng hàm sigmoid cho các hidden layer (phi tuyến)
            # Dùng hàm tuyến tính (linear) cho layer cuối cùng (output)
            FC(
                n_in=hiddens[i - 1] if i > 0 else n_input,
                # Nếu i = 0 thì đầu vào là lớp input, ngược lại là đầu ra của lớp trước
                n_out=hiddens[i],  # Số neuron của lớp hiện tại (đầu ra)
                activation="sigmoid" if i < len(hiddens) - 1 else None
                # Các lớp ẩn dùng sigmoid, lớp cuối không dùng activation
            )
            for i in range(len(hiddens))  # Lặp qua toàn bộ các lớp được định nghĩa trong danh sách hiddens
        ]

    def forward(self, x):
        out = x                         # Khởi tạo đầu vào ban đầu (input của mạng)
        for layer in self.layers:       # Duyệt qua từng lớp FC trong danh sách layers
            out = layer.forward(out)    # Đầu ra của lớp trước trở thành đầu vào của lớp sau
        return out                      # Trả về đầu ra cuối cùng của toàn bộ mạng

    def backward(self, dout):
        # Lan truyền ngược (Backpropagation) qua toàn bộ mạng
        for layer in self.layers[::-1]:     # Duyệt qua các lớp FC từ cuối về đầu
            din = layer.backward(dout)      # Gradient đầu vào của lớp hiện tại = gradient đầu ra của lớp trước
            dout = din                      # Cập nhật để truyền tiếp ngược lại

        dx = din                            # Gradient cuối cùng (so với input ban đầu)
        return dx                           # Trả về đạo hàm theo đầu vào (∂L/∂x)

    def parameters(self):
        # Trả về danh sách tất cả các tham số (W, b) của từng lớp FC trong mạng
        # Sử dụng sum(..., []) để nối (flatten) danh sách các tuple từ mỗi lớp
        return [p for layer in self.layers for p in layer.parameters()]

    def grads(self):
        # Trả về danh sách tất cả các gradient (dW, db) tương ứng với từng lớp FC
        # Dùng cùng cấu trúc như parameters() để đảm bảo thứ tự khớp nhau
        return [g for layer in self.layers for g in layer.grads()]

# Cross Entropy loss
class CrossEntropyLoss:
    @staticmethod
    def softmax(x):
        # Trừ đi giá trị lớn nhất trên mỗi hàng để tránh tràn số (overflow)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))

        # Chuẩn hóa theo công thức softmax:
        # p_i = e^(a_i) / Σ e^(a_j)
        # (ở đây ta dùng e^(a_i - max(a)) để ổn định số học)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, ypred, ytrue):
        n, n_out = ypred.shape   # Kích thước đầu ra: (số mẫu, số lớp)
        self.ypred = ypred  # N x n_out (logit value - giá trị đầu ra trước softmax)
        self.ytrue = ytrue  # N (nhãn thật, dạng chỉ số lớp: 0 → n_out-1)
        self.mu = self.softmax(ypred)  # mu = softmax(ypred) → Xác suất dự đoán cho từng lớp

        # Tính hàm mất mát cross-entropy:
        # L = -Σ log(mu_true)
        # Trong đó mu_true là xác suất dự đoán của lớp đúng (từ ytrue)
        # Nếu y ở dạng one-hot, công thức gốc là L = -Σ y * log(mu)
        # nhưng ở đây ytrue là chỉ số lớp nên ta dùng mu[range(n), ytrue] để lấy đúng xác suất đó.
        loss = np.sum(-np.log(self.mu[range(n), ytrue]))
        return loss

    def backward(self):
        n, n_out = self.ypred.shape  # n: batch size, n_out: số lớp đầu ra
        d_ypred = self.mu.copy()  # Khởi tạo đạo hàm theo softmax output: dL/da = mu - y
        d_ypred[range(n), self.ytrue] -= 1  # Trừ 1 tại lớp đúng → tương đương mu - y (vì y là one-hot)
        return d_ypred  # Trả về gradient dL/da

#Stochastic Gradient Descent ( thuật toán xuống đồi )
class SGDOptimizer:
    def __init__(self,model,learning_rate):
        self.model = model    #model : mô hình cần cập nhật tham số (các layer có W, b, ...)
        self.learning_rate = learning_rate   #learning_rate : tốc độ học (η) - hệ số điều chỉnh bước cập nhật
    def parameters(self):
        return self.model.parameters()    #Trả về danh sách tất cả tham số của mô hình (W, b, ...)
    def grads(self):
        return self.model.grads()        # Trả về danh sách tất cả gradient tương ứng (dW, db, ...)
    def zero_grad(self):
        for g in self.grads():
            g.fill(0)                    #Đặt lại tất cả gradient về 0 trước khi tính gradient mới
    def step(self):
        #input is the derivative of loss function on the output of the model
        #dW has been computed by backward function
        #perform a gradient step W = W - lamda dW
        #Với mỗi cặp (tham số p, gradient g):
        #    p = p - learning_rate * g
        #   Nếu g > 0 → giảm p
        #   Nếu g < 0 → tăng p
        #Cả trọng số W và bias b đều được cập nhật vì chúng
        for p,g in zip(self.model.parameters(), self.grads()):
            p -= self.learning_rate * g

