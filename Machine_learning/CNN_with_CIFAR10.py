# #Trong huấn luyện mô hình học sâu (deep learning), ta thường có:
# Dữ liệu huấn luyện → loader
# Mô hình → model (ví dụ CNN, ResNet,…)
# Hàm mất mát → loss_func
# Bộ tối ưu → optimizer
# Thiết bị → device (CPU hoặc GPU)
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt

from collections import namedtuple

from sklearn.metrics import classification_report
# Tạo kiểu dữ liệu TrainTest có hai trường: 'train' và 'test'
TrainTest = namedtuple('TrainTest',['train','test'])
def prepare_data():  #Hàm chuẩn bị dữ liệu huấn luyện và kiểm thử
  # Chuyển đổi ảnh từ định dạng PIL.Image sang Tensor để đưa vào mô hình
  transform = transforms.Compose([
      transforms.ToTensor() ])
  # Tải 50,000 ảnh huấn luyện từ CIFAR-10
  trainset = torchvision.datasets.CIFAR10(root='./data',download = True, train = True ,transform= transform )
  # Tải 10,000 ảnh kiểm thử từ CIFAR-10
  testset = torchvision.datasets.CIFAR10(root='./data',download = True, train = False , transform= transform )
  # Gộp hai tập dữ liệu vào cấu trúc TrainTest và trả về
  return TrainTest(trainset,testset)
def prepare_loader(datasets):   # Hàm chuẩn bị DataLoader cho tập huấn luyện và kiểm thử
  # Tạo DataLoader cho tập huấn luyện, chia batch và xáo trộn dữ liệu mỗi epoch
  trainloader = DataLoader( dataset = datasets.train,batch_size=128,shuffle=True,num_workers= 4)
  # Tạo DataLoader cho tập kiểm thử, không xáo trộn dữ liệu
  testloader = DataLoader( dataset = datasets.test,batch_size=128,shuffle=False , num_workers= 4)
  return TrainTest(trainloader,testloader)
def get_classes():   #Hàm lấy nhãn từng ảnh
  # Dán nhãn ảnh
  classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  return classes
class VGG16(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = self._make_features()
    self.classification_head = nn.Linear(in_features = 512 , out_features = 10)  # 512 đầu vào và 10 đầu ra phân lớp . ImageNet có 1000 phân lớp
  def forward(self,x):
    out = self.features(x)    # Có kích thước 128x512x1x1 : 128 → là batch size, tức là trong một lần huấn luyện (hoặc dự đoán), ta xử lý 128 ảnh cùng lúc  , 512 → là số lượng kênh trích xuất một đặc trưng khác nhau của ảnh , 1, 1 → là chiều cao và chiều rộng của feature map
    out = out.view(out.size(0),-1)  # out.size(0) là 128 , -1 là các chiều phía sau gộp lại 512 x 1 x 1 = 512 sau cùng thành kích thước 128 x 512
    out = self.classification_head(out) # giờ chỉ còn 128x10
    return out
  @torch.no_grad
  def _make_features(self):
    config = [64,64,'MP',128,128,'MP', 256 , 256, 256 , 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']
    layers = []
    c_in = 3
    for c in config:
      if c == 'MP':
        layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
      else :
        layers += [nn.Conv2d(in_channels = c_in,out_channels = c,kernel_size = 3, stride = 1 , padding = 1)
                             ,nn.BatchNorm2d(num_features= c)
                             ,nn.ReLU6(inplace= True)]
        c_in = c
    return nn.Sequential(*layers)
  def imshow(images, labels, predicted, target_names): # Hàm này dùng để hiển thị một nhóm ảnh (images) , target_names là classes
    img = torchvision.utils.make_grid(images)          # ghép nhiều ảnh trong 1 batch để dễ quan sát . Mỗi ảnh trong images có dạng [C, H, W] (số kênh, cao, rộng), nên hàm này sẽ xếp chúng thành ma trận
    plt.imshow(img.permute(1,2,0).cpu().numpy())       # Vì trong torch ảnh là [C,H,W] còn plt là [H,W,C] nên img.permute(1,2,0) để đổi thứ  tự để plt in ra . Chuyển dữ liệu và CPU phòng khi tensor nằm trên GPU . chuyển tensor thành mảng NumPy để imshow() có thể vẽ được
    [print(target_names[c], end=' ') for c in list(labels.cpu().numpy()) ]  #Dòng này in ra tên lớp thật (ground truth) của từng ảnh trong batch .
    print()
    [print(target_names[c], end=' ') for c in list(predicted.cpu().numpy()) ]  #In ra tên lớp dự đoán (predicted) của từng ảnh
    print()
  def train_epoch(epoch, model, loader, loss_func, optimizer, device):   # Hàm huấn luyện mô hình trong 1 epoch
    model.train()         # Đặt mô hình ở chế độ huấn luyện
    running_loss = 0      # Biến tích luỹ tính trung bình loss
    reporting_steps = 60  # Sau 60 batch in ra 1 lần loss trung bình
    for i,(images, labels ) in enumerate(loader):            # i là thứ tự batch hiện  tại , images , labels nội dung batch thứ i . enumerate(loader) là lấy ra thứ tự các batch và tự động đếm chỉ số i
      images, labels = images.to(device), labels.to(device)       # Chuyển dữ liệu tới GPU hoặc CPU để tính toán
      outputs = model(images)                                # Truyền các batch ảnh vào mô hình để có đầu ra
      loss = loss_func(outputs, labels)                                         # Tính giá trị hàm loss giữa dự đoán và nhãn thật

      optimizer.zero_grad()                                                     # Cập nhật trọng số gradient = 0
      loss.backward()                                                           # Lan truyền ngược để tính gradient của loss theo các tham số
      optimizer.step()                                                          # Cập nhật trọng số mô hình  theo gradient vừa tính được

      running_loss += loss.item()                                               # Cộng dồn loss của từng batch
      if i % reporting_steps == reporting_steps-1:                              # Mỗi 60 batch in ra loss trung bình và reset lại biến . 0->59 rồi in ra cứ thế tiếp tục
        print(f"Epoch {epoch} step {i} ave_loss {running_loss/reporting_steps:.4f}")
        running_loss = 0.0
  def test_epoch(epoch, model, loader, device):                                 #hàm test epoch : epoch: số thứ tự epoch hiện tại (chỉ để in log, không ảnh hưởng tính toán) , model: mô hình đã huấn luyện ,loader: DataLoader chứa dữ liệu test , 
    ytrue = []                                                                  # cho vào list
    ypred = []                                                                  # cho vào list
    with torch.no_grad():                                                       # tắt tính  gradient Trong quá trình test, ta không cần lan truyền ngược (backpropagation) : giảm bộ nhớ sử dụng , tăng tốc độ , tránh làm thay đổi trọng số mô hình
      model.eval()                                                              # chế độ đánh giá (evaluation mode)M ột số lớp như Dropout, BatchNorm hoạt động khác nhau giữa train và eval

      for i, (images, labels) in enumerate(loader):                             # i là thứ tự batch hiện tại , images,labels nội dung batch thứ i . enumerate(loader) là duyệt qua các batch và tự động đếm chỉ số i
        images, labels = images.to(device), labels.to(device)                   # Chuyển dữ liệu tới GPU hoặc CPU để tính toán
        outputs = model(images)                                                 # Truyền các batch ảnh vào mô hình để có đầu ra
        _, predicted = torch.max(outputs, dim=1)                                #Dòng này chọn nhãn có xác suất cao nhất trong outputs.torch.max(outputs, dim=1) trả về Giá trị lớn nhất , Vị trí (index) của giá trị đó → chính là nhãn dự đoán . (gạch dưới) là để bỏ qua giá trị tối đa, chỉ lấy predicted

        ytrue += list(labels.cpu().numpy())                                     #Chuyển tensor về CPU → numpy → list, rồi nối vào list tổng
        ypred += list(predicted.cpu().numpy())                                  #Mỗi batch sẽ thêm kết quả vào list để cuối cùng có tất cả nhãn thật và nhãn dự đoán của tập test
    return ypred, ytrue
    def main(PATH='./model.pth'):
  classes = get_classes()
  datasets = prepare_data()
  # img, label = datasets.train[0]
  # plt.imshow(img)
  # print(classes[label], img.size)
  # print('train', len(datasets.train), 'test', len(datasets.test))

  loaders = prepare_loader(datasets)
  # images, labels = iter(loaders.train).next()
  # print(images.shape, labels.shape)

  device = torch.device("cuda:0") #Dùng GPU số 0 để train
  model = VGG16().to(device)
  # images, labels = iter(loaders.train).next()
  # outputs = model(images)
  # print(outputs.shape)
  # print(outputs[0])
  # _, predicted = torch.max(outputs, dim=1)
  # print(predicted)
  # imshow(images, labels, predicted, classes)

  loss_func = nn.CrossEntropyLoss()                   # Hàm mất mát
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # Sử dụng Stochastic Gradient Descent, một thuật toán tối ưu cơ bản. lấy tất cả các trọng số (weights) của mạng để optimizer có thể cập nhật.learning rate .giúp quá trình học “mượt hơn”. egularization (phạt trọng số quá lớn), giúp mô hình tránh overfitting
  for epoch in range(10):                                                       # huấn luyện mô hình trong 10 epoch.
    train_epoch(epoch, model, loaders.train, loss_func, optimizer, device)      #hàm train_epoch
    ypred, ytrue = test_epoch(epoch, model, loaders.test, device)               #hàm test_epoch
    print(classification_report(ytrue, ypred, target_names=classes))            #Precision (độ chính xác) Recall (độ bao phủ) F1-score cho từng lớp

    torch.save(model.state_dict(), PATH)                                        #lưu toàn bộ trọng số của model (dưới dạng dict).PATH → đường dẫn file để lưu (ví dụ "cnn_model.pth").

  return model

model = main()
