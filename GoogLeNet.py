import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import torch.nn.functional as F

# 장치 설정 (GPU 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV 파일 로드
data = pd.read_csv('StockVector.csv')
data = data.iloc[:1000]

# 데이터 전처리
X = data.iloc[:, :44].values  # 앞 44개 float
y = data.iloc[:, 44].values    # 마지막 컬럼 (라벨)

# 데이터를 Tensor로 변환 및 이미지 형태로 변경
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1, 44, 1)  # (N, C, H, W) 형태로 변환
X_tensor = X_tensor.repeat(1, 3, 1, 1)  # 3채널로 복사

# 크기 조정: 44x1 이미지를 224x224로 변환
X_tensor = F.interpolate(X_tensor, size=(224, 224), mode='bilinear', align_corners=False)

y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)  # (N,) 형태로 변환

# 데이터셋과 데이터로더 생성
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# GoogLeNet 모델 정의
class CustomGoogLeNet(nn.Module):
    def __init__(self):
        super(CustomGoogLeNet, self).__init__()
        self.googlenet = models.googlenet(weights='DEFAULT')  # 최신 가중치 사용
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, 1)  # 출력층을 1로 설정 (이진 분류)

    def forward(self, x):
        return self.googlenet(x)

# 모델, 손실 함수 및 최적화 도구 설정
model = CustomGoogLeNet().to(device)  # 모델을 GPU로 이동
criterion = nn.BCEWithLogitsLoss()  # 이진 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 모델 학습
num_epochs = 100  # 원하는 에폭 수 설정
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 입력과 라벨을 GPU로 이동
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # 출력 크기를 맞추기 위해 squeeze() 사용
        loss = criterion(outputs, labels)  # 크기 맞추기
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 모델 저장
torch.save(model.state_dict(), 'googlenet_model.pth')

print("모델 학습 완료 및 저장됨.")
