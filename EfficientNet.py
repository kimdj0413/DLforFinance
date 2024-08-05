import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
from tqdm import tqdm
import numpy as np

# 1. 데이터 준비
data = pd.read_csv('StockVector.csv')

X = data.iloc[:, :44].values
y = data.iloc[:, 44].values

# 2. 그래프 데이터 생성
num_nodes = X.shape[0]
num_features = X.shape[1]

# 노드 특성
x = torch.tensor(X, dtype=torch.float)

# 엣지 리스트 (예: 선형 그래프)
edge_index = []
num_rows = len(data)
for i in tqdm(range(num_rows - 1)):
    edge_index.append([i, i + 1])
    edge_index.append([i + 1, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 라벨
y = torch.tensor(y, dtype=torch.float).view(-1, 1)

# PyTorch Geometric 데이터 객체 생성
graph_data = Data(x=x, edge_index=edge_index, y=y)

# 3. GNN 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 128)  # 첫 번째 레이어
        self.conv2 = GCNConv(128, 64)             # 두 번째 레이어
        self.conv3 = GCNConv(64, 32)              # 세 번째 레이어
        self.fc1 = torch.nn.Linear(32, 16)        # 첫 번째 완전 연결층
        self.fc2 = torch.nn.Linear(16, 1)         # 두 번째 완전 연결층

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# 4. 모델 훈련
model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

# 데이터 로더
loader = DataLoader([graph_data], batch_size=1)

# 추가: 새로운 CSV 파일 로드
new_data = pd.read_csv('StockVectorVal.csv')
X_new = new_data.iloc[:, :44].values
y_new = new_data.iloc[:, 44].values
y_new_tensor = torch.tensor(y_new, dtype=torch.float).view(-1, 1)

# 새로운 그래프 데이터 생성
x_new = torch.tensor(X_new, dtype=torch.float)
edge_index_new = []
num_rows_new = len(new_data)
for i in range(num_rows_new - 1):
    edge_index_new.append([i, i + 1])
    edge_index_new.append([i + 1, i])

edge_index_new = torch.tensor(edge_index_new, dtype=torch.long).t().contiguous()
graph_data_new = Data(x=x_new, edge_index=edge_index_new, y=y_new_tensor)

for epoch in range(10000):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    # 손실 출력
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 새로운 데이터셋에 대한 정확도 계산
    model.eval()
    with torch.no_grad():
        pred_new = model(graph_data_new)
        pred_label_new = (pred_new > 0.5).float()
        accuracy = (pred_label_new.eq(y_new_tensor).sum().item() / y_new_tensor.size(0)) * 100
        print(f'Accuracy on new data: {accuracy:.2f}%')

# 5. 모델 평가
model.eval()
with torch.no_grad():
    pred = model(graph_data)
    pred_label = (pred > 0.5).float()
    print(f'Predicted Labels: {pred_label}')

"""
edges = []

# tqdm을 사용하여 진행 상황 표시
for i in tqdm(range(num_nodes), desc="Creating edges"):
    for j in range(num_nodes):
        if i != j:
            edges.append([i, j])  # 모든 노드 간의 엣지 생성

# NumPy 배열로 변환
edge_index = np.array(edges)

# Torch Tensor로 변환
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
"""