import pandas as pd
import torch
from torch_geometric.data import Data

# 데이터 로드
df = pd.read_csv('C:/webimagecrawling/ready.csv')

# 노드 특징 행렬 (Node feature matrix)
x = torch.tensor(df.iloc[:, 1:-1].values, dtype=torch.float)

# 엣지 리스트 (Edge list)
edge_index = []
num_rows = len(df)
for i in range(num_rows - 1):
    edge_index.append([i, i + 1])
    edge_index.append([i + 1, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 레이블
y = torch.tensor(df['Column49'].values, dtype=torch.long)

# 그래프 데이터 객체
graph_data = Data(x=x, edge_index=edge_index, y=y)

import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=8, concat=True)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

##      어텐션 모델
# class GAT(torch.nn.Module):
#     def __init__(self, num_node_features, num_classes):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(num_node_features, 16, heads=8, dropout=0.5)
#         self.bn1 = BatchNorm1d(16 * 8)
#         self.conv2 = GATConv(16 * 8, 32, heads=8, dropout=0.5)
#         self.bn2 = BatchNorm1d(32 * 8)
#         self.conv3 = GATConv(32 * 8, num_classes, heads=1, concat=False, dropout=0.5)
#         self.dropout = torch.nn.Dropout(p=0.5)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.elu(x)
#         x = self.dropout(x)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.elu(x)
#         x = self.dropout(x)

#         x = self.conv3(x, edge_index)

#         return F.log_softmax(x, dim=1)

# 모델 초기화
model = GAT(num_node_features=x.size(1), num_classes=len(y.unique()))

import torch.optim as optim

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = graph_data.to(device)

# 옵티마이저 정의
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 정확도 계산 함수
def accuracy(pred, labels):
    _, pred_classes = pred.max(dim=1)
    correct = (pred_classes == labels).sum().item()
    return correct / len(labels)

# 모델 훈련
best_acc = 0.0
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc = accuracy(out, data.y)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'gat.pth')
            print(f'New best model saved with accuracy: {best_acc:.4f}')

# 학습 완료 후 평가
model.load_state_dict(torch.load('gat.pth'))
model.eval()
with torch.no_grad():
    out = model(data)
    eval_loss = F.nll_loss(out, data.y)
    eval_acc = accuracy(out, data.y)

print(f'Final Eval Loss: {eval_loss.item():.4f}, Final Eval Accuracy: {eval_acc:.4f}')