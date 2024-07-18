import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d

# 새로운 데이터 프레임 로드
df = pd.read_csv('path_to_data.csv')

# 새로운 데이터 프레임에서 노드 특징 행렬 (Node feature matrix) 생성
x = torch.tensor(df.iloc[:, 1:-1].values, dtype=torch.float)

# 새로운 데이터 프레임에서 엣지 리스트 (Edge list) 생성
edge_index = []
num_rows = len(df)
for i in range(num_rows - 1):
    edge_index.append([i, i + 1])
    edge_index.append([i + 1, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 새로운 데이터 프레임에서 레이블 (레이블이 없는 경우 임의로 설정)
y = torch.tensor(df['label'].values, dtype=torch.long)  # 임시 레이블

# 새로운 그래프 데이터 객체 생성
graph_data = Data(x=x, edge_index=edge_index, y=y)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.bn1 = BatchNorm1d(16)
        self.conv2 = GCNConv(16, 32)
        self.bn2 = BatchNorm1d(32)
        self.conv3 = GCNConv(32, 64)
        self.bn3 = BatchNorm1d(64)
        self.conv4 = GCNConv(64, 128)
        self.bn4 = BatchNorm1d(128)
        self.conv5 = GCNConv(128, 256)
        self.bn5 = BatchNorm1d(256)
        self.conv6 = GCNConv(256, 512)
        self.bn6 = BatchNorm1d(512)
        self.conv7 = GCNConv(512, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv6(x, edge_index)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv7(x, edge_index)

        return F.log_softmax(x, dim=1)
    
# 장치 설정
model = GCN(num_node_features=x.size(1), num_classes=len(y.unique()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
graph_data = graph_data.to(device)

# 모델을 평가 모드로 전환
model.load_state_dict(torch.load('gcn.pth'))
model.eval()

# 새로운 데이터에 대한 예측 수행
with torch.no_grad():
    out = model(graph_data)
    predicted_labels = out.argmax(dim=1)

# 예측 결과 출력
print(predicted_labels)
