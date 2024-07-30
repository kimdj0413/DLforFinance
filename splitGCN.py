import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

print(f'GPU available : {torch.cuda.is_available()}')

## 데이터 로드
train_df = pd.read_csv('C:/webimagecrawling/split/split_1.csv')  # 훈련 데이터 CSV 파일 경로
test_df = pd.read_csv('C:/webimagecrawling/split/split_2.csv')    # 테스트 데이터 CSV 파일 경로

# 노드 특징 행렬 (Node feature matrix)
x_train = torch.tensor(train_df.iloc[:, 1:-1].values, dtype=torch.float)
y_train = torch.tensor(train_df['col45'].values, dtype=torch.long)

x_test = torch.tensor(test_df.iloc[:, 1:-1].values, dtype=torch.float)
y_test = torch.tensor(test_df['col45'].values, dtype=torch.long)

# 엣지 리스트 (Edge list) 생성 (훈련 데이터와 테스트 데이터에서 동일하게 구성)
num_train_rows = len(train_df)
num_test_rows = len(test_df)

# 훈련 데이터 엣지 리스트
train_edge_index = []
for i in range(num_train_rows - 1):
    train_edge_index.append([i, i + 1])
    train_edge_index.append([i + 1, i])

train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()

# 테스트 데이터 엣지 리스트
test_edge_index = []
for i in range(num_test_rows - 1):
    test_edge_index.append([i, i + 1])
    test_edge_index.append([i + 1, i])

test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()

# 데이터셋 생성
train_data = Data(x=x_train, edge_index=train_edge_index, y=y_train)
test_data = Data(x=x_test, edge_index=test_edge_index, y=y_test)

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
        x = F.gelu(x)
        # x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        # x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.gelu(x)
        # x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.gelu(x)
        # x = self.dropout(x)
        
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv6(x, edge_index)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = self.conv7(x, edge_index)

        return F.log_softmax(x, dim=1)

# 모델 초기화
model = GCN(num_node_features=x_train.size(1), num_classes=len(y_train.unique()))

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)

# 옵티마이저 정의
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

# 조기 종료 설정
early_stopping_patience = 100
early_stopping_counter = 0
best_acc = 0.0

# 정확도 계산 함수
def accuracy(pred, labels):
    _, pred_classes = pred.max(dim=1)
    correct = (pred_classes == labels).sum().item()
    return correct / len(labels)

# 모델 훈련
model.train()
for epoch in range(10000):
    optimizer.zero_grad()
    out = model(train_data)
    loss = F.nll_loss(out, train_data.y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 평가 모드로 전환하여 테스트 셋에서의 성능을 평가
    model.eval()
    with torch.no_grad():
        test_out = model(test_data)
        test_loss = F.nll_loss(test_out, test_data.y)
        test_acc = accuracy(test_out, test_data.y)

    # 모델을 다시 훈련 모드로 전환
    model.train()

    if epoch % 10 == 0:
        train_acc = accuracy(out, train_data.y)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}     Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_acc:.4f}    Lr : {current_lr:.6f}')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'gcn.pth')
            early_stopping_counter = 0
            print(f'New best model saved with test accuracy: {best_acc:.4f}')
        else:
            early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 학습 완료 후 최종 평가
model.load_state_dict(torch.load('gcn.pth'))
model.eval()

with torch.no_grad():
    # 테스트 데이터에서 예측 수행
    final_out = model(test_data)
    
    # 테스트 데이터의 손실 계산
    final_loss = F.nll_loss(final_out, test_data.y)
    
    # 정확도 계산
    final_acc = accuracy(final_out, test_data.y)
    
    # 예측 값과 실제 값 추출
    test_pred = final_out.max(dim=1)[1].cpu().numpy()
    test_true = test_data.y.cpu().numpy()

# 혼동 행렬 및 분류 보고서 계산
conf_matrix = confusion_matrix(test_true, test_pred)
class_report = classification_report(test_true, test_pred)

# 결과 출력
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

print(f'Final Eval Loss: {final_loss.item():.4f}, Final Eval Accuracy: {final_acc:.4f}')