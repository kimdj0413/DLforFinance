import pandas as pd
import torch
from torch_geometric.data import Data

# 새로운 데이터 프레임 로드
new_df = pd.read_csv('path_to_new_data.csv')

# 새로운 데이터 프레임에서 노드 특징 행렬 (Node feature matrix) 생성
new_x = torch.tensor(new_df.iloc[:, 1:-1].values, dtype=torch.float)

# 새로운 데이터 프레임에서 엣지 리스트 (Edge list) 생성
new_edge_index = []
num_rows = len(new_df)
for i in range(num_rows - 1):
    new_edge_index.append([i, i + 1])
    new_edge_index.append([i + 1, i])

new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()

# 새로운 데이터 프레임에서 레이블 (레이블이 없는 경우 임의로 설정)
new_y = torch.zeros(num_rows, dtype=torch.long)  # 임시 레이블

# 새로운 그래프 데이터 객체 생성
new_graph_data = Data(x=new_x, edge_index=new_edge_index, y=new_y)

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
new_graph_data = new_graph_data.to(device)

# 모델을 평가 모드로 전환
model.eval()

# 새로운 데이터에 대한 예측 수행
with torch.no_grad():
    new_out = model(new_graph_data)
    predicted_labels = new_out.argmax(dim=1)

# 예측 결과 출력
print(predicted_labels)
