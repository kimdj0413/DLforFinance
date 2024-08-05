"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# CSV 파일 읽기
df = pd.read_csv('StockVector.csv')
df = df.iloc[:, :44]

# 이미지 수 설정
images_per_page = 1  # 한 페이지에 1개의 이미지
total_images = len(df)
print(total_images)
# 모든 이미지에 대해 반복
for overall_index in tqdm(range(total_images)):
    plt.figure(figsize=(10, 10))  # 전체 그림 크기 설정
    
    row = df.iloc[overall_index].values
    
    # 데이터의 범위를 정규화 (0과 1 사이로)
    normalized_row = (row - np.min(row)) / (np.max(row) - np.min(row))
    
    # x 좌표 생성
    x = np.linspace(0, 2 * np.pi, len(normalized_row))  # 데이터 길이에 맞춰 x 좌표 생성
    
    # 복잡한 패턴 생성 (사인파와 코사인파 조합)
    y = normalized_row * np.sin(x * 2) + normalized_row * np.cos(x * 3)

    plt.plot(x, y, color='black')
    plt.axis('off')  # 축 및 제목 제거

    plt.savefig(f'D:\\Images\\image_{overall_index + 1}.jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # 현재 플롯 닫기"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CSV 파일에서 데이터 프레임 생성
df = pd.read_csv('StockVector.csv')

# 각 행을 3채널 이미지로 변환 (RGB)
num_rows = df.shape[0]
images = df.values[:, :44].reshape(num_rows, 11, 4)

# 이미지 한 장씩 시각화
for i in range(num_rows):
    plt.imshow(images[i])  # 각 이미지를 시각화
    plt.title(f"Image {i+1}: {images[i].shape}")  # 이미지 번호 및 shape 표시
    plt.axis('off')  # 축 숨기기
    plt.show()  # 이미지 보여주기
    
    # 이미지를 저장하기 위해 PIL을 사용하여 변환
    img_to_save = Image.fromarray((images[i] * 255).astype(np.uint8))  # 0-255로 변환
    img_to_save.save(f'./images/image_{i+1}.jpg')  # 이미지 저장
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# CSV 파일 읽기
df = pd.read_csv('StockVector.csv')

# 이미지 수 및 데이터 준비
num_rows = df.shape[0]
images = df.values[:, :44].reshape(num_rows, 11, 4)
print(num_rows)

# 저장할 디렉토리 설정
dir_0 = 'D:/Images/0'  # 마지막 열이 0일 때 저장할 경로
dir_1 = 'D:/Images/1'  # 마지막 열이 1일 때 저장할 경로

# 디렉토리가 없으면 생성
os.makedirs(dir_0, exist_ok=True)
os.makedirs(dir_1, exist_ok=True)

for i in tqdm(range(num_rows)):
    img_array = (images[i] * 255).astype(np.uint8)
    img_to_save = Image.fromarray(img_array)

    resized_image = img_to_save.resize((128, 128))

    plt.figure(figsize=(10, 10))
    
    row = df.iloc[i].values[:-1]
    print(row)
    normalized_row = (row - np.min(row)) / (np.max(row) - np.min(row))
    
    x = np.linspace(0, 2 * np.pi, len(normalized_row))
    y = normalized_row * np.sin(x * 2) + normalized_row * np.cos(x * 3)
    
    plt.plot(x, y, color='white')
    plt.axis('off')
    plt.imshow(resized_image, aspect='auto', extent=[0, 2 * np.pi, np.min(y), np.max(y)])
    plt.axis('off')

    # 마지막 열의 값에 따라 저장할 디렉토리 선택
    last_value = df.iloc[i].values[-1:]  # 마지막 열의 값
    print(last_value)
    if last_value == 1:
        save_path = os.path.join(dir_1, f'image_{i + 1}.jpg')
    else:
        save_path = os.path.join(dir_0, f'image_{i + 1}.jpg')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)  # 이미지 저장
    plt.close()  # 현재 플롯 닫기
