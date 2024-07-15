import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv('../MLforFinance/ready.csv')

# 데이터 전처리
df = df.iloc[:,1:]
X = df.drop('Column49', axis=1)
y = df['Column49']

# 스케일링 적용
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 시계열 데이터를 위해 3D 형태로 변환
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10  # 시퀀스 길이
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 레이블을 원-핫 인코딩
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# CNN 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_steps, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 출력층에서 클래스 수만큼 노드 생성

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train_cat, epochs=3, batch_size=32, validation_split=0.2)

# 예측
y_pred_cat = model.predict(X_test)
y_pred = np.argmax(y_pred_cat, axis=1)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
