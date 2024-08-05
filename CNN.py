import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os

# 데이터 전처리 및 로드
data_dir = 'D:\\images'

# 이미지와 라벨을 저장할 리스트
images = []
labels = []

# 데이터셋 로드
for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(int(label))

# numpy 배열로 변환
images = np.array(images)
labels = np.array(labels)

# Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

# 클래스 비율 맞추기 위한 샘플링
class_counts = np.bincount(y_train)
class_weights = {0: class_counts[1] / class_counts[0], 1: class_counts[0] / class_counts[1]}

# 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 이진 분류를 위해 softmax 사용
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
num_epochs = 10
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset, class_weight=class_weights)

# 최종 테스트 결과
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'최종 테스트 손실: {test_loss:.4f}, 최종 테스트 정확도: {test_accuracy:.2f}%')
