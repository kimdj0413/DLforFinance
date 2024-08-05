import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

# 사용 가능한 GPU 리스트 출력
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU 사용 가능:", gpus)
else:
    print("GPU를 사용할 수 없습니다.")
    
# 데이터 경로 설정
base_dir = 'D:/images'
train_dir = os.path.join(base_dir, 'D:/images/train')
validation_dir = os.path.join(base_dir, 'D:/images/validation')

# ImageDataGenerator를 통해 이미지 데이터를 로드하고 전처리
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # 이진 분류
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# GoogLeNet (InceptionV3) 모델 불러오기
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 모델에 추가 레이어 추가
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=100
)

# 평가
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

"""
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 정규화
    prediction = model.predict(img_array)
    return prediction

result = predict_image('path/to/new/image.jpg')
print("Predicted class:", '1' if result[0][0] > 0.5 else '0')
"""