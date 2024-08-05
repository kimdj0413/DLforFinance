import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

df = pd.read_csv('StockVector.csv', nrows=10000)

num_rows = df.shape[0]
images = df.values[:, :44].reshape(num_rows, 11, 4)
print(num_rows)

dir_0 = 'D:/Images/0'
dir_1 = 'D:/Images/1'

os.makedirs(dir_0, exist_ok=True)
os.makedirs(dir_1, exist_ok=True)

for i in tqdm(range(num_rows)):
    img_array = (images[i] * 255).astype(np.uint8)
    img_to_save = Image.fromarray(img_array)

    resized_image = img_to_save.resize((128, 128))
    
    last_value = df.iloc[i].values[-1:]
    if last_value == 1:
        save_path = os.path.join(dir_1, f'image_{i + 1}.jpg')
    else:
        save_path = os.path.join(dir_0, f'image_{i + 1}.jpg')
    plt.figure(figsize=(10, 10))
    plt.imshow(resized_image)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close() 