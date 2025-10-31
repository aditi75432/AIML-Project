import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Set the data path and get the list of categories, there are total of 6 categories
data_path = "archive/seg_train/seg_train"
data_categories = sorted(os.listdir(data_path))

# Display the number of image in each category each category has more than 2000 images hence we can say they are equally distributed
for category in data_categories:
    category_path = os.path.join(data_path, category)
    print(f"{category} images: {len(os.listdir(category_path))}")
    
# Downsample images to 64x64 for faster processing
target_size = (64, 64)

for category in data_categories:
    category_path = os.path.join(data_path, category)

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        img.save(img_path)  

# Display one sample image from each category to verify resizing
plt.figure(figsize=(10, 5))

for i, category in enumerate(data_categories, 1):
    category_path = os.path.join(data_path, category)
    if not os.path.isdir(category_path):
        continue
    first_img_name = os.listdir(category_path)[0]
    img_path = os.path.join(category_path, first_img_name)
    
    img = Image.open(img_path)
    plt.subplot(2, 3, i)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")

plt.tight_layout()
plt.show()

