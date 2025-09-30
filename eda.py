import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


data_path = "archive/seg_train/seg_train/"

data_categories = sorted(os.listdir(data_path))
print("Classes:", data_categories)

img_counts = {cat: len(os.listdir(os.path.join(data_path, cat))) for cat in data_categories}

print(img_counts)

df_counts = pd.DataFrame(list(img_counts.items()), columns=['Category', 'Image_Count'])
print(df_counts)

plt.figure(figsize=(12,6))
sns.barplot(x='Category', y='Image_Count', data=df_counts, palette="viridis")
plt.title("Number of Images per Category")
plt.show()


for i, cat in enumerate(data_categories[:6]): 
    plt.figure(figsize=(5, 8))
    img_name = sorted(os.listdir(os.path.join(data_path, cat)))[1]
    img_path = os.path.join(data_path, cat, img_name)
    img = Image.open(img_path)

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(cat)
    plt.axis("off")

plt.show()

plt.figure(figsize=(12, 12))

for i, cat in enumerate(data_categories[:6]):
    img_name = sorted(os.listdir(os.path.join(data_path, cat)))[1]
    img_path = os.path.join(data_path, cat, img_name)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    plt.subplot(3, 6, 2*i+1)
    plt.imshow(img_rgb)
    plt.title(cat + " (Original)")
    plt.axis("off")

    plt.subplot(3, 6, 2*i+2)
    plt.imshow(edges, cmap="hot")
    plt.title("Focus Areas")
    plt.axis("off")

plt.suptitle("First Image per Category with Focus Areas", fontsize=16)
plt.show()