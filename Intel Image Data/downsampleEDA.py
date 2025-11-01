import os
import random
import shutil
from PIL import Image
import matplotlib.pyplot as plt

# CONFIG
data_path = "archive/seg_train/seg_train"         # original dataset
output_path = "archive/downsampled_seg_train"     # new downsampled dataset
target_count = 1000                               # images per category
target_size = (64, 64)                            # resize target
random_seed = 42                                  # reproducibility

random.seed(random_seed)

# Output directory 
os.makedirs(output_path, exist_ok=True)

# Get list of category folders
data_categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
print("Detected categories:", data_categories, "\n")

selected_per_category = {}

# Randomly select and resize 
for category in data_categories:
    category_path = os.path.join(data_path, category)
    output_category_path = os.path.join(output_path, category)
    os.makedirs(output_category_path, exist_ok=True)
    
    # Collect all .jpg/.JPG files
    all_images = [f for f in os.listdir(category_path)
                  if os.path.isfile(os.path.join(category_path, f)) and f.lower().endswith(".jpg")]
    
    n_total = len(all_images)
    
    k = min(target_count, n_total)
    sampled_images = random.sample(all_images, k)
    selected_per_category[category] = sampled_images
    
    print(f"{category}: selected {k} of {n_total} images.")
    
    # Resize and save in new folder
    for img_name in sampled_images:
        src_path = os.path.join(category_path, img_name)
        dst_path = os.path.join(output_category_path, img_name)
        img = Image.open(src_path).convert("RGB")
        img = img.resize(target_size)
        img.save(dst_path)

# Display one sample image per category 
plt.figure(figsize=(10, 6))
cols = 3
rows = (len(data_categories) + cols - 1) // cols

for i, category in enumerate(data_categories, 1):
    output_category_path = os.path.join(output_path, category)
    if not os.path.exists(output_category_path):
        continue

    # pick first image
    first_img_name = os.listdir(output_category_path)[0]
    img_path = os.path.join(output_category_path, first_img_name)

    img = Image.open(img_path)
    plt.subplot(rows, cols, i)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")

plt.tight_layout()
plt.show()
