import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_dir, split_ratio=0.8, seed=42):
    random.seed(seed)
    classes = os.listdir(source_dir)

    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_imgs = images[:split_point]
        val_imgs = images[split_point:]

        # Create destination dirs
        for split in ['train', 'val']:
            Path(os.path.join(dest_dir, split, cls)).mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_imgs:
            src = os.path.join(class_dir, img)
            dst = os.path.join(dest_dir, 'train', cls, img)
            shutil.copy2(src, dst)

        for img in val_imgs:
            src = os.path.join(class_dir, img)
            dst = os.path.join(dest_dir, 'val', cls, img)
            shutil.copy2(src, dst)

        print(f"[{cls}] Train: {len(train_imgs)}, Val: {len(val_imgs)}")

if __name__ == "__main__":
    source = "./data_raw"
    destination = "./data"
    split_dataset(source, destination)
