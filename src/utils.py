"""
Utility functions for Dogs vs Cats classification
"""
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_image_paths(data_dir):
    """
    Load all image paths and labels from the dataset
    """
    image_paths = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            # Extract label from filename
            if filename.startswith('cat'):
                labels.append(0)
                image_paths.append(os.path.join(data_dir, filename))
            elif filename.startswith('dog'):
                labels.append(1)
                image_paths.append(os.path.join(data_dir, filename))
    
    return np.array(image_paths), np.array(labels)

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    Load and preprocess a single image
    """
   
    img = Image.open(image_path)
    
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    return img_array

def visualize_sample_images(image_paths, labels, num_samples=10):
    """
    Display random sample images from the dataset
    """
    plt.figure(figsize=(15, 6))
    indices = np.random.choice(len(image_paths), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        img = load_and_preprocess_image(image_paths[idx], target_size=(128, 128))
        plt.imshow(img)
        label = "Dog" if labels[idx] == 1 else "Cat"
        plt.title(label)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_basic_features(image_paths, feature_type='pixel', target_size=(64, 64)):
    """
    Extract features from images
    feature_type: 'pixel' (raw pixels) or 'hog' (Histogram of Oriented Gradients)
    """
    features = []
    
    for img_path in image_paths:
        img = load_and_preprocess_image(img_path, target_size)
        
        if feature_type == 'pixel':
            # Flatten the image
            features.append(img.flatten())
        
        elif feature_type == 'hog':
            # Extract HOG features
            from skimage.feature import hog
            # Convert to grayscale for HOG
            gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=False)
            features.append(hog_features)
    
    return np.array(features)

def split_dataset(image_paths, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    """
    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train and val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test