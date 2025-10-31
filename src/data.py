from abc import ABC, abstractmethod
from pathlib import Path
import random
from collections import defaultdict
from typing import Union
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.transforms import v2
from transformers.image_utils import load_image
from typing import Tuple
import numpy as np

RANDOM_SEED = 42


def make_transform(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def create_dataset_splits(
    original_dir: Path,
    augmented_dir: Path,
    train_ratio: float,
    val_ratio: float,
    k_query: int,
    seed: int=RANDOM_SEED
):
    """
    Creates train, gallery, and query splits based on class-level separation.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Get all available classes (e.g., ["001", "002", ...])
    # We assume the class folders are the same in both directories
    if not original_dir.exists():
        print(f"Error: Original data directory not found at {original_dir}")
        return None
        
    all_classes = sorted([d.name for d in original_dir.iterdir() if d.is_dir()])
    num_classes = len(all_classes)
    
    if num_classes == 0:
        print(f"Error: No class folders found in {original_dir}")
        return None

    print(f"Found {num_classes} total classes.")

    # 2. Split classes into train, val, and test
    shuffled_classes = np.random.permutation(all_classes)
    
    num_train = int(num_classes * train_ratio)
    num_val = int(num_classes * val_ratio)
    
    # Ensure at least 1 class in each split if ratios are small
    num_train = max(1, num_train)
    num_val = max(1, num_val)
    
    train_classes = set(shuffled_classes[:num_train])
    val_classes = set(shuffled_classes[num_train : num_train + num_val])
    test_classes = set(shuffled_classes[num_train + num_val:])
    
    print(f"Splitting classes: {len(train_classes)} Train / {len(val_classes)} Val / {len(test_classes)} Test\n")

    # 3. Initialize lists for our dataloaders
    train_paths, train_labels = [], []
    gallery_paths, gallery_labels = [], []
    val_query_paths, val_query_labels = [], []
    test_query_paths, test_query_labels = [], []

    # 4. Process Train Classes
    print("Processing Train classes...")
    for class_name in train_classes:
        # A. Populate Train Loader (from augmented data)
        aug_class_dir = augmented_dir / class_name
        aug_images = [str(p) for p in aug_class_dir.glob('*.*')]
        train_paths.extend(aug_images)
        train_labels.extend([int(class_name)] * len(aug_images))
        
        # B. Populate Gallery (from original data)
        # All original samples from train classes go into the gallery
        orig_class_dir = original_dir / class_name
        orig_images = [str(p) for p in orig_class_dir.glob('*.*')]
        gallery_paths.extend(orig_images)
        gallery_labels.extend([int(class_name)] * len(orig_images))

    # 5. Process Validation Classes
    print("Processing Validation classes...")
    for class_name in val_classes:
        orig_class_dir = original_dir / class_name
        all_class_images = [str(p) for p in orig_class_dir.glob('*.*')]
        random.shuffle(all_class_images) # Shuffle to pick random queries
        
        # A. Split into Query and Gallery
        val_query_paths.extend(all_class_images[:k_query])
        val_query_labels.extend([int(class_name)] * k_query)
        
        gallery_paths.extend(all_class_images[k_query:])
        gallery_labels.extend([int(class_name)] * (len(all_class_images) - k_query))

    # 6. Process Test Classes
    print("Processing Test classes...")
    for class_name in test_classes:
        orig_class_dir = original_dir / class_name
        all_class_images = [str(p) for p in orig_class_dir.glob('*.*')]
        random.shuffle(all_class_images) # Shuffle to pick random queries
        
        # A. Split into Query and Gallery
        test_query_paths.extend(all_class_images[:k_query])
        test_query_labels.extend([int(class_name)] * k_query)
        
        gallery_paths.extend(all_class_images[k_query:])
        gallery_labels.extend([int(class_name)] * (len(all_class_images) - k_query))

    print("\n--- Data Split Summary ---")
    print(f"Training Loader:   {len(train_paths):>5} samples from {len(train_classes)} classes (Augmented)")
    print(f"Gallery Loader:    {len(gallery_paths):>5} samples from {num_classes} classes (Original)")
    print(f"Val Query Loader:  {len(val_query_paths):>5} samples from {len(val_classes)} classes (Original)")
    print(f"Test Query Loader: {len(test_query_paths):>5} samples from {len(test_classes)} classes (Original)")
    
    # 7. Return all the file lists
    data_splits = {
        "train": (train_paths, train_labels),
        "gallery": (gallery_paths, gallery_labels),
        "val_query": (val_query_paths, val_query_labels),
        "test_query": (test_query_paths, test_query_labels)
    }
    
    return data_splits


class ImageCollectionDataset(ABC):
    def __init__(self, images_paths: list[Path], labels: list[int], transform: v2.Compose=None):
        self.images_paths = images_paths
        self.labels = labels
        self.transform = transform if transform else make_transform()
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: Union[int, list[int]]):
        pass


# ==== Dataset simple ====
class CachedCollection(ImageCollectionDataset, Dataset):
    def __init__(self, images_paths: list[Path], labels: list[int], transform: v2.Compose=None):
        super().__init__(images_paths, labels, transform)
        self.images_instances = self.load_images()

    def load_images(self) -> list:
        images_instances = []
        for path in self.images_paths:
            images_instances.append(v2.Resize((224, 224))(load_image(path)))
        return images_instances

    def __len__(self):
        return len(self.images_instances)

    def __getitem__(self, idx):
        img = self.transform(self.images_instances[idx])
        label = self.labels[idx]
        return img, label
    

class LazyLoadCollection(ImageCollectionDataset, Dataset):
    def __init__(self, images_paths: list[Path], labels: list[int], transform: v2.Compose=None):
        super().__init__(images_paths, labels, transform)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img = load_image(self.images_paths[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
    

# ==== Sampler pour P classes × K images ====
class PKSampler(Sampler):
    def __init__(self, dataset, P, K):
        self.P = P  # classes per batch
        self.K = K  # images per class

        if hasattr(dataset, "indices"):
            # dataset is a Subset
            self.subset_indices = dataset.indices  # indices in the original dataset
            self.dataset = dataset.dataset         # the original dataset
            # Build class_to_indices for the subset (values are indices in the subset, not the original dataset)
            self.class_to_indices = {}
            for cls, orig_indices in self.dataset.class_to_indices.items():
                # Find which indices are in the subset
                subset_class_indices = [i for i, idx in enumerate(self.subset_indices) if idx in orig_indices]
                if subset_class_indices:
                    self.class_to_indices[cls] = subset_class_indices
            self.classes = list(self.class_to_indices.keys())
        else:
            self.dataset = dataset
            self.class_to_indices = self.dataset.class_to_indices
            self.classes = list(self.class_to_indices.keys())

    def __iter__(self):
        for _ in range(len(self)):
            # choose P classes at random
            selected_classes = random.sample(self.classes, self.P)
            batch_indices = []
            for c in selected_classes:
                indices = self.class_to_indices[c]
                # choose K samples from these indices
                chosen = []
                start_idx = random.randint(0, len(indices)-1)
                for i in range(self.K):
                    chosen.append(indices[(start_idx+i) % len(indices)])
                batch_indices.extend(chosen)
            yield batch_indices

    def __len__(self):
        return len(self.class_to_indices) // self.P
