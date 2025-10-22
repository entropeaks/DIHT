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


def make_transform(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

#dirty hacky way to do stratified split
def train_test_split(paths: list, labels: list, ratio: int, random_state: int=None) -> Tuple[list, list, list, list]:
    paths = np.array(paths)
    labels = np.array(labels)
    random.seed(a=random_state)
    n_classes = len(set(labels))
    samples_per_class = int(len(labels)/n_classes)
    train_indices = []
    test_indices = []
    split_size = int(samples_per_class*ratio)
    for i in range(n_classes):
        cursor = i*samples_per_class
        split_idx = random.randint(0, samples_per_class-1)
        for i in range(split_size):
            test_indices.append(cursor+(split_idx + i) % samples_per_class)
        for i in range(samples_per_class - split_size):
            train_indices.append(cursor+(split_idx + split_size + i) % samples_per_class)
    
    return paths[train_indices].tolist(), paths[test_indices].tolist(), labels[train_indices].tolist(), labels[test_indices].tolist()


def extractPaths(root_path: Path) -> Tuple[list[Path], list[int]]:
    images_paths = []
    labels = []
    for label, class_dir in enumerate(root_path.iterdir()):
            if class_dir.is_dir():
                for image_path in class_dir.iterdir():
                    images_paths.append(image_path.as_posix())
                    labels.append(label)
    
    return images_paths, labels


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
