import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from src.eval import Metric, Score


class SiameseDino(nn.Module):
    def __init__(self, backbone, hidden_dim: int, output_dim: int, normalize: bool=True, dropout: float=0.0):
        super().__init__()
        self.backbone = backbone
        embedding_dim = backbone.config.hidden_size
        self.normalize = normalize
        self.dropout = dropout
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
            ) if hidden_dim > 0 else nn.Sequential(nn.Linear(embedding_dim, output_dim), nn.Dropout(dropout))


    def forward(self, **inputs):
        outputs = self.backbone(**inputs)
        x = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        x = self.projection_head(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x
    
    def to(self, device):
        self.backbone.to(device)
        self.projection_head.to(device)
        self.device = device
        return self
    

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone, output_dim, normalize: bool=True, dropout: float=0.0):
        super().__init__()
        pass


class OrbAlgorithmTester:

    def __init__(self, paths: list, labels: list):
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.descriptors = self._get_imgs_descriptors()
        self.similarity_matrix = self._compute_similarity_matrix()
        

    def _get_imgs_descriptors(self):
        print("Computing images descriptors...")
        descriptors = []
        for img_path in tqdm(self.paths):
            img_descriptors = self._compute_img_descriptors(img_path)
            descriptors.append(img_descriptors)
        
        return descriptors

    def _compute_img_descriptors(self, path_to_img: str):
        img = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
        _, descriptors = self.orb.detectAndCompute(img, None)

        return descriptors

    def evaluate(self, metric: Metric) -> Score:
        return metric.compute(self.similarity_matrix, self.labels)

    def _compute_similarity_matrix(self):
        print("Computing similarity matrix...")
        n = len(self.descriptors)
        m = np.zeros((n, n))
        for i in tqdm(range(len(self.descriptors))):
            for j in range(i):
                m[i][j] = m[j][i]
            for j in range(1, len(self.descriptors)-i):
                m[i][j+i] = self._compute_similarity(self.descriptors[i], self.descriptors[j+i])

        return m
    
    def _compute_similarity(self, des1: str, des2: str) -> float:
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        avg_distance = sum(m.distance for m in matches) / len(matches)
        similarity = 1 / (1 + avg_distance)

        return similarity