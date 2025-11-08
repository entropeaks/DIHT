from abc import ABC, abstractmethod
from typing import List, Dict, TypeAlias
import numpy as np

Score: TypeAlias = float | List[float]


class Metric(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def compute() -> Score:
        pass


class Recall(Metric):
    def __init__(self, k: List=[1, 3, 10]):
        self.k = k

    def compute(self, matrix: np.ndarray, labels: np.ndarray) -> Dict:
        sorted_idx = np.argsort(matrix, axis=1)
        k_successes = np.zeros(len(self.k))
        for i in range(len(sorted_idx)):
            ref_label = labels[i]
            for shift, k in enumerate(self.k):
                nearest_neighbours = [labels[sorted_idx[i][-k:]]] if k==1 else labels[sorted_idx[i][-k:]]
                if ref_label in nearest_neighbours:
                    k_successes[shift] += 1
        
        return k_successes/len(sorted_idx)