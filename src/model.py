import torch.nn as nn
import torch.nn.functional as F


class SiameseDino(nn.Module):
    def __init__(self, base_model, hidden_dim, output_dim, normalize: bool=True, dropout: float=0.0):
        super().__init__()
        self.base_model = base_model
        embedding_dim = base_model.config.hidden_size
        self.normalize = normalize
        self.dropout = dropout
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
            ) if hidden_dim > 0 else (nn.Linear(embedding_dim, output_dim), nn.Dropout(dropout))


    def forward(self, **inputs):
        outputs = self.base_model(**inputs)
        x = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        x = self.projection_head(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x
    
    def to(self, device):
        self.base_model.to(device)
        self.projection_head.to(device)
        self.device = device
        return self
    

class PrototypicalNetwork(nn.Module):
    def __init__(self, base_model, output_dim, normalize: bool=True, dropout: float=0.0):
        super().__init__()
        pass