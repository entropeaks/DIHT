from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms as v2
import torch.nn.functional as F
from transformers import AutoImageProcessor


@torch.no_grad()
def evaluate(model: nn.Module, processor, gallery_loader: DataLoader, query_loader: DataLoader, recall_k: list=[1, 3]):
    """
    Compute Recall@k in pure PyTorch.
    
    Args:
        model: torch.nn.Module returning embeddings
        gallery_loader: DataLoader with (image, label) for gallery
        query_loader: DataLoader with (image, label) for queries
        k: int, top-k
        device: str
    
    Returns:
        recall_at_k: float
    """
    model.eval()

    # 1. Compute gallery embeddings
    gallery_embs, gallery_labels = [], []
    for images, labels in gallery_loader:
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        emb = model(**inputs)
        if not model.normalize:
            emb = F.normalize(emb, p=2, dim=1)  # normalize
        gallery_embs.append(emb.cpu())
        gallery_labels.append(labels)
    gallery_embs = torch.cat(gallery_embs, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0)

    # 2. Compute query embeddings
    query_embs, query_labels = [], []
    for images, labels in query_loader:
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        emb = model(**inputs)
        if not model.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        query_embs.append(emb.cpu())
        query_labels.append(labels)
    query_embs = torch.cat(query_embs, dim=0)
    query_labels = torch.cat(query_labels, dim=0)

    # 3. Compute pairwise distances (queries vs gallery)
    # torch.cdist gives Euclidean distances; use cosine if you prefer
    dists = torch.cdist(query_embs, gallery_embs, p=2)

    # 4. For each query, get top-k closest gallery items
    topk_indices = dists.topk(max(recall_k), largest=False).indices  # (num_queries, k)

    # 5. Compute Recall@k: score of 1 if correct label in one of the k first returned items
    recall_at_k = {}
    for k in recall_k:
        correct = 0
        for i, q_label in enumerate(query_labels):
            retrieved_labels = gallery_labels[topk_indices[i, :k]]
            if (retrieved_labels == q_label).any():
                correct += 1
        recall_at_k[f"recall@{k}"] = correct / len(query_labels)

    return recall_at_k


def mine_hard_triplets_cdist(embeddings: torch.Tensor, labels: np.ndarray, oneNegativeOnlyPerAnchor: bool=False):
    """
    Vectorized hard negative mining using torch.cdist.
    For each anchor, selects a random positive and the closest hard negative (neg_dist < pos_dist).
    embeddings: torch.Tensor of shape (N, D)
    labels: list or np.array of length N
    Returns: list of (anchor_idx, positive_idx, hard_negative_idx)
    """
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings)
    labels = labels.numpy()
    n = embeddings.shape[0]
    dists = torch.cdist(embeddings, embeddings, p=2)
    triplets = []
    for anchor_idx in range(n):
        anchor_label = labels[anchor_idx]
        pos_mask = (labels == anchor_label) & (np.arange(n) != anchor_idx)
        pos_indices = np.where(pos_mask)[0]
        neg_mask = (labels != anchor_label)
        neg_indices = np.where(neg_mask)[0]
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue
        positive_idx = np.random.choice(pos_indices)
        pos_dist = dists[anchor_idx, positive_idx].item()
        hard_neg_mask = dists[anchor_idx, neg_indices] < pos_dist
        hard_neg_indices = neg_indices[hard_neg_mask.cpu().numpy()]
        if oneNegativeOnlyPerAnchor and len(hard_neg_indices) > 0:
            hard_neg_dists = dists[anchor_idx, hard_neg_indices]
            hard_neg_idx = hard_neg_indices[torch.argmin(hard_neg_dists).item()]
            triplets.append((anchor_idx, positive_idx, hard_neg_idx))
        else:
            for hard_neg_idx in hard_neg_indices:
                triplets.append((anchor_idx, positive_idx, hard_neg_idx))
    return triplets


def mine_semi_hard_triplets_cdist(embeddings: torch.Tensor, labels: np.ndarray, margin: float):
    """
    Vectorized semi-hard negative mining using torch.cdist.
    For each anchor, finds a random positive and all semi-hard negatives.
    A semi-hard negative `n` satisfies: d(a, p) < d(a, n) < d(a, p) + margin
    
    embeddings: torch.Tensor of shape (N, D)
    labels: list or np.array of length N
    margin: float, the margin used in the TripletLoss
    
    Returns: list of (anchor_idx, positive_idx, semi_hard_negative_idx)
    """
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings)
    # Assurez-vous que les labels sont un ndarray numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n = embeddings.shape[0]
    # Calcule la matrice des distances au carré pour la stabilité, ou p=2 pour euclidienne
    dists = torch.cdist(embeddings, embeddings, p=2)
    
    triplets = []
    for anchor_idx in range(n):
        anchor_label = labels[anchor_idx]
        
        # Masques pour positifs et négatifs
        pos_mask = (labels == anchor_label) & (np.arange(n) != anchor_idx)
        pos_indices = np.where(pos_mask)[0]
        
        neg_mask = (labels != anchor_label)
        neg_indices = np.where(neg_mask)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue
            
        # Itérer sur tous les positifs possibles pour cet ancre
        for positive_idx in pos_indices:
            pos_dist = dists[anchor_idx, positive_idx]
            
            # --- C'est ici que la logique change ---
            # Condition 1: d(a, n) > d(a, p)
            cond1 = dists[anchor_idx, neg_indices] > pos_dist
            # Condition 2: d(a, n) < d(a, p) + margin
            cond2 = dists[anchor_idx, neg_indices] < (pos_dist + margin)
            
            semi_hard_neg_mask = cond1 & cond2
            
            semi_hard_indices = neg_indices[semi_hard_neg_mask.cpu().numpy()]
            
            for semi_hard_neg_idx in semi_hard_indices:
                triplets.append((anchor_idx, positive_idx, semi_hard_neg_idx))
                
    return triplets


def train(model: nn.Module, processor, dataloader: DataLoader, epochs=10, lr=1e-4, margin=0.2):
    model.train()
    losses = []
    loss = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        for images, labels in dataloader:
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            embeddings = model(**inputs)
            triplets = mine_semi_hard_triplets_cdist(embeddings, labels)
            if not triplets:
                continue
            anchor_indices, positive_indices, negative_indices = zip(*triplets)
            anchor_embeddings = embeddings[list(anchor_indices)]
            positive_embeddings = embeddings[list(positive_indices)]
            negative_embeddings = embeddings[list(negative_indices)]
            anchor_embeddings = anchor_embeddings.to(model.device)
            positive_embeddings = positive_embeddings.to(model.device)
            negative_embeddings = negative_embeddings.to(model.device)
            optimizer.zero_grad()
            triplet_loss = loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            triplet_loss.backward()
            optimizer.step()
            losses.append(triplet_loss.detach().cpu().item())

    return losses


def train_and_evaluate(run: wandb.Run,
                       model: nn.Module,
                       processor: AutoImageProcessor,
                       train_dataloader: DataLoader,
                       gallery_dataloader: DataLoader,
                       val_dataloader: DataLoader,
                       optim_params: dict,
                       epochs: int=10,
                       weight_decay: float=1e-4,
                       margin: int=0.2,
                       recall_k: list=[5]):
    
    scores = {f"recall@{k}": 0 for k in recall_k}
    best_score = 0.0
    best_loss = float('inf')
    loss = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.AdamW(optim_params, weight_decay=weight_decay)
    for epoch in tqdm(range(epochs)):
        model.train()
        cumulative_loss = 0.0
        cumulative_pos_dist = 0.0
        cumulative_neg_dist = 0.0
        cumulative_triplets_count = 0
        for images, labels in train_dataloader:
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            embeddings = model(**inputs)
            triplets = mine_semi_hard_triplets_cdist(embeddings, labels, margin)
            if not triplets:
                continue
            cumulative_triplets_count += len(triplets)
            anchor_indices, positive_indices, negative_indices = zip(*triplets)
            anchor_embeddings = embeddings[list(anchor_indices)]
            positive_embeddings = embeddings[list(positive_indices)]
            negative_embeddings = embeddings[list(negative_indices)]
            anchor_embeddings = anchor_embeddings.to(model.device)
            positive_embeddings = positive_embeddings.to(model.device)
            negative_embeddings = negative_embeddings.to(model.device)
            optimizer.zero_grad()
            triplet_loss = loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            triplet_loss.backward()
            optimizer.step()
            cumulative_pos_dist += F.pairwise_distance(anchor_embeddings, positive_embeddings, p=2).mean().item()
            cumulative_neg_dist += F.pairwise_distance(anchor_embeddings, negative_embeddings, p=2).mean().item()
            cumulative_loss += triplet_loss.detach().cpu().item()

        cumulative_loss /= len(train_dataloader)
        cumulative_pos_dist /= len(train_dataloader)
        cumulative_neg_dist /= len(train_dataloader)
        scores = evaluate(model, processor, gallery_dataloader, val_dataloader, recall_k)
        run.log({"loss": cumulative_loss, "mean_pos_dist": cumulative_pos_dist, "mean_neg_dist": cumulative_neg_dist, "triplets_mined": cumulative_triplets_count, **scores})
        if np.mean([score for score in scores.values()]) > best_score:
            best_score = np.mean([score for score in scores.values()])
            torch.save(model.state_dict(), f"model_checkpoints/{run.name}.pth")
        if cumulative_loss < best_loss:
            best_loss = cumulative_loss
            torch.save(model.state_dict(), f"model_checkpoints/{run.name}_best_loss.pth")
    torch.save(model.state_dict(), f"model_checkpoints/{run.name}_last.pth")

    return scores