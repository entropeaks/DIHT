import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import hydra
from omegaconf import OmegaConf

from src.data import CachedCollection, PKSampler, DataPreparator, extract_paths_and_labels, make_transform
from src.model import SiameseDino
from src.train import train, evaluate
from src.config import Config
from src.utils import set_device

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2

@hydra.main(config_path="../config", config_name="base_config", version_base=None)
def main(config: Config):

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),   
        v2.RandomHorizontalFlip(p=0.5),                            
        v2.RandomVerticalFlip(p=0.1),                              
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0), 
        v2.RandomRotation(degrees=15),                             
    ])

    device = set_device(config.base.device)

    RANDOM_STATE = 42
    
    processor = AutoImageProcessor.from_pretrained(config.model.backbone_name)
    backbone = AutoModel.from_pretrained(config.model.backbone_name, dtype=torch.float32)

    EXPERIMENT_DATA_PATH = Path(config.dataset.augmented_dataset_path)
    ORIGINAL_DATA_PATH = Path(config.dataset.original_dataset_path)

    data_preparator = DataPreparator(ORIGINAL_DATA_PATH, EXPERIMENT_DATA_PATH, RANDOM_STATE)
    original_data_splits = data_preparator.train_val_test_split(0, 1, config.eval.k_query)
    train_paths, train_labels = extract_paths_and_labels(EXPERIMENT_DATA_PATH)
    gallery_paths, gallery_labels = original_data_splits["gallery"]
    queries_paths, queries_labels = original_data_splits["val_query"]

    train_dataset = CachedCollection(train_paths, train_labels, transform=v2.Compose([train_transforms, make_transform()]))
    pksampler = PKSampler(train_dataset, config.train.sampler_P, config.train.sampler_K)
    train_dataloader = DataLoader(train_dataset, batch_sampler=pksampler)

    gallery_dataset = CachedCollection(gallery_paths, gallery_labels, make_transform())
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)
    queries_dataset = CachedCollection(queries_paths, queries_labels, make_transform())
    queries_dataloader = DataLoader(queries_dataset, batch_size=32)

    siamese_model = SiameseDino(backbone, hidden_dim=config.model.hidden_dim, output_dim=config.model.output_dim, normalize=config.model.normalize, dropout=config.model.dropout)
    for param in siamese_model.backbone.parameters():
        param.requires_grad = False
    for param in siamese_model.backbone.layer[-2:].parameters():
        param.requires_grad = True
    _ = siamese_model.to(device)

    optim_params = [
        {"params": siamese_model.projection_head.parameters(), "lr": config.train.head_lr},
        {"params": siamese_model.backbone.layer[-2:].parameters(), "lr": config.train.backbone_lr}
    ]

    model_id = "best_train_color_invariant_LLft"
    losses, scores = train(model_id,
                           siamese_model,
                           processor,
                           train_dataloader,
                           gallery_dataloader,
                           queries_dataloader,
                           optim_params,
                           config.train.epochs,
                           config.train.weight_decay,
                           config.train.margin,
                           config.eval.recall_k)
    
    print(f"Training ended with loss={losses[-1]}")
    print(f"Model evaluation on data seen during training:\n{scores}")


if __name__ == "__main__":
    main()