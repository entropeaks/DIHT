import yaml
from pathlib import Path
import wandb

from src.model import SiameseDino
from src.data import PKSampler, CachedCollection, LazyLoadCollection,  make_transform, create_dataset_splits
from src.training import train_and_evaluate, evaluate
from src.utils import set_device, load_config

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2


config = load_config("config/config.yaml")
training_config = load_config("config/training.yaml")
model_config = load_config("config/model.yaml")
data_config = load_config("config/data.yaml")
eval_config = load_config("config/evaluation.yaml")

train_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),   
    v2.RandomHorizontalFlip(p=0.5),                            
    v2.RandomVerticalFlip(p=0.1),                              
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0), 
    v2.RandomRotation(degrees=15),                             
])

device = set_device(config["device"])

random_state = 42

processor = AutoImageProcessor.from_pretrained(model_config["backbone_name"])
backbone = AutoModel.from_pretrained(model_config["backbone_name"], dtype=torch.float32)

EXPERIMENT_DATA_PATH = Path(data_config["augmented_dataset_path"])
ORIGINAL_DATA_PATH = Path(data_config["original_dataset_path"])

training_size = 1 - (eval_config["val_size"] + eval_config["test_size"])
data_splits = create_dataset_splits(ORIGINAL_DATA_PATH, EXPERIMENT_DATA_PATH, training_size, eval_config["val_size"], eval_config["k_query"], random_state)
train_paths, train_labels = data_splits["train"]
gallery_paths, gallery_labels = data_splits["gallery"]
val_query_paths, val_query_labels = data_splits["val_query"]
test_query_paths, test_query_labels = data_splits["test_query"]

train_dataset = CachedCollection(train_paths, train_labels, transform=v2.Compose([train_transforms, make_transform()]))
pksampler = PKSampler(train_dataset, training_config["sampler"]["P"], training_config["sampler"]["K"])
train_dataloader = DataLoader(train_dataset, batch_sampler=pksampler)

gallery_dataset = CachedCollection(gallery_paths, gallery_labels, transform=make_transform())
gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)

val_dataset = CachedCollection(val_query_paths, val_query_labels, transform=make_transform())
val_dataloader = DataLoader(val_dataset, batch_size=32)

# --------------------------------------------------------------------------
siamese_model = SiameseDino(backbone, hidden_dim=model_config["hidden_dim"], output_dim=model_config["output_dim"], normalize=model_config["normalize"], dropout=model_config["dropout"])
for param in siamese_model.backbone.parameters():
    param.requires_grad = False
for param in siamese_model.backbone.layer[:2].parameters():
    param.requires_grad = True
_ = siamese_model.to(device)

optim_params = [
    {"params": siamese_model.projection_head.parameters(), "lr": training_config["head_lr"]},
    {"params": siamese_model.backbone.layer[:2].parameters(), "lr": training_config["backbone_lr"]}
]

run = wandb.init(project=config["wandb"]["project_name"], entity=config["wandb"]["entity"], config=config)
train_and_evaluate(run,
                   siamese_model,
                   processor,
                   train_dataloader,
                   gallery_dataloader,
                   val_dataloader,
                   optim_params,
                   training_config["epochs"],
                   training_config["weight_decay"],
                   training_config["margin"],
                   eval_config["recall_k"]
                   )
run.finish()