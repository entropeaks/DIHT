import yaml
from pathlib import Path
import wandb

from src.model import SiameseDino
from src.data import PKSampler, CachedCollection, extractPaths, make_transform, train_test_split
from src.training import train_and_evaluate, evaluate

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2

device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")


with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

run = wandb.init(project=config["wandb"]["project_name"], entity=config["wandb"]["entity"], config=config)

train_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),   
    v2.RandomHorizontalFlip(p=0.5),                            
    v2.RandomVerticalFlip(p=0.1),                              
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
    v2.RandomRotation(degrees=15),                             
])

training_config = config["training"]
model_config = config["model"]
data_config = config["data"]

random_state = 42

processor = AutoImageProcessor.from_pretrained(model_config["base_model_name"])
base_model = AutoModel.from_pretrained(model_config["base_model_name"], dtype=torch.float32)

# --------------------------------------------------------------------------
# For hyperparameter search, we create a separate training and validation set from the dataset specified
experiment_data_root_path = Path(data_config["dataset_path"])
experiment_images_paths, experiment_labels = extractPaths(experiment_data_root_path)
train_paths, val_paths, train_labels, val_labels = train_test_split(experiment_images_paths, experiment_labels, data_config["test_size"], random_state=random_state)
train_dataset = CachedCollection(train_paths, train_labels, transform=v2.Compose([train_transforms, make_transform()]))
pksampler = PKSampler(train_dataset, config["data"]["sampler"]["P"], config["data"]["sampler"]["K"])
train_dataloader = DataLoader(train_dataset, batch_sampler=pksampler)
val_dataset = CachedCollection(val_paths, val_labels, transform=make_transform())
val_dataloader = DataLoader(val_dataset, batch_size=32)

# We will use a separate test set for the final evaluation
original_data_root_path = Path("data")
original_images_paths, original_labels = extractPaths(original_data_root_path)
gallery_paths, test_paths, gallery_labels, test_labels = train_test_split(original_images_paths, original_labels, config["evaluation"]["test_size"], random_state=random_state)
gallery_dataset = CachedCollection(gallery_paths, gallery_labels, transform=make_transform())
test_dataset = CachedCollection(test_paths, test_labels, transform=make_transform())
gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# --------------------------------------------------------------------------
siamese_model = SiameseDino(base_model, hidden_dim=model_config["hidden_dim"], output_dim=model_config["output_dim"], normalize=model_config["normalize"], dropout=model_config["dropout"])
for param in siamese_model.base_model.parameters():
    param.requires_grad = False
_ = siamese_model.to(device)

train_and_evaluate(run, siamese_model, processor, train_dataloader, val_dataloader, training_config["epochs"], training_config["lr"], training_config["weight_decay"], training_config["margin"], config["evaluation"]["recall_k"])
test_scores = evaluate(siamese_model, processor, gallery_dataloader, test_dataloader, recall_k=config["evaluation"]["recall_k"])
    
run.log({f"test_{k}": v for k, v in test_scores.items()})
print("Scores on test data:", test_scores)

run.finish()