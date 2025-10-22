import yaml
from pathlib import Path
import wandb
import optuna
from functools import partial

from src.model import SiameseDino
from src.data import PKSampler, CachedCollection, extractPaths, make_transform, train_test_split
from src.training import train_and_evaluate, evaluate

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2
device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")


with open("config/optuna_config.yaml", "r") as f:
    config = yaml.safe_load(f)

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

processor = AutoImageProcessor.from_pretrained(model_config["base_model_name"])
base_model = AutoModel.from_pretrained(model_config["base_model_name"], dtype=torch.float32)

# --------------------------------------------------------------------------
# For hyperparameter search, we create a separate training and validation set from the dataset specified
experiment_data_root_path = Path(data_config["dataset_path"])
experiment_images_paths, experiment_labels = extractPaths(experiment_data_root_path)
train_paths, val_paths, train_labels, val_labels = train_test_split(experiment_images_paths, experiment_labels, data_config["test_size"])
train_dataset = CachedCollection(train_paths, train_labels, transform=v2.Compose([train_transforms, make_transform()]))
val_dataset = CachedCollection(val_paths, val_labels, transform=make_transform())
val_dataloader = DataLoader(val_dataset, batch_size=32)

# We will use a separate test set for the final evaluation
original_data_root_path = Path("data")
original_images_paths, original_labels = extractPaths(original_data_root_path)
gallery_paths, test_paths, gallery_labels, test_labels = train_test_split(original_images_paths, original_labels, config["evaluation"]["test_size"])
gallery_dataset = CachedCollection(gallery_paths, gallery_labels, transform=make_transform())
test_dataset = CachedCollection(test_paths, test_labels, transform=make_transform())
gallery_dataloader = DataLoader(gallery_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# --------------------------------------------------------------------------
def objective(trial: optuna.Trial,
            config: dict,
            dinov3_model: nn.Module,
            processor: AutoImageProcessor,
            train_dataset: CachedCollection,
            val_dataloader: DataLoader,
            gallery_dataloader: DataLoader,
            test_dataloader: DataLoader):
    training_config = config["training"]
    model_config = config["model"]
    sampler_config = config["data"]["sampler"]
    epochs = trial.suggest_int("epochs", training_config["epochs"][0], training_config["epochs"][1])
    lr = trial.suggest_float("lr", training_config["lr"][0], training_config["lr"][1], log=True)
    weight_decay = trial.suggest_float("weight_decay", training_config["weight_decay"][0], training_config["weight_decay"][1], log=True)
    margin = trial.suggest_float("margin", training_config["margin"][0], training_config["margin"][1])
    dropout = trial.suggest_float("dropout", model_config["dropout"][0], model_config["dropout"][1])
    output_dim = trial.suggest_categorical("output_dim", model_config["output_dim"])
    hidden_dim = trial.suggest_categorical("hidden_dim", model_config["hidden_dim"])
    P = trial.suggest_categorical("P", sampler_config["P"])
    K = trial.suggest_categorical("K", sampler_config["K"])
    pksampler = PKSampler(train_dataset, P, K)
    train_dataloader = DataLoader(train_dataset, batch_sampler=pksampler)

    run = wandb.init(project=config["wandb"]["project_name"], entity=config["wandb"]["entity"], config=trial.params, finish_previous=False)

    siamese_model = SiameseDino(dinov3_model, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    for param in siamese_model.base_model.parameters():
        param.requires_grad = False
    _ = siamese_model.to(device)

    scores = train_and_evaluate(run, siamese_model, processor, train_dataloader, val_dataloader, epochs, lr, weight_decay, margin, config["evaluation"]["recall_k"])
    test_scores = evaluate(siamese_model, processor, gallery_dataloader, test_dataloader, recall_k=config["evaluation"]["recall_k"])
    
    run.log({f"test_{k}": v for k, v in test_scores.items()})
    print("Scores on test data:", test_scores)
    run.finish()

    return np.mean([score for score in scores.values()])

study = optuna.create_study(direction="maximize", study_name="siamese_dino_hyperparameter_search")
study.optimize(partial(objective, config=config, dinov3_model=base_model, processor=processor, train_dataset=train_dataset, val_dataloader=val_dataloader, gallery_dataloader=gallery_dataloader, test_dataloader=test_dataloader), n_trials=config["n_trials"])