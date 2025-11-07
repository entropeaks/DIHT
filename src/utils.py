import torch
from abc import ABC, abstractmethod
from typing import Tuple, Generator, List
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from pathlib import Path
import yaml

def set_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps:0":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Transform(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_transformed(image: Image) -> Image:
        pass

class CropTransform(Transform):
    
    def __init__(self, processor: AutoProcessor, model: AutoModelForZeroShotObjectDetection, device: str, objects: List[str]):
        self.model = model
        self.processor = processor
        self.device = device
        self.objects = objects

    def get_transformed(self, image):
        inputs = self.processor(images=image, text=self.objects, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[(image.height, image.width)]
        )
        result = results[0]

        highest_score_box = result["boxes"][0]
        box = [round(x, 2) for x in highest_score_box.tolist()]

        return image.crop(box)


class Browser:

    def __init__(self, path: Path):
        self.path = path

    def extract_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        paths = []
        labels = []
        for path, label in self._iterate_on_files():
            paths.append(path.as_posix())
            labels.append(int(label))

        return paths, labels
    
    def _iterate_on_files(self) -> Generator[Tuple[Path, str], None, None]:
        for class_dir in self._iterate_on_classes():
            for path in class_dir.iterdir():
                yield path, class_dir.name
        
    def _iterate_on_classes(self) -> Generator[Path, None, None]:
        for class_dir in self.path.iterdir():
            if class_dir.is_dir():
                yield class_dir


    def generate_transformed_dataset(self, destination_path: str, transform: Transform) -> None:
        destinationPath = Path(destination_path)
        destinationPath.mkdir(exist_ok=True)
        for class_dir in self._iterate_on_classes():
            label = class_dir.name
            destinationPath.joinpath(label).mkdir(exist_ok=True)
        
        for path, label in tqdm(self._iterate_on_files()):
            src_img = Image.open(path.as_posix())
            new_img = transform.get_transformed(src_img)
            new_img.save(destinationPath.joinpath(label).joinpath(path.name))