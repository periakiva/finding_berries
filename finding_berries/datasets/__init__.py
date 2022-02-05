from .cranberry_dataset import CBDatasetSemanticSeg
import torchvision
from torch.utils.data import DataLoader

datasets = {
            'craid': CBDatasetSemanticSeg,
            }

def build_dataset(dataset_name: str, root: str, batch_size: int,
                  num_workers: int, split: str, transforms: object, **kwargs) -> DataLoader:
    
    print(f"Building {split} dataset {dataset_name} with root: {root}")
    dataset = datasets[dataset_name](root=root, transforms=transforms, split=split, **kwargs)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True)

    print(f"The dataset has length of {len(dataloader)}")
    return dataloader
    