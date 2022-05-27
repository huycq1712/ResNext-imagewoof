from torch.utils import data

from .datasets.ImagewoofDataset import ImageWoofDataset
from .transforms import build_transforms


def build_dataset(transforms, 
                  dataset_root, 
                  annotation_file,
                  is_train=True):
    dataset = ImageWoofDataset(root=dataset_root,
                               anno_file=annotation_file, 
                               transforms=transforms,  
                               is_train=is_train)
    
    return dataset


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
