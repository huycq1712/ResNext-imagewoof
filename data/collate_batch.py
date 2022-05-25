from cProfile import label
import torch

def collate_fn(batch):
    batch = list(zip(*batch))
    images = torch.stack(batch[0])
    labels = batch[1]
    labels_1 = batch[2]
    labels_5 = batch[3]
    
    return images, labels, labels_1, labels_5
