import torch 
import torch.utils.data as data
import numpy as np
import os
import pandas as pd
from PIL import Image

class ImagewoofDataset(data.Dataset):
    
    """_summary_
    """

    def __init__(self, 
                 root, 
                 anno_file, 
                 transforms=None,
                 is_train=True,
                 ):
        self.root = root
        self.transforms = transforms
        self.is_train = is_train
        
        print("Reading annotation file: " + anno_file)
        self.anno = pd.read_csv(os.path.join(self.root, anno_file))
        
    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.anno.iloc[index]["path"])
        label = self.anno.iloc[index]["noisy_labels_0"]
        label_1 = self.anno.iloc[index]["noisy_labels_1"]
        label_5 = self.anno.iloc[index]["noisy_labels_5"]
        
        image = Image.open(file_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transforms(image)
        
        return image, label, label_1, label_5

    def __len__(self):
        return len(self.anno)


if __name__ == "__main__":
    test_data = ImagewoofDataset("./datasets", "noisy_imagewoof.csv", is_train=False)
    print(len(test_data))
    image, _, _ , _ = test_data[150]
    
    image.show()