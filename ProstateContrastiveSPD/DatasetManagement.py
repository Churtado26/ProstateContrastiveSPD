from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
from torchvision import transforms
import torchvision

class DatasetManagement(Dataset):
    def __init__(self, X, y, data_augmentation = 'baseline'):     
        self._X_ = [X[:,i,...] for i in range(3)]
        self._y_ = y
        self._data_augmentation_ = data_augmentation
        
        
    def __transform__ (self, x):
        if self._data_augmentation_ == 'baseline':
            random_h_flip = random.random()
            random_v_flip = random.random()
            degree, translate, scale, shears = transforms.RandomAffine.get_params(degrees = (0,0), 
                                       translate=(0.3, 0.3), 
                                       img_size = x[0].shape, 
                                       scale_ranges = None, 
                                       shears = None )
            for idx in range (len(x) - 1):
                if random_h_flip > 0.5:
                    x[idx] = torchvision.transforms.functional.hflip(x[idx])

                if random_v_flip > 0.5:
                    x[idx] = torchvision.transforms.functional.vflip(x[idx])



                x[idx] = torchvision.transforms.functional.affine(img=x[idx],
                                                 angle = degree,
                                                 translate = translate,
                                                 scale = 1.,
                                                 shear = 0)
                
                #print(x[idx].shape)
                #Normalize the images
                mean = x[idx].mean()
                std = x[idx].std()
                std = std.add(1e-10) # we add a small error term just in case that std is 0.
                transform_norm = transforms.Compose([
                    transforms.Normalize(mean, std)
                ])

                # get normalized image
                x[idx] = transform_norm(x[idx])
            return x

        elif self._data_augmentation_ == 'nothing':
            for idx in range (len(x) - 1):
                #Normalize the images
                mean = x[idx].mean()
                std = x[idx].std()
                std = std.add(1e-10) # we add a small error term just in case that std is 0.
                transform_norm = transforms.Compose([
                    transforms.Normalize(mean, std)
                ])

                # get normalized image
                x[idx] = transform_norm(x[idx])
        return x

    def __len__(self):
        return len(self._y_)

    def __getitem__(self, idx):
        #print("ENTRÓ A GET ITEM")
        X = []
        for modality in range(len(self._X_)-1):
            current_image = np.array(self._X_[modality][idx], dtype =  np.int16)
            current_image = torch.tensor(current_image, dtype = torch.float32)
            X.append(
                current_image
            ) 
        
        X.append(torch.tensor(self._X_[-1][idx], dtype = torch.float32))
        X = self.__transform__(X)
        y = torch.tensor(self._y_[idx], dtype = torch.float32)
        y = torch.unsqueeze(y, dim = -1)
        return X, y