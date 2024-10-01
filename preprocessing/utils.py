import torch
import os


def load_graphs(path,mode="TRAIN",pk=None):
    
    assert not pk == None
    
    datas = []
    for file in os.listdir(path):
        split = file.split("_")[-2]    
        if split == mode:
            data = torch.load(path+file)
            data[pk].y = data[pk].y.reshape(-1).long()
            datas.append(data)
            
    return datas