import torch 
import torchvision 

device = torch.device(type="cuda")

training_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/train")

testing_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/test") 

class CNN(torch.nn.Module): 

    def __init__(self):
        super().__init__() 
        self.nn = torch.nn.Sequential(




        ) 

