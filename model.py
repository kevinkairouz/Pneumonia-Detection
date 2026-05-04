import torch 
import torchvision 

device = torch.device(type="cuda")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((200,200))])


training_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/train", transform=transform)
testing_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/test",transform=transform)  

print(training_data[0][0].shape)

class CNN(torch.nn.Module): 

    def __init__(self):
        super().__init__() 
        self.nn = torch.nn.Sequential(

            



        ) 

