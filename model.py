import torch 
import torchvision 

# device = torch.device(type="cuda")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((200,200))])


training_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/train", transform=transform)
testing_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/test",transform=transform)  

print(training_data[0][0].shape) 

dummy = torch.randn([3,200,200])

class CNN(torch.nn.Module): 

    def __init__(self):
        super().__init__() 
        self.nn = torch.nn.Sequential( 
            torch.nn.Conv2d(3, 30, 3), 
            torch.nn.Sigmoid(), 
            torch.nn.Conv2d(30, 100, 3), 
            torch.nn.MaxPool2d(2,2), 
            torch.nn.Flatten(), 
            torch.nn.Linear(960400, 10000), 
            torch.nn.ReLU(), 
            torch.nn.Linear(10000, 1000), 
            torch.nn.ReLU(), 
            torch.nn.Linear(1000, 100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 2)
        ) 
    
    def forward(self, data): 
        return self.nn(data) 
    

cnn = CNN() 

print(cnn.nn(dummy).shape) 
print(100 * 9604) 

#use argsmax in this because it is binary classification 
# use sigmoid function at the end 
# 2 classes phemounia or normal 

