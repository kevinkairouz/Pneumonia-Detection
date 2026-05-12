import torch 
import torchvision 
import random 




device = torch.device(type="cuda")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((200,200))])


training_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/train", transform=transform)
testing_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/test",transform=transform)  

training_dataLoader = torch.utils.data.DataLoader(training_data) 
testing_dataLoader = torch.utils.data.DataLoader(testing_data)


# print(training_data[0][0].shape) 

# dummy = torch.randn([3,200,200])

class CNN(torch.nn.Module): 

    def __init__(self):
        super().__init__() 
        self.nn = torch.nn.Sequential( 
            torch.nn.Conv2d(3, 30, 3), 
            torch.nn.ReLU(), 
            torch.nn.Conv2d(30, 100, 3), 
            torch.nn.MaxPool2d(2,2), 
            torch.nn.Flatten(), 
            torch.nn.Linear(960400, 10000), 
            torch.nn.ReLU(), 
            torch.nn.Linear(10000, 1000), 
            torch.nn.ReLU(), 
            torch.nn.Linear(10, 2), 
        ) 
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.001) 
        self.loss_func = torch.nn.CrossEntropyLoss() 
        
    def fit(self, trainingData):  
        self.nn.train() 
        num_epochs = 10 
        for epoch in range(0, num_epochs): 
            for xbatch, ybatch in trainingData: 
                self.optimizer.zero_grad() 
                prediction = self.nn(xbatch) 
                loss = self.loss_func(prediction, ybatch) 
                loss.backward()
                self.optimizer.step()
                print(f"loss {loss}, epoch {epoch}")
        print("Training Finished")

    def predict(self, data): 
        self.nn.eval() 
        with torch.no_grad():
            pred = self.nn(data) 
            pos = torch.argmax(pred) 
            return pred[pos]

    def forward(self, data): 
        return self.nn(data) 


    def printRandomNumberInRange(self, firstNum, secondNum): #test function

        return random.randint(firstNum, secondNum)
    

cnn = CNN()  
# print(torch.cuda.is_available()) 
cnn.fit(training_dataLoader)



#use argsmax in this because it is binary classification 
# use sigmoid function at the end 
# 2 classes phemounia or normal 

