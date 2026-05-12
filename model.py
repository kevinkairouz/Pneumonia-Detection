import torch 
import torchvision 
import random 




device = torch.device(type="cuda") 
# torch.Tensor.to()
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((200,200)), torchvision.transforms.ToTensor()])


training_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/train", transform=transform)
testing_data = torchvision.datasets.ImageFolder(root="archive/chest_xray/test",transform=transform)  

training_dataLoader = torch.utils.data.DataLoader(training_data,batch_size=32, shuffle = True)
testing_dataLoader = torch.utils.data.DataLoader(testing_data, shuffle=True)

print(training_data[0][0].shape) 

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
            torch.nn.Linear(960400, 1000),
            torch.nn.ReLU(), 
            torch.nn.Linear(1000, 10),
            torch.nn.ReLU(), 
            torch.nn.Linear(10, 2), 
        ) 
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.001) 
        self.loss_func = torch.nn.CrossEntropyLoss() 
        
    def fit(self, trainingData): 
        self.nn.to(device)
        self.nn.train() 
        num_epochs = 10 
        for epoch in range(0, num_epochs): 
            for xbatch, ybatch in trainingData:  
                xbatch, ybatch = xbatch.to(device), ybatch.to(device) 
                self.optimizer.zero_grad() 
                prediction = self.nn(xbatch) 
                # print(f"prediction {prediction}, label {ybatch}")
                loss = self.loss_func(prediction, ybatch) 
                loss.backward()
                self.optimizer.step()
                print(f"loss {loss}, epoch {epoch}")
        print("Training Finished") 

    def score(self, testingData): 
        attempts = 0 
        numCorrect = 0 

        for x, y in testingData: 
            with torch.no_grad(): 
                attempts += 1 
                x = x.to(device) 
                y = y.to(device) 
                pred = self.nn(x) 
                classification = torch.argmax(pred) 
                if classification == y.item(): 
                    numCorrect += 1     
        return (numCorrect/attempts) 


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
cnn.score(testing_dataLoader)

