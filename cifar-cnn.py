import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1).to(device)
        self.act1 = nn.ReLU().to(device)
        self.drop1 = nn.Dropout(0.3).to(device)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1).to(device)
        self.act2 = nn.ReLU().to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)).to(device)

        self.flat = nn.Flatten().to(device)

        self.fc3 = nn.Linear(8192, 512).to(device)
        self.act3 = nn.ReLU().to(device)
        self.drop3 = nn.Dropout(0.5).to(device)

        self.fc4 = nn.Linear(512, 10).to(device)

    def forward(self, x):
        # input 3, 32, 32
        # output 32, 32, 32
        x = x.to(device)
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32, 32, 32
        # output 32, 32, 32
        x = self.act2(self.conv2(x))
        # input 32, 32, 32
        # output 32, 16, 16
        x = self.pool2(x)
        # input 32, 16, 16
        # output 8192
        x = self.flat(x)
        # input 8192
        # output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512
        # output 10
        x = self.fc4(x)
        return x


model = CIFAR10Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 20
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
        y_pred = model(inputs.to(device))
        loss = loss_fn(y_pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs.to(device))
        acc += (torch.argmax(y_pred, 1) == labels.to(device)).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch: %d model accuracy: %.2f%%" % (epoch, acc*100))

torch.save(model.state_dict(), "cifar10_model.pth")
