from forwardoperator import ForwardOperator
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ForwardOperator().to(device)

criterion = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        x_hat, gan_hat_x, loss = model(inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
    
print('Finished Training')
torch.save(model.state_dict(), 'model.pth')