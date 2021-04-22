import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from DataLoader import DirDataset
import matplotlib.pyplot as plt
import numpy as np


# image must be converted to a tensor, hence using pytorch's Compose
# Compose will combine several transforms into one based on the order
compose = transforms.Compose([
   transforms.ToTensor(),
])

if __name__ == '__main__':

    # checking to run on GPU if it's available, if not on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # please copy these from data folder into the project if not present
    train_data = f'./train'
    target_data = f'./train_blur_masks'

    dataset = DirDataset(train_data, target_data)
    val_sample_size = int(len(dataset) * 0.1)
    train_sample_size = len(dataset) - val_sample_size

    test_sample_size = int(train_sample_size * 0.2)
    train_ds, val_ds = random_split(dataset, [train_sample_size, val_sample_size])

    train_sample_size = train_sample_size - test_sample_size

    train_ds, test_ds = random_split(train_ds, [train_sample_size, test_sample_size])

    train_loader = DataLoader(train_ds, batch_size=1, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=True, shuffle=False)

    learning_rate = 1e-4
    wt_decay = 1e-4


    # Extending Pytorch's nn module to implement neural network models (using predefined methods functions etc..)
    class FILTER_MODEL(nn.Module):
        def __init__(self):
            # Referred from https://debuggercafe.com/image-deblurring-using-convolutional-neural-networks-and-deep-learning/
            nn.Module.__init__(self)
            self.conv_1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
            self.conv_2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
            self.conv_3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        def forward(self, x):
            x = F.relu(self.conv_1(x))
            x = F.relu(self.conv_2(x))
            x = self.conv_3(x)
            return x


    model = FILTER_MODEL().to(device)
    print("filter model has been defined")

    # Optmizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # Refrred from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    criterion = nn.MSELoss()

    train_sample_count = len(train_ds)
    val_sample_count = len(val_ds)
    test_sample_count = len(test_ds)

    num_epochs = 1

    for epoch in range(num_epochs):
        # training on training dataset
        # model.train()
        training_loss = 0.0
        # Referred from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i, (images, labels) in enumerate(train_loader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            print(f'epoch:{epoch + 1}/{num_epochs}, trining_loss:{training_loss / (i + 1):.4f}')

        # Referred from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        train_loss = training_loss / train_sample_count
        print(f"Training Loss: {train_loss:.4f}")


    val_loss = 0.0
    with torch.no_grad():
        # Referred from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i, (images, labels) in enumerate(val_loader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        val_loss = val_loss / val_sample_count
        print(f"Validation Loss: {val_loss:.4f}")

    test_loss = 0.0
    with torch.no_grad():
        # Referred from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i, (images, labels) in enumerate(test_loader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        test_loss = test_loss / test_sample_count
        print(f"Testing Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("MODEL IS SAVED")

    with torch.no_grad():
        # image_242 Feeding the model from the device
        data = plt.imread('image_114.jpg')
        # converting the image to a tensor
        image = compose(data)

        # adding a dimension i.e a batch size to feed the model
        image = image.unsqueeze(0)
        # Feeding a test sample to the model to predict the blurred image
        # dataiter = iter(test_loader)
        # data = dataiter.next()
        # image, labels = data

        # converting back to cpu tensor to convert it into numpy array
        output = model(image.to(device)).cpu()

        # Referred from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        np_output = output / 2 + 0.5  # unnormalize
        np_img = np_output.numpy()

        # Referred from: https://www.w3resource.com/numpy/manipulation/squeeze.php#:~:text=The%20squeeze()%20function%20is,the%20shape%20of%20an%20array.&text=Input%20data.&text=Selects%20a%20subset%20of%20the,one%2C%20an%20error%20is%20raised.
        np_img = np.squeeze(np_img, axis=0)
        # 0 = channels, 1 = width, 2 = height values of the numpy image
        # Referred from: https://discuss.pytorch.org/t/trying-to-understand-torch-transpose/5656
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

