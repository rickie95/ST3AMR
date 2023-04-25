import logging
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from convolutional_autoencoder import ConvolutionalAutoEncoder


def main():
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available.")

    device = torch.device("cuda")

    model = ConvolutionalAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 40 

    # Load MNIST data
    logging.info("Downloading MNIST dataset.")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="./datasets/MNIST", train=True, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    # Traning
    logging.info("Starting training phase...")

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        logging.info("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    
    torch.save(model.state_dict(), "./parameters.ckpts")

    # Test the encoder with some example
    logging.info("Testing.....")
    
    test_dataset = torchvision.datasets.MNIST(
        root="./datasets/MNIST", train=False, transform=transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False, num_workers=4
    )
    
    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.to(device)
            reconstruction = model(test_examples)
            break

    logging.info("Saving results...")

    for index, r in enumerate(reconstruction):

        im = Image.fromarray(test_examples[index].to("cpu").numpy().reshape(28, 28)*255)
        path = "./results/original_{}.jpg".format(index)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(path)

        im = Image.fromarray(reconstruction[index].to("cpu").numpy().reshape(28, 28)*255)
        path = "./results/reconstructed_{}.jpg".format(index)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(path)


    logging.info("Done.")

if __name__ == "__main__":
    main()
