import logging
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_utils

from PIL import Image

from convolutional_autoencoder import ConvolutionalAutoEncoder


def main():
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available.")

    device = torch.device("cuda")

    model = ConvolutionalAutoEncoder().to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 40
    validation_split = 0.1

    # Load MNIST data
    logging.info("Downloading MNIST dataset.")

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="./datasets/MNIST", train=True, transform=transform, download=True
    )

    dataset_len = len(train_dataset)
    indices = list(range(dataset_len))

    # Randomly splitting indices:
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = torch_utils.SubsetRandomSampler(train_idx)
    validation_sampler = torch_utils.SubsetRandomSampler(validation_idx)


    train_loader = torch_utils.DataLoader(
        train_dataset, batch_size=128, num_workers=4, pin_memory=True, sampler=train_sampler
    )

    validation_loader = torch_utils.DataLoader(
        train_dataset, batch_size=128, num_workers=4, pin_memory=True, sampler=validation_sampler
    )

    data_loaders = {
        'train' : train_loader,
        'val' : validation_loader
    }

    losses = {
        'train' : 0,
        'val' : 0
    }

    # Traning
    logging.info("Starting training phase...")

    for epoch in range(epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = optim.scheduler(optimizer, epoch)
                model.train(True)
            else:
                model.train(False)

            losses[phase] = 0
            loss = 0

            for batch_features, _ in data_loaders[phase]:
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                batch_features = batch_features.to(device)
                
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
                
                # compute reconstructions
                outputs = model(batch_features)
                
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features)
                
                if phase == 'train':
                    # compute accumulated gradients
                    train_loss.backward()
                    
                    # perform parameter update based on current gradients
                    optimizer.step()
                
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            
            # compute the epoch training loss
            losses[phase] = loss / len(data_loaders[phase])
        
        # display the epoch training loss
        logging.info("epoch : {}/{}, train loss = {:.6f}, val loss = {:.6f}".format(epoch + 1, epochs, losses['train'], losses['val']))

    
    torch.save(model.state_dict(), "./checkpoints/parameters.ckpts")

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
