from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch import config
from pyimagesearch.model import UNet

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import time
import os

imagepaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH))) 
maskpaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

split = train_test_split(imagepaths, maskpaths, test_size=config.TEST_SPLIT, random_state=10)

(trainImages, testImages) = split[:2] 
(trainMasks, testMasks) = split[2:]

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
                                 transforms.ToTensor()])

# create the train and test datasets
trainDS = SegmentationDataset(imagepaths=trainImages, maskpaths=trainMasks, transforms=transforms)
testDS = SegmentationDataset(imagepaths=testImages, maskpaths=testMasks, transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

trainLoader = DataLoader(trainDS, 
                         shuffle=True,
                         batch_size=config.BATCH_SIZE,
                         pin_memory=config.PIN_MEMORY)

testLoader = DataLoader(testDS, 
                         shuffle=False,
                         batch_size=config.BATCH_SIZE,
                         pin_memory=config.PIN_MEMORY)

#Sanity Check
# for test_images, test_labels in testLoader:  
#     sample_image = test_images   # Reshape them according to your needs.
#     sample_label = test_labels
#     print(len(sample_image))
#     break

unet = UNet().to(config.DEVICE)


# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opti = Adam(unet.parameters(), lr = config.INIT_LR)


# initialize loss function and optimizer
trainsteps = len(trainDS)//config.BATCH_SIZE
teststeps = len(testDS)//config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train loss": [], "test_loss":[]}

#Training the model
print("[INFO] training the network...")
startTime = time.time()

for e in tqdm(range(config.NUM_EPOCHS)):
    unet.train()

    totaltrainloss = 0
    totaltestloss = 0

    for i, (x,y) in tqdm(enumerate(trainLoader)):
        (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        pred = unet(x)
        loss = lossFunc(pred, y)

        opti.zero_grad()
        loss.backward()
        opti.step()

        totaltrainloss += loss
    
    with torch.no_grad():

        unet.eval()

        for(x,y) in testLoader:
            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = unet(x)
            totaltestloss += lossFunc(pred, y)

    # calculate the average training and validation loss
    avgTrainLoss = totaltrainloss / trainsteps
    avgTestLoss = totaltestloss / teststeps

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))