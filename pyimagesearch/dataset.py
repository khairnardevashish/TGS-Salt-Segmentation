from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
   def __init__(self, imagepaths, maskpaths, transforms):
      self.imagepaths = imagepaths
      self.maskpaths = maskpaths
      self.transforms = transforms

   def __len__(self):
      return len(self.imagepaths)
   
   def __getitem__(self, index):
      imagepath = self.imagepaths[index]
      
      image = cv2.imread(imagepath)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(self.maskpaths[index], 0)

      if self.transforms:
         image = self.transforms(image)
         mask = self.transforms(mask)

      return (image, mask)

    