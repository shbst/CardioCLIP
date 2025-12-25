import numpy as np
import torch
from torchvision import transforms as transforms
from config import CLIPConfig
from cardtransform import BaseLiner

class Numpy2Tensor:
  def __init__(self):
    pass

  def __call__(self, x_numpy):
    x_numpy = x_numpy.astype(np.float32)
    x_tensor = torch.from_numpy(x_numpy).clone()
    return x_tensor

class NumpyNorm:
  def __init__(self, src_mean, src_std, dst_mean=0, dst_std=1):
    self.src_mean, self.src_std, self.dst_mean, self.dst_std = src_mean, src_std, dst_mean, dst_std
  
  def __call__(self, x_numpy):
    return ((x_numpy - self.src_mean) / self.src_std) * self.dst_std + self.dst_mean

  def reverse(self, x_numpy):
    return ((x_numpy - self.dst_mean) / self.dst_std) * self.src_std + self.src_mean

  def __repr__(self):
    return f"Normalizer(src_mean, src_std, dst_mean, dst_std)=>({self.src_mean},{self.src_std},{self.dst_mean},{self.dst_std})"

class XpCardTransforms:
  """
  This class defines transform of ecg and xp-images.
  Args)
    config(CLIPConfig object): config of clip.
    noise(tuple of int): Value of noise added to the ecg.
    randomcrop_scale(float): Minimum cropping scale of random cropping. Setting 1 makes this equivalent to identity function.
    random_rotation(float): Maximum value of random rotation. Setting 0 makes this equivalent to identity function.
    randomerase(float): This specifyes the probability of apllying random erasing. Setting 0 makes this equivalent to identity function.

  Returns)
    Instance function of this class returns the tuple of transforms of ecg and xp images.
  """
  def __init__(self,
    config,
    noise=(0,0),
    randomcropscale=0.85,
    randomrotation=30,
    randomerase=0.5,
    ):
    self.train_cardtransform = transforms.Compose(
      [
        BaseLiner(noise=noise),
        Numpy2Tensor(),
      ]
      )
    self.train_xptransform = transforms.Compose(
      [
        transforms.Resize(config.imsize, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(*config.normalize),
        transforms.ColorJitter(brightness=0.3, contrast=0.5),
        transforms.RandomResizedCrop(size=config.imsize, scale=(randomcropscale, 1.0), antialias=True),
        transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5.)),
        transforms.RandomRotation(degrees=(-randomrotation, randomrotation)), 
        transforms.RandomErasing(p=randomerase),
      ]
      )
    self.valid_cardtransform = transforms.Compose(
      [
        BaseLiner(noise=(0,0)),
        Numpy2Tensor(),
      ]
      )
    self.valid_xptransform = transforms.Compose(
      [
        transforms.Resize(config.imsize),
        transforms.ToTensor(),
        transforms.Normalize(*config.normalize),
      ]
      )

  def get_transform(self, mode='train'):
    if mode=='train':
      return (self.train_cardtransform, self.train_xptransform)
    elif mode=='valid':
      return (self.valid_cardtransform, self.valid_xptransform)
    elif mode=='test':
      return (self.valid_cardtransform, self.valid_xptransform)
    else:
      raise RuntimeError("mode {} for function get_transform is not implemented yet.".format(mode))
    












