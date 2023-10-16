import torch
import torch.nn as nn


################################# loss functions ##########################################################

def denoise_loss_mse(denoise, clean):      
  criterion = nn.MSELoss()
  loss = criterion(denoise, clean)
  return torch.mean(torch.mean(loss))

