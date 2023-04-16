import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class Architect(object):

  def __init__(self, model, criterion, momentum, weight_decay, arch_learning_rate, arch_weight_decay):
    self.network_momentum = momentum
    self.network_weight_decay = weight_decay
    self.model = model
    self.criterion = criterion # MultiBoxLoss
    self.optimizer = torch.optim.Adam(model.base.arch_parameters(), lr=arch_learning_rate, betas=(0.5, 0.999), weight_decay=arch_weight_decay)


  def step(self, images_search, boxes_search, labels_search):
    self.optimizer.zero_grad()
    self._backward_step(images_search, boxes_search, labels_search)
    self.optimizer.step()

  def _backward_step(self, images_search, boxes_search, labels_search):
    # loss backward
    #loss = self.model._loss(input_valid, target_valid)
    predicted_locs, predicted_scores = self.model(images_search)
    loss = self.criterion(predicted_locs, predicted_scores, boxes_search, labels_search)
    
    loss.backward()


