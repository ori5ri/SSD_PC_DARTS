import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#from model import SSD300, MultiBoxLoss
#from model_darts import SSD300, MultiBoxLoss
from model_darts_search import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils_search import *

from architect_search_ssd import Architect
import logging
import sys
import numpy as np
import torch.nn.functional as F

# Data parameters
data_folder = './data_search'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = './weights_search/checkpoint_ssd300_search.pth.tar'  # path to model checkpoint, None if none
batch_size = 2  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

arch_learning_rate = 6e-4
arch_weight_decay = 1e-3
train_portion = 0.5 # portion of training data

# LOG
log_folder = './search'
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_folder, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
seed =2

cudnn.benchmark = True

def main():
    """
    Training.
    """
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(device)

    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler= torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        collate_fn=train_dataset.collate_fn,
        pin_memory=True, num_workers=workers
    )

    valid_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler= torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        collate_fn=train_dataset.collate_fn,
        pin_memory=True, num_workers=workers
    )


    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    #epochs = iterations // (len(train_dataset) // 32)
    epochs = 20
    print(epochs)
    decay_lr_at = [5,12]
    print(decay_lr_at)
    #decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
    

    architect = Architect(model, criterion, momentum, weight_decay, arch_learning_rate, arch_weight_decay)
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        genotype = model.base.genotype()
        logging.info('genotype = %s', genotype)

        #print(F.softmax(model.base.alphas_normal, dim=-1))
        #print(F.softmax(model.base.alphas_reduce, dim=-1))
        
        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_queue=train_queue,
              valid_queue= valid_queue,
              model=model,
              architect = architect,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_queue):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # get a random minibatch from the search queue with replacement
        #images_search, boxes_search, labels_search, _= next(iter(valid_queue))
        try:
            images_search, boxes_search, labels_search, _ = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            images_search, boxes_search, labels_search, _ = next(valid_queue_iter)
        
        images_search = images_search.to(device)
        boxes_search = [b.to(device) for b in boxes_search]
        labels_search = [l.to(device) for l in labels_search]

        # if epoch >=3:
        #     print("architect step")
        #     architect.step(images_search, boxes_search, labels_search)
        architect.step(images_search, boxes_search, labels_search)
        
        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_queue),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
