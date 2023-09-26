import torch
from torch.utils import data
import time

from torch import nn
import torch.nn.functional as F
from scipy import misc

from dataset import CamvidDataSet
from segmentationModel import SegmentationModel

# The datafolder must be downloaed
# The path to data folder must be correct
# confusion matrix
# TODO - solve the cuda run time issue


is_cuda = torch.cuda.is_available()
net = SegmentationModel()
if is_cuda:
    net.cuda()
net.train()

# Training
#path = '/home/sherin/mypro/HODLWP/ThePyTorchBookDataSet/camvid'
path ='D:\YoLo\Pytorch\9781788834131_Code\9781788834131_Code\4.ComputerVision\SemSeg\camvid'
epochs = 64
bsize = 8
dataset = CamvidDataSet('train', path)
loader = data.DataLoader(dataset, batch_size=bsize, num_workers=4, shuffle=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
loss_fn = nn.NLLLoss()


def create_image(out):
    """ Creating image from the outbatch """
    img = out[0].max(0)[1].data.cpu().numpy()
    misc.imsave('{}.png'.format(time.time()), img)


def save_model(model):
    torch.save(model.state_dict(), '{}.pth'.format(time.time()))


for epoch in range(epochs):
    for in_batch, target_batch in loader:
        if is_cuda:
            in_batch, target_batch = in_batch.cuda(), target_batch.cuda()
        optimizer.zero_grad()
        out = net(in_batch)
        loss = loss_fn(F.log_softmax(out, 1), target_batch)
        loss.backward()
        optimizer.step()
    print('Training Loss: {:.5f}, Epochs: {:3d}'.format(loss.item(), epoch))
    if epoch % 50 == 0:
        net.eval()
        test_dataset = CamvidDataSet('test', path)
        test_loader = data.DataLoader(
            test_dataset, batch_size=bsize, num_workers=4, shuffle=True)
        loss = 0
        counter = 0
        for in_batch, target_batch in test_loader:
            if is_cuda:
                in_batch, target_batch = in_batch.cuda(), target_batch.cuda()
            with torch.no_grad():
                out = net(in_batch)
                counter += 1
                loss += loss_fn(F.log_softmax(out, 1), target_batch).item()
        loss = loss / counter
        print(' ========== Testing Loss: {:.5f} =========='.format(loss, epoch))
        create_image(out)
        save_model(net)
        net.train()
