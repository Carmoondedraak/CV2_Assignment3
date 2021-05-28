import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio
# import matplotlib.pyplot as plt
import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils
from vgg import VGG
from discriminators_pix2pix import MultiscaleDiscriminator
from utils import loadModels
from PIL import Image
import PIL

# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender'
data_root = '../Data/data_set/'
train_list = 'train.str'
test_list = 'test.str'
batch_size = 8
nthreads = 4
max_epochs = 20
displayIter = 20
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

done = u'\u2713'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
# Define your generators and Discriminators here
G = res_unet.MultiScaleResUNet(in_nc =7).to(device)
# netG.apply(weights_init)
# print(netG)
D = MultiscaleDiscriminator().to(device)

print('done')

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
    # use the .to(device) function, where device is automatically detected
    # above.
checkpoint = torch.load('../Data/Pretrained_model/checkpoint_G.pth')
# netG_l =(loadModels(netG, '../Data/Pretrained_model/checkpoint_G.pth')).to(device)
G.load_state_dict(checkpoint['model'],strict=False)
G.to(device)

checkpoint = torch.load('../Data/Pretrained_model/checkpoint_D.pth')
# netD_l = loadModels(netD, '../Data/Pretrained_model/checkpoint_D.pth').to(device)
D.load_state_dict(checkpoint['model'],strict=False)
D.to(device)

# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


optimizerG=torch.optim.SGD(G.parameters(),lr = 0.0004, momentum=0.9,weight_decay=1e-4)
schedulerG=torch.optim.lr_scheduler.StepLR(optimizerG,step_size=30,gamma=0.1)
optimizerD=torch.optim.SGD(D.parameters(), lr =0.0004,momentum=0.9,weight_decay=1e-4)
schedulerG=torch.optim.lr_scheduler.StepLR(optimizerD,step_size=30,gamma=0.1)
# .train()
print('done')

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
criterion_pixelwise = nn.L1Loss()
criterion_id = vgg_loss.VGGLoss()
criterion_GAN = gan_loss.GANLoss()
criterion_DIS = gan_loss.GANLoss()
print('done')

print('[I] STATUS: Initiate Dataloaders...',flush=True)
# Initialize your datasets here
trainSet = SwappedDatasetLoader(train_list, data_root)
print('[+] Create workers')
trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)

testSet = SwappedDatasetLoader(test_list, data_root)
print('[+] Create workers')
testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
print('done')

print('[I] STATUS: Initiate Logs...', end='',flush=True)
trainLogger = open(logTrain, 'w')
print('done')


def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1 - mask)


def blend_imgs_bgr(source_img, target_img, mask):
    # Implement poisson blending here. You can us the built-in seamlessclone
    # function in opencv which is an implementation of Poisson Blending.
    # mask= mask.cpu()
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return target_img

    center = (np.min(a[1]) + np.max(a[1])) // 2, (np.min(a[0]) + np.max(a[0])) // 2
    output = cv2.seamlessClone(source_img, target_img, mask*255, center, cv2.NORMAL_CLONE)

    return output


def blend_imgs(source_tensor, target_tensor, mask_tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)


def Train(G, D, epoch_count, iter_count):
    G.train(True)
    D.train(True)
    epoch_count += 1
    pbar = tqdm(enumerate(trainloader), total=iter_count, leave=False)

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        images, _ = data

        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        # source,target,swap,mask = data
        source = images['source'].to(device)
        target = images['target'].to(device)
        swap = images['swap'].to(device)
        mask = images['mask'].to(device)
        ground_truth = blend_imgs(source,target,mask)
        # poisson = blend_imgs_bgr(source, target, mask)
        # imageio.imwrite(visuals_loc + '/im.png', poisson)

        input = torch.cat((swap,target,mask),dim=1)
        print(input.shape)

        # 2) Feed the data to the networks.
        output = G.forward(input)
        # if np.random.randint(0,1) > 0.5:
        #     input_d = target
        #     label = 1
        # else:
        #     input_d = output
        #     label = 0
        output_d = D.forward(output)
        # 4) Calculate the losses.
        print(output.shape)
        print(ground_truth.shape)
        # 5) Perform backward calculation.
        loss_pixelwise = criterion_pixelwise(output.to(device), ground_truth.to(device))
        loss_id = criterion_id(output.to(device), ground_truth.to(device))
        loss_GAN = criterion_GAN(output_d,True).detach()
        # loss_DIS = criterion_DIS(target.to(device),True)
        loss_rec = 0.1 * loss_pixelwise + 0.5
        loss_G_total = 1.0* loss_rec + 0.001 * loss_GAN
        # loss_pixelwise.backward()
        # loss_id.backward()
        # loss_GAN.backward()
        # loss_DIS.backward()
        optimizerG.zero_grad()
        loss_G_total.backward(retain_graph=True)
        optimizerG.step()
        # 6) Perform the optimizer step.
        optimizerD.zero_grad()
        loss_id.backward()
        optimizerD.step()


        if iter_count % displayIter == 0:
            # Write to the log file.
            trainLogger.write()

            trainLogger.update('losses', pixelwise=loss_pixelwise, id=loss_id,rec=loss_rec,g_gan=loss_GAN, tg_gan =loss_G_total, d_gan=loss_DIS)


        # Print out the losses here. Tqdm uses that to automatically print it
        # in front of the progress bar.
        print(loss_G_total,flush=True)
        pbar.set_description()
            # print('dit?')


    # Save output of the network at the end of each epoch. The Generator

    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    return np.nanmean(total_loss_pix),\
        np.nanmean(total_loss_id), \
        np.nanmean(total_loss_rec), np.nanmean(total_loss_G_Gan),\
        np.nanmean(total_loss_D_Gan), iter_count


def Test(G):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, (images) = next(t)

        # Feed the network with images from test set

        # Blend images
        pred = G(img_transfer_input)
        # You want to return 4 components:
            # 1) The source face.
            # 2) The 3D reconsturction.
            # 3) The target face.
            # 4) The prediction from the generator.
            # 5) The GT Blend that the network is targettting.


iter_count = 0
# Print out the experiment configurations. You can also save these to a file if
# you want them to be persistent.
print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tSaving frequency (per epoch): ', saveIter)
print('\tModels Dumped at: ', check_point_loc)
print('\tVisuals Dumped at: ', visuals_loc)
print('\tExperiment Name: ', experiment_name)

for i in range(max_epochs):
    loss = Train(G,D,100, 50)
    print(loss)
    lr_scheduler.step()
    # Call the Train function here
    # Step through the schedulers if using them.
    # You can also print out the losses of the network here to keep track of
    # epoch wise loss.

trainLogger.close()
