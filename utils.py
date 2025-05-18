import os
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from args_fusion import args
# from scipy.misc import imread, imsave, imresize
from imageio import imread, imsave
import imageio
from skimage.transform import resize as imresize
import matplotlib as mpl
from torch.autograd import Variable
from os import listdir
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def tensor_save_rgb_channels(tensor, filename, cuda=True):
    # Ensure range [0, 1]
    if cuda:
        img = tensor.detach().cpu().clamp(0, 1).numpy()  
    else:
        img = tensor.clamp(0, 1).numpy()  

#     # Split channels
#     red = img[0,:,:]
#     green = img[1,:,:]
#     blue = img[2,:,:]

    # Save each channel as separate image
    Image.fromarray((img).transpose(1, 2, 0).astype('uint8')).save(filename + '_red.png')
#     Image.fromarray((green * 255).astype('uint8')).save(filename + '_green.png')
#     Image.fromarray((blue * 255).astype('uint8')).save(filename + '_blue.png')

def tensor_save_grayimage(tensor, filename,cuda=True):
    if cuda:
        img = tensor.detach().cpu().clamp(0,1).numpy()
    else:
        img = tensor.clamp(0,1).numpy()
    img = (img*255).astype("uint8")
    if img.ndim == 3:
        img = img.squeeze(0)
    img = Image.fromarray(img, 'L')
    img.save(filename)

def tensor_save_rgbimage_t(tensor, filename, cuda=True):
    # print(tensor.size())
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.detach().cpu().clamp(0, 255).numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.squeeze(0).transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def tensor_save_rgbimage(tensor, filename, cuda=True):
    # print(tensor.size())
    if cuda:
        img = tensor.detach().cpu().clamp(0, 1).numpy()  # Ensure range [0, 1]
    else:
        img = tensor.clamp(0, 1).numpy()  # Ensure range [0, 1]
    img = (img * 255).transpose(1, 2, 0).astype('uint8')  # Ensure range [0, 255]
#     print(img)
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=True):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    # random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

"""
def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        image = imread(path, pilmode='RGB')
    else:
        image = imread(path, pilmode='L')

    if height is not None and width is not None:
        image = imresize(image, [height, width], order=0)
    return image
"""
def get_test_images_auto(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

"""
def get_image(path, height=None, width=None, flag=False):
    if flag is True:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image
"""
import numpy as np
from skimage.transform import resize
from PIL import Image

def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = Image.open(path).convert(mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    image = np.array(image)
    if height is not None and width is not None:
        image = resize(image, (height, width), mode='reflect', anti_aliasing=True)
    return image


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            #print(image.shape)
            image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        else:
            image = np.transpose(image, (2, 0, 1))
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
#     print(images)
    return images


def get_test_images_auto(paths, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
        images.append(image)

    images = torch.from_numpy(np.concatenate(images, axis=0)).float()
    return images
"""
def get_train_images_auto(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
"""
# load images - test phase
def get_test_image(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = imread(path, pilmode='L')
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        base_size = 512
        h = image.shape[0]
        w = image.shape[1]
        c = 1
        if h > base_size or w > base_size:
            c = 4
            images = get_img_parts(image, h, w)
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()

    # images = np.stack(images, axis=0)
    # images = torch.from_numpy(images).float()
    return images # , h, w, c


def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).cuda()
        count = torch.zeros(1, 1, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list

"""

def save_image(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        # print(img_fusion.data.size())
        # print('img_fusion.data.numpy():', img_fusion.data[0].size())
        
        img_fusion = img_fusion.cpu().data.numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)
"""
def save_image(img_tensor, output_path):
    img_np = img_tensor.detach().cpu().numpy()  # Detach tensor and convert to numpy
    img_np = np.transpose(img_np, (1, 2, 0))  # Swap channel dimension
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # Clip values and convert to uint8
    cv2.imwrite(output_path, img_np)

def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images_ir = []
    images_vi = []
    for path in paths:
        image = get_image(path, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/ir_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_ir.append(image)

        path_vi = path.replace('lwir', 'visible')
        image = get_image(path_vi, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/vi_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_vi.append(image)

    images_ir = np.stack(images_ir, axis=0)
    images_ir = torch.from_numpy(images_ir).float()

    images_vi = np.stack(images_vi, axis=0)
    images_vi = torch.from_numpy(images_vi).float()
    return images_ir, images_vi



def visualize_images(image1, image2, title1="IR", title2="Visible"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.show()


# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)

"""
def gradient(input):
    filter = torch.reshape(torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),[1, 1, 3, 3]).cuda()
    d = F.conv2d(input, filter, stride=1, padding=1)
    #print(d)
    return d
"""
def gradient(input):
    # Determine the input shape (channel, height, width)
    channel, height, width = input.shape
    
    # Reshape input as a single-channel input
    input = torch.reshape(input, [1, 1, height, width]).cuda()
    
    # Define the filter：采用sobel算子，在中心像素的垂直和水平两个方向上计算图像的变化梯度，更加稳定和有效
    filter = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]).reshape([1, 1, 3, 3]).cuda()
    
    # Apply convolution to calculate gradient
    d = F.conv2d(input, filter, stride=1, padding=1)
    
    return d  # .reshape([channel, height, width])

"""
def grad(img):
    channel, height, width = img.shape
    
    img = torch.reshape(img, [1, 1, height, width]).cuda()
    kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]).reshape([1, 1, 3, 3]).cuda()
    g = F.conv2d(img, kernel, stride=1, padding=1)
    return g
"""
def grad(img):
    channel, height, width = img.shape
    
    # Ensure that the image is on the GPU and has the correct shape
    img = torch.reshape(img, [1, channel, height, width]).cuda()
    
    # Create a kernel for each channel
    single_channel_kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    
    # Repeat the kernel for each channel in the image
    multi_channel_kernel = single_channel_kernel.repeat(channel, 1, 1, 1).cuda()
    
    # Perform the convolution operation
    g = F.conv2d(img, multi_channel_kernel, stride=1, padding=1, groups=channel)
    
    return g


def calculate_gradient_penalty(real_images, fake_images, D):
    # eta = torch.FloatTensor(args.batch_size,1,1,1).uniform_(0,1)
    # eta = eta.expand(args.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = torch.FloatTensor(1,1,1).uniform_(0,1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2))
    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)  # 随机权重平均
    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambda_term
    return grad_penalty


#####################Class#####################
class GANLoss(torch.nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        #self.eps = 1e-6
        self.eps = 0

    def forward(self, x ):

        #b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2) + self.eps,0.5)


        return k

