import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.utils.data

from torchvision.models.inception import inception_v3
from torchvision.utils import make_grid


import numpy as np
from scipy.stats import entropy

import matplotlib.pyplot as plt

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def show_tensor_images(image_tensor, title, num_images=8, nrow=4, save_path=None):
    """
    Plot image tensors in a grid.
    Args:
        image_tensor: A tensor of images with shape (N,C,H,W).
        title: The title of the plot.
        num_images: Number of images to plot.
        nrow: Number of images displayed in each row of the grid.
        save_path: If save_path is provided, save the plot instead of showing.
    """
    image_tensor = image_tensor.detach().cpu()
    image_grid = make_grid(image_tensor[:num_images], nrow=nrow, padding=5)
    aspect_ratio = image_grid.shape[2] / image_grid.shape[1]
    plt.figure(dpi=150, figsize=(6 * aspect_ratio, 6), constrained_layout=True)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return image_grid

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def postprocess(images):
	return [tensor2img(image) for image in images]

class ScoreMeter:
    def __init__(self, score_names):
        self.score_names = score_names
        self.score_sum = {name: 0 for name in score_names}
        self.count = 0

    def update(self, scores, n=1):
        for name, score in zip(self.score_names, scores):
            self.score_sum[name] += score * n
        self.count += n

    def reset(self):
        self.__init__(self.score_names)

    def stats_dict(self):
        stats = dict()
        for k, v in self.score_sum.items():
            stats[k] = v / self.count
        return stats

    def stats_string(self):
        stats = self.stats_dict()
        stats_string = ""
        for name, stat in stats.items():
            stats_string += f"{name} {stat:.3f} | "
        return stats_string[:-3]