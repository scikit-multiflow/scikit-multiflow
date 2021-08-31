import numpy as np
import math
import pickle
import random
import numpy as np
import torch as th
import pdb

class Rotate(object):

    def __init__(self, 
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """

        # Acknowledgements
        # Thank you to the following people and contributors:

        # All Keras contributors
        # @deallynomore
        # @recastrodiaz
        
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rotation_matrix = th.FloatTensor([
                                    [math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]
                                ])
        if self.lazy:
            return rotation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       rotation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]

def th_nearest_interp2d(input, coords):
    """
    2d nearest neighbor interpolation th.Tensor
    """
    # take clamp of coords so they're in the image bounds
    x = th.clamp(coords[:,:,0], 0, input.size(1)-1).round()
    y = th.clamp(coords[:,:,1], 0, input.size(2)-1).round()

    stride = th.LongTensor(input.stride())
    x_ix = x.mul(stride[1]).long()
    y_ix = y.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1)

    mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

    return mapped_vals.view_as(input)

def th_bilinear_interp2d(input, coords):
    """
    bilinear interpolation in 2d
    """
    x = th.clamp(coords[:,:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = th.clamp(coords[:,:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = th.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1)

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))
    
    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)

def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on th.Tensor
    
    Arguments
    ---------
    x : th.Tensor of size (C, H, W)
        image tensor to be transformed
    matrix : th.Tensor of size (3, 3) or (2, 3)
        transformation matrix
    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use
    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin
    Example
    ------- 
    >>> import torch
    >>> from torchsample.utils import *
    >>> x = th.zeros(2,1000,1000)
    >>> x[:,100:1500,100:500] = 10
    >>> matrix = th.FloatTensor([[1.,0,-50],
    ...                             [0,1.,-50]])
    >>> xn = th_affine2d(x, matrix, mode='nearest')
    >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """

    # Acknowledgements
    # Thank you to the following people and contributors:

    # All Keras contributors
    # @deallynomore
    # @recastrodiaz

    if matrix.dim() == 2:
        matrix = matrix[:2,:]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3,3):
            matrix = matrix[:,:2,:]

    A_batch = matrix[:,:,:2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0),1,1)
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1),x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0),1,1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(1) / 2. - 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(2) / 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(1) / 2. - 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(2) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed

def th_iterproduct(*args):
    return th.from_numpy(np.indices(args).reshape((len(args),-1)).T)