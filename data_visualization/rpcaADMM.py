import sys
import time
import numpy as numpy
from numpy import *
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 5
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000

# Set image patch size
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    data = mpimg.imread(filename)
    return data[:,:,0].reshape((data.shape[0],data.shape[1]))

def bw_img_to_rgb(img):
    img_rgb = numpy.zeros((img.shape[0],img.shape[1],3), dtype=numpy.uint8)
    img_u8 = img_float_to_uint8(img)
    img_rgb[:,:,0] = img_u8
    img_rgb[:,:,1] = img_u8
    img_rgb[:,:,2] = img_u8
    return img_rgb


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = bw_img_to_rgb(gt_img)
        img8 = bw_img_to_rgb(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def prox_l1(v,lambdat):
    """
    The proximal operator of the l1 norm.

    prox_l1(v,lambdat) is the proximal operator of the l1 norm
    with parameter lambdat.

    Adapted from: https://github.com/cvxgrp/proximal/blob/master/matlab/prox_l1.m
    """

    return maximum(0, v - lambdat) - maximum(0, -v - lambdat)


def prox_matrix(v,lambdat,prox_f):
    """
    The proximal operator of a matrix function.

    Suppose F is a orthogonally invariant matrix function such that
    F(X) = f(s(X)), where s is the singular value map and f is some
    absolutely symmetric function. Then

    X = prox_matrix(V,lambdat,prox_f)

    evaluates the proximal operator of F via the proximal operator
    of f. Here, it must be possible to evaluate prox_f as prox_f(v,lambdat).

    For example,

    prox_matrix(V,lambdat,prox_l1)

    evaluates the proximal operator of the nuclear norm at V
    (i.e., the singular value thresholding operator).

    Adapted from: https://github.com/cvxgrp/proximal/blob/master/matlab/prox_matrix.m
    """

    U,S,V = svd(v,full_matrices=False)
    S = S.reshape((len(S),1))
    pf = diagflat(prox_f(S,lambdat))
    # It should be V.conj().T given MATLAB-Python conversion, but matrix
    # matches with out the .T so kept it.
    return U.dot(pf).dot(V.conj())


def avg(*args):
    N = len(args)
    x = 0
    for k in range(N):
        x = x + args[k]
    x = x/N
    return x


def objective(X_1, g_2, X_2, g_3, X_3):
    """
    Objective function for Robust PCA:
        Noise - squared frobenius norm (makes X_i small)
        Background - nuclear norm (makes X_i low rank)
        Foreground - entrywise L1 norm (makes X_i small)
    """
    tmp = svd(X_3,compute_uv=0)
    tmp = tmp.reshape((len(tmp),1))
    return norm(X_1,'fro')**2 + g_2*norm(hstack(X_2),1) + g_3*norm(tmp,1)


def rpcaADMM(data):
    """
    ADMM implementation of matrix decomposition. In this case, RPCA.

    Adapted from: http://web.stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html
    """

    pool = ThreadPool(processes=3) # Create thread pool for asynchronous processing

    N = 3         # the number of matrices to split into 
                  # (and cost function expresses how you want them)
 
    A = float_(data)    # A = S + L + V
    m,n = A.shape

    g2_max = norm(hstack(A).T,inf)
    g3_max = norm(A,2)
    g2 = 0.15*g2_max
    g3 = 0.15*g3_max

    MAX_ITER = 100
    ABSTOL   = 1e-4
    RELTOL   = 1e-2

    start = time.time()

    lambdap = 1.0
    rho = 1.0/lambdap

    X_1 = zeros((m,n))
    X_2 = zeros((m,n))
    X_3 = zeros((m,n))
    z   = zeros((m,N*n))
    U   = zeros((m,n))

    print '\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                  'r norm', 
                                                  'eps pri', 
                                                  's norm', 
                                                  'eps dual', 
                                                  'objective')

    # Saving state
    h = {}
    h['objval'] = zeros(MAX_ITER)
    h['r_norm'] = zeros(MAX_ITER)
    h['s_norm'] = zeros(MAX_ITER)
    h['eps_pri'] = zeros(MAX_ITER)
    h['eps_dual'] = zeros(MAX_ITER)

    def x1update(x,b,l):
        return (1.0/(1.0+l))*(x - b)
    def x2update(x,b,l,g,pl):
        return pl(x - b, l*g)
    def x3update(x,b,l,g,pl,pm):
        return pm(x - b, l*g, pl)

    def update(func,item):
        return map(func,[item])[0]

    for k in range(MAX_ITER):

        B = avg(X_1, X_2, X_3) - A/N + U

        # Original MATLAB x-update
        # X_1 = (1.0/(1.0+lambdap))*(X_1 - B)
        # X_2 = prox_l1(X_2 - B, lambdap*g2)
        # X_3 = prox_matrix(X_3 - B, lambdap*g3, prox_l1)

        # Parallel x-update
        async_X1 = pool.apply_async(update, (lambda x: x1update(x,B,lambdap), X_1))
        async_X2 = pool.apply_async(update, (lambda x: x2update(x,B,lambdap,g2,prox_l1), X_2))
        async_X3 = pool.apply_async(update, (lambda x: x3update(x,B,lambdap,g3,prox_l1,prox_matrix), X_3))

        X_1 = async_X1.get()
        X_2 = async_X2.get()
        X_3 = async_X3.get()

        # (for termination checks only)
        x = hstack([X_1,X_2,X_3])
        zold = z
        z = x + tile(-avg(X_1, X_2, X_3) + A*1.0/N, (1, N))

        # u-update
        U = B

        # diagnostics, reporting, termination checks
        h['objval'][k]   = objective(X_1, g2, X_2, g3, X_3)
        h['r_norm'][k]   = norm(x - z,'fro')
        h['s_norm'][k]   = norm(-rho*(z - zold),'fro');
        h['eps_pri'][k]  = sqrt(m*n*N)*ABSTOL + RELTOL*maximum(norm(x,'fro'), norm(-z,'fro'));
        h['eps_dual'][k] = sqrt(m*n*N)*ABSTOL + RELTOL*sqrt(N)*norm(rho*U,'fro');

        if (k == 0) or (mod(k+1,10) == 0):
            print '%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k+1,
                                                                  h['r_norm'][k], 
                                                                  h['eps_pri'][k], 
                                                                  h['s_norm'][k], 
                                                                  h['eps_dual'][k], 
                                                                  h['objval'][k])
        if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    h['addm_toc'] = time.time() - start
    h['admm_iter'] = k
    h['X1_admm'] = X_1
    h['X2_admm'] = X_2
    h['X3_admm'] = X_3

    return h


def main(argv=None):  # pylint: disable=unused-argument
    
    #data_dir = 'output-masks/mask_4.png'
    data_dir = '../data/training/images/satImage_001.png'
    # Extract it into numpy arrays.
    X = extract_data(data_dir, TRAINING_SIZE)
    h = rpcaADMM(X)

    L = h['X2_admm'] # Lower rank matrix that represents X

    imgBefore = X
    imgAfter = L

    cimg = concatenate_images(imgBefore, imgAfter)
    #fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg)
    #plt.imshow(imgs[imgIndx], cmap='Greys_r')
    plt.show()

if __name__ == '__main__':
    main()