import gym
import sys, os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
from scipy import misc

class CarparkEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0

    def __init__(self):
        
    def _step(self, action):
    
    def _reset(self):
    
    def _get_agent_start_target_state(self):

    def _render(self, mode='human', close=False):

    def change_start_state(self, sp):

    def change_target_state(self, tg):

    def get_agent_state(self):

    def get_start_state(self):

    def get_target_state(self):

    def jump_to_state(self):



def img_reshape(img, target_shape):
    ''' Input: img: numpy array of size W*H*C,
               target_shape: target shape list,
                    tw, th, the same channels
    '''
    img = Image.fromarray(img)
    size = (target_shape[0], target_shape[1])
    img = img.resize(size)
    res_img = np.array(img)
    return res_img

def homography_solve(u, v):
    ''' Input: u, v : are both 2*4 matrix, representing 4 points
        Output: H matrix, 3*3
    '''
    A = np.zeros((8,8))
    for i in range(8): # rows
        for j in range(8): # columns
            if i>=0 and i <=3 and j >=0 and j <=1:
                A[i,j] = u[j,i]
            elif i>=0 and i<=3 and j == 2:
                A[i,j] = 1
            elif i>=0 and i<=3 and j >=6 and j <= 7:
                A[i,j] = -1.0*u[j-6,i]*v[0,i]
            elif i>3 and j>=3 and j<=4:
                A[i,j] = u[j-3,i-4]
            elif i>3 and j == 5:
                A[i,j] = 1
            elif i>3 and j >=6 and j <= 7:
                A[i,j] = -1.0*u[j-6,i-4]*v[1,i-4]
    b = np.zeros((8,1))
    for i in range(8):
        if i < 4:
            b[i] = v[0, i]
        else:
            b[i] = v[1, i-4]
    h = np.linalg.inv(A).dot(b)
    H = np.ones((9,1))
    H[0:8] = h
    H = H.reshape((3,3))
    return H

def homography_transform(u, H):
    '''
    u: 2 * N matrix
    H: 3 * 3 matrix
    '''
    U = np.ones((3, u.shape[1]))
    U[0:2,:] = u
    V = H.dot(U)
    V = V/(V[2,:])
    v = V[0:2,:]
    return v

def premerge_img(img1, img2, Hmat):
    '''
    merge img1 into img2
    '''
    H, W, C = img1.shape
    H2, W2, C = img2.shape
    
    # map points in img 1 to img 2
    U = np.zeros((2, H*W))
    for i in range(W):
        U[0, i*H : (i+1)*H] = i
        U[1, i*H : (i+1)*H] = np.arange(H)
    V = homography_transform(U, Hmat)
    V = np.around(V)
    V = V.astype(np.int32)
    U = U.astype(np.int32)
    for i in range(U.shape[1]):
        if V[0,i] < W2 and V[0,i] >= 0 and V[1,i] < H2 and V[1,i] >= 0:
            img2[V[1,i], V[0,i], 0:3] = img1[U[1,i], U[0,i], 0:3]
    return img2, V

def merge_img(img1, img2, v):
    ''' merge two images img1, img2 based on
        two sets of corresponding points 
    '''
    H, W, C = img1.shape
    u = np.array([[1,1],[W-1,1], [1, H-1],[W-1,H-1]]).T
    H1 = homography_solve(u, v)
    img, V = premerge_img(img1, img2, H1)
    return img, V
