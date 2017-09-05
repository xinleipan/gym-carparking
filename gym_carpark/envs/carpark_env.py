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
        ''' action space defintion '''
        self.actions = (0, 1, 2, 3, 4) # stay, move ahead, move back, turn left, right
        self.inv_actions = (0, 2, 1, 4, 3)
        self.action_space = spaces.Discrete(5)
        
        ''' observation space definition '''
        self.obs_shape = [256,256,4]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape)
         
        ''' initialize system state '''
        this_file_path = os.path.dirname(os.path.realpath(__file))
        self.bg_img_path = os.path.join(this_file_path, 'parking3.png')
        self.car_img_path = os.path.join(this_file_path, 'Car.jpg')
        self.bg_img = np.array(Image.open(self.bg_img_path))
        self.car_img = np.array(Image.open(self.car_img_path))
        self.background = img_reshape(self.bg_img, (self.obs_shape[0], self.obs_shape[1]))
        self.car_size = (80, 50)

        ''' initialize observation space '''
        self.observation = copy.deepcopy(self.background)
    
        ''' agent state: start, target, current state '''
        self.agent_start_state = [30, self.car_size[0]+29, 30, self.car_size[0]+29, \
                                    30, 30, self.car_size[1]+29, self.car_size[1] + 29, 0.0]
        self.agent_target_state = [162, 241, 162, 241, 152, 152, 201, 201, 0.0]
        self.agent_state = copy.deepcopy(self.agent_start_state) 
        self.observation, _ = self.update_observation(self.agent_state) 
        
        ''' set other parameters '''
        self.restart_once_done = True # restart once done or not
        self.verbose = False # to show the environment or not

        if self.verbose == True:
            CarparkEnv.num_env += 1
            self.fig = plt.figure(CarparkEnv.num_env)
            plt.show(block=False)
            plt.axis('off')
            self._render() 

    def _agent_state_to_vex(self, agent_state):
        (x1, x2, x3, x4, y1, y2, y3, y4, angle) = agent_state
        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
 
    def update_observation(self, agent_state=None):
        if agent_state is None:
            agent_state = copy.deepcopy(self.agent_state)
        agent_vex = self._agent_state_to_vex(agent_state)
        old_observation = copy.deepcopy(self.observation)
        observation = copy.deepcopy(self.background)
        observation = np.asarray(observation, np.uint8)
        observation, V = merge_img(self.car_img, observation, agent_vex.T)
        overlap_region = self.background[V[1,:], V[0,:], 0:3]
        overlap_region = list(overlap_region)
        is_collision = False
        if any((np.array([255,255,255]) == x).all() for x in overlap_region):
            is_collision = True
        elif any((np.array([0,255,0]) == x).all() for x in overlap_region):
            is_collision = True
        if is_collision:
            return old_observation, is_collision
        else:
            return observation, is_collision

    def _step(self, action):
        ''' step function, returns obs, reward, done, success '''
        success = False
        if action == 0:  # stay in place
            return (self.observation, 0, False, True)
        elif action == 1:
            new_carpos_vex = self._agent_state_to_vex(self.agent_state)
            sign = True
            angle = self.agent_state[-1]
            for j in range(new_carpos_vex.shape[0]):
                new_carpos_vex[j][0] += np.round(2.0 * np.cos(angle))
                new_carpos_vex[j][1] += np.round(2.0 * np.sin(angle))
                if new_carpos_vex[j][0] < 0 or new_carpos_vex[j][0] >= self.obs_shape[0] \
                    or new_carpos_vex[j][1] < 0 or new_carpos_vex[j][1] >= self.obs_shape[1]:
                    sign = False
                    break
            if sign == True:
                success = True
                new_vex = copy.deepcopy(new_carpos_vex)
        elif action == 2: # 'move back'
            new_carpos_vex = self._agent_state_to_vex(self.agent_state)
            sign = True
            angle = self.agent_state[-1]
            for j in range(new_carpos_vex.shape[0]):
                new_carpos_vex[j][0] -= np.round(2.0 * np.cos(angle))
                new_carpos_vex[j][1] -= np.round(2.0 * np.sin(angle))
                if new_carpos_vex[j][0] < 0 or new_carpos_vex[j][0] >= self.obs_shape[0] \
                    or new_carpos_vex[j][1] < 0 or new_carpos_vex[j][1] >= self.obs_shape[1]:
                    sign = False
                    break
            if sign == True:
                success = True
                new_vex = copy.deepcopy(new_carpos_vex)
        elif action == 3 or action == 4:
            old_angle = self.agent_state[-1]
            rot_angle = 90.0/180.0*np.pi
            new_carpos_vex = self._agent_state_to_vex(self.agent_state)
            if action == 4:
                rot_angle *= -1.0
            new_angle = old_angle + rot_angle
            rot_mat = np.array([[np.cos(rot_angle), -1.0 * np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
            identity_mat = np.array([[1. ,0.], [0., 1.]])
            coord_avg = np.mean(new_carpos_vex, axis=0)
            x_avg, y_avg = int(np.round(coord_avg[0])), int(np.round(coord_avg[1]))
            avg_coord = np.array([x_avg, y_avg])
            sign = True
            for j in range(new_carpos_vex.shape[0]):
                this_coord = new_carpos_vex[j]
                new_coord = rot_mat.dot(this_coord)+(identity_mat-rot_mat).dot(avg_coord)
                new_coord = np.round(new_coord)
                new_carpos_vex[j] = new_coord
                if new_carpos_vex[j][0] < 0 or new_carpos_vex[j][0] >= self.obs_shape[0]\
                    or new_carpos_vex[j][1] < 0 or new_carpos_vex[j][1] >= self.obs_shape[1]:
                    sign = False
                    break
            if sign == True:
                success = True
                new_vex = copy.deepcopy(new_carpos_vex)
        new_agent_state = [new_vex[0,0], new_vex[1,0], new_vex[2,0], new_vex[3,0],
                        new_vex[0,1], new_vex[1,1], new_vex[3,1], new_vex[4,1], 0]
        self.observation, is_collision = self.update_observation(new_agent_state)
        if is_collision:
            success = False
        if success:
            if action ==3 or action == 4:
                while new_angle < 0:
                    new_angle += 2.0 * np.pi
                while new_angle >= np.pi * 2.0:
                    new_angle -= 2.0 * np.pi
                if new_angle >= 2.0 * np.pi:
                    new_angle = new_angle - 2.0 * np.pi
                if new_angle < 0 or new_angle >= 2.0 * np.pi:
                    sys.exit('wrong angle!')
                self.agent_state = new_agent_state[:-1] + [new_angle]
            else:
                self.agent_state[:-1] = new_agent_state[:-1]    
        diff = np.sum(np.abs(np.array(self.agent_state) - np.array(self.agent_target_state)))
        if diff <= 10:
            done = True
            if self.restart_once_done:
                self.observation = self._reset()
            reward = 1
        else:
            done = False
            reward = 0
        self._render()
        return (self.observation, reward, done, success)

    def _reset(self):
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.observation, is_collision = self.update_observation(self.agent_state)
        self._render()
        return self.observation   
 
    def _render(self, mode='human', close=False):
        if self.verbose == False:
            return
        else:
            img = self.observation
        fig = plt.figure(CarparkEnv.num_env)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 

    def change_start_state(self, sp):
        if self.agent_start_state == sp:
            _ = self._reset()
            return True
        else:
            observation, is_collision = self.update_observation(sp)
            if is_collision:
                return False
            else:
                self.agent_start_state = sp
                self.agent_state = copy.deepcopy(sp)
                self._reset()
                return True

    def change_target_state(self, tg):
        if self.agent_target_state == tg:
            _ = self._reset()
            return True
        else:
            observation, is_collision = self.update_observation(tg)
            if is_collision:
                return False
            else:
                self.agent_target_state = tg
                self._reset()
                return True

    def get_agent_state(self):
        return self.agent_state

    def get_start_state(self):
        return self.agent_start_state

    def get_target_state(self):
        return self.agent_target_state

    def jump_to_state(self, to_state):
        ''' move agent to another state '''
        ''' to_state: a list of x1-x4, y1-y4, angle '''
        (x1, x2, x3, x4, y1, y2, y3, y4, angle) = to_state
        self.agent_state = to_state
        self.observation, is_collision = self.update_observation(to_state)
        self._render()
        return self.observation 


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
