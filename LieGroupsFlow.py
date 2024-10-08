
# Initilaze:

import numpy as np
import math
import matplotlib.pyplot as plt

from skimage import data, color, filters
from scipy.linalg import logm, expm
from skimage.transform import resize
from tqdm import tqdm



def angles_range(array):
    '''Keep the angles representation in [-pi, pi] using vectorized operations.'''
    return (array + np.pi) % (2 * np.pi) - np.pi


########### Orientations matrix to Lie group manifold: ######################

def so2(theta):
    '''' 
    Parameters:
    - theta: one orientation or image of orientarions.

    Returns:
    -  tensor with the corresponding so(2) group action. '''

#     def orientation2matrix(single_orientation):
#         t = single_orientation
#         matrix = np.ndarray(shape=(2, 2),buffer=np.array([np.cos(t), -np.sin(t), np.sin(t), np.cos(t)]))
#         return matrix

#     if type(theta) == float:                   # Single orientation to SO(2) matrix:
#         return orientation2matrix(theta)

#     if type(theta) == np.float64:                   # Single orientation to SO(2) matrix
#         return orientation2matrix(theta)

#     else:
#         n, m = theta.shape
#         matrix = np.empty(shape=(n, m, 2, 2))
#         for x in range(n):
#             for y in range(m):
#                 matrix[x, y] = orientation2matrix(theta[x, y])
#     return matrix

    if isinstance(theta, (float, np.float64)):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    # For an array of orientations
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        # Stack to create the matrix array with shape (n, m, 2, 2)
        matrix = np.stack((np.stack((c, -s), axis=-1), np.stack((s, c), axis=-1)), axis=-2)
        return matrix


# ######### Lie group manifold to Lie algebra manifold:######################

def to_lie_alg_manifold(lie_group_manifold):
    n, m , nx, ny = lie_group_manifold.shape
    lie_alg_manifold = np.zeros(lie_group_manifold.shape)
    for x in range(n):
        for y in range(m):
            alg = logm(lie_group_manifold[x,y]+1e-6)
#             if alg.dtype =='complex128': # The angle is too close to pi. 
#                 print(alg)
            lie_alg_manifold[x,y] = alg

    return lie_alg_manifold


def to_lie_group_manifold(lie_alg_manifold):
    n, m , nx, ny = lie_alg_manifold.shape
    lie_g_manifold = np.zeros(lie_alg_manifold.shape)
    for x in range(n):
        for y in range(m):
            g = expm(lie_alg_manifold[x,y])
#             if alg.dtype =='complex128': # The angle is too close to pi. 
#                 print(alg)
            lie_g_manifold[x,y] = g

    return lie_g_manifold


def tensor_dexp(A, B):
    ''' Perform the dexp action over each element in tensors A and B '''
    
    def dexp(A, B):
        # Commutator Operator applied to entire arrays at once
        op = A @ B - B @ A
        
        ad1 = op
        ad2 = A @ ad1 - ad1 @ A
        ad3 = A @ ad2 - ad2 @ A
        ad4 = A @ ad3 - ad3 @ A
        
        return B + ad1/2 + ad2/6 + ad3/24 + ad4/120

    return dexp(A, B)


def lie_alg_manifold_to_orientations(lie_alg_manifold):
    n, m , nx, ny = lie_alg_manifold.shape
    thetas = np.zeros((n,m))
    for x in range(n):
        for y in range(m):
            thetas[x,y] = lie_alg_manifold[x,y][1,0].real
            
    t = angles_range(thetas)
    return t

def angle_subtraction(angle1, angle2):
    """
    Calculate the circular distance between two angles in radians.

    Parameters:
    - angle1: First angle in radians.
    - angle2: Second angle in radians.

    Returns:
    - Circular distance between the angles.
    """
    # We know angles are within (-pi, pi) radians
#     assert angle1_rad > -np.pi and angle2_rad > -np.pi and angle1_rad < np.pi and angle2_rad < np.pi

    # Calculate circular distance
    dist = (angle1 - angle2) % (2*np.pi)
    
    if type(angle1) == float:
        if dist > np.pi:
            dist = dist - 2*np.pi
    else:
        dist = np.where(dist>np.pi, dist-2*np.pi , dist)

    return dist



def angles_dist(angles_img1,angles_img2):
    
    all_dists = np.zeros(angles_img1.shape)
    for x in range(angles_img1.shape[0]):
        for y in range(angles_img1.shape[1]):
            all_dists[x,y] = angle_subtraction(angles_img1[x,y],angles_img2[x,y])
    return all_dists

def mean_angle(angle1, angle2):
    """
    Calculate the mean angle between two angles on the unit circle.

    Parameters:
    - angle1: First angle in radians.
    - angle2: Second angle in radians.

    Returns:
    - Mean angle between angle1 and angle2 in radians.
    """
    # Convert angles to unit vectors
    x1, y1 = np.cos(angle1), np.sin(angle1)
    x2, y2 = np.cos(angle2), np.sin(angle2)
    
    # Calculate the mean vector
    x_mean = (x1 + x2) / 2
    y_mean = (y1 + y2) / 2
    
    # Convert the mean vector back to an angle in [-pi
    mean_angle = np.arctan2(y_mean, x_mean)
    
    return mean_angle


#################### Lie algebra derivative ####################


# def algebra_dx(alg):
#     ''' Vectorized dx derivative of algebra-manifold '''
#     # Pad the array by extending the first and last columns
#     pad_alg = np.pad(alg, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='edge')
    
#     # Compute the difference between adjacent columns
#     dx_alg = angles_dist(pad_alg[:, 2:, :, :] , pad_alg[:, :-2, :, :]) / 2.0
    
#     return dx_alg

# def algebra_dy(alg):
#     ''' Vectorized dy derivative of algebra-manifold '''
#     # Pad the array by extending the first and last rows
#     pad_alg = np.pad(alg, ((1, 1), (0, 0), (0, 0), (0, 0)), mode='edge')
    
#     # Compute the difference between adjacent rows
#     dy_alg = angles_dist(pad_alg[2:, :, :, :] , pad_alg[:-2, :, :, :]) / 2.0
    
#     return dy_alg

# def image_Ix(image):
#     ''' Backward derivetive f_x(x,y)=f(x,y)-f(x-1,y) '''
#     P = np.pad(image, 1, mode='edge')
#     O1 = P[1:-1, 1:-1]
#     O2 = P[1:-1, :-2]
#     n,m = image.shape
#     Ox = np.zeros((n,m))
#     for x in range(m):
#             for y in range(n):
#                    Ox[y,x] = angle_subtraction(O1[y,x],O2[y,x])
#     return Ox

# def image_Iy(image):
#     ''' Backward derivetive '''
#     P = np.pad(image, 1, mode='edge')
#     O1 = P[1:-1, 1:-1]
#     O2 = P[:-2, 1:-1]

#     n,m = image.shape
#     Oy = np.zeros((n,m))
#     for x in range(m):
#         for y in range(n):
#             Oy[y,x] = angle_subtraction(O1[y,x],O2[y,x])
  
#     return Oy


def algebra_dx(alg):
    ''' Vectorized dx derivative of algebra-manifold '''
    # Pad the array by extending the first and last columns
    pad_alg = np.pad(alg, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='edge')
    
    # Compute the difference between adjacent columns
    dx_alg = mean_angle(pad_alg[:, 2:, :, :] , pad_alg[:, :-2, :, :]) 
    
    return dx_alg

def algebra_dy(alg):
    ''' Vectorized dy derivative of algebra-manifold '''
    # Pad the array by extending the first and last rows
    pad_alg = np.pad(alg, ((1, 1), (0, 0), (0, 0), (0, 0)), mode='edge')
    
    # Compute the difference between adjacent rows
    dy_alg = mean_angle(pad_alg[2:, :, :, :] , pad_alg[:-2, :, :, :]) 
    
    return dy_alg

def image_Ix(image):
    ''' Backward derivetive f_x(x,y)=f(x,y)-f(x-1,y) '''
    P = np.pad(image, 1, mode='edge')
    O1 = P[1:-1, 1:-1]
    O2 = P[1:-1, :-2]
    Ox = mean_angle(O1,O2)
    return Ox

def image_Iy(image):
    ''' Backward derivetive '''
    P = np.pad(image, 1, mode='edge')
    O1 = P[1:-1, 1:-1]
    O2 = P[:-2, 1:-1]

    Oy = mean_angle(O1,O2)
    return Oy


def gamma_metric(thetas):
    " g is an element in the group, termμ is (g-1 ∂μ g) "


    Ox = image_Ix( thetas )
    Oy = image_Iy( thetas )
    xx = 1 + np.power(Ox,2)
    xy = Ox * Oy
    yy = 1 + np.power(Oy,2)

    r, c = thetas.shape
    
    # # this section calculates the full metric tensor but is unnecessary for the current implementation,
    # # which only requires the inverse tensor and its determinant.
    # gamma = np.zeros((r,c,2,2))
    # gamma[:, :, 0, 0] = xx
    # gamma[:, :, 0, 1] = xy
    # gamma[:, :, 1, 1] = yy
    # gamma[:, :, 1, 0] = xy

    gamma_det = xx + np.power(Oy,2)

    gamma_inv = np.zeros((r,c,2,2))
    gamma_inv[:, :, 0, 0] = yy /gamma_det
    gamma_inv[:, :, 0, 1] = -xy /gamma_det
    gamma_inv[:, :, 1, 1] = xx /gamma_det
    gamma_inv[:, :, 1, 0] = -xy /gamma_det

    return gamma_inv , gamma_det


# In[141]:


def tensor_fn(manifold, action):
    ''' Perform the given action over g(x,y), the so2 matrix in the place (x,y) on the manifold. '''

    n, m, i, j = manifold.shape
    tns = np.empty(manifold.shape)

    for x in range(n):
        for y in range(m):
            tns[x, y] = action(manifold[x, y])

    return tns



def tensor2matrix_fn(manifold, action):
    " perform the action over g(x,y), with matrix as result, for actions like det, tensor.. "
    n, m, i, j = manifold.shape
    mat = np.empty((n, m))

    for x in range(n):
        for y in range(m):
            mat[x, y] = action(manifold[x, y])
    return mat


# In[143]:


def entrywise_matrix_product(A, B):
    n, m, i, j = A.shape

    mat = np.empty((n, m, i, j))

    for x in range(n):
        for y in range(m):
            mat[x, y] = A[x, y] @ B[x, y]
    return mat

def product(A, T):
    ''' A is nXn matrix
        T is nXnXkXk tensor
        we want to multiply the value aij with the matrix Tij '''
    n,m = A.shape
    resultado = np.empty(T.shape)
    for i in range(n):
        for j in range(m):
            resultado[i, j] = A[i, j] * T[i, j]
    return resultado

def lie_algebra_derivative(lie_alg_manifold):
    
    # 2. Evaluate the term 𝑔−1∂𝜈𝑔 in the Lie algebra via the dexp series over the algebra terms
    dx_a = algebra_dx(lie_alg_manifold)
    dy_a = algebra_dy(lie_alg_manifold)
    dexpPx = tensor_dexp(-lie_alg_manifold, dx_a)
    dexpPy = tensor_dexp(-lie_alg_manifold, dy_a)


    # pre 3. calculate the thetas
    thetas = lie_alg_manifold_to_orientations(lie_alg_manifold)

    # 3. Calculate γ^μν and γ
    gamma_inv , gamma_det = gamma_metric(thetas)  # tensor of gammas

    sq = np.sqrt(gamma_det)

    # Calculate the term ∂μ(√γ γ μν (g−1 ∂ν g))

    term_xx = product(sq * gamma_inv[:, :, 0, 0], dexpPx)
    term_xy = product(sq * gamma_inv[:, :, 0, 1], dexpPy)
    term_yx = product(sq * gamma_inv[:, :, 1, 0], dexpPx)
    term_yy = product(sq * gamma_inv[:, :, 1, 1], dexpPy)

    Px = algebra_dx(term_xx +term_xy)
    Py =  algebra_dy(term_yx +term_yy)

    direction = Px+Py

    # term11 = product(sq * gamma_inv[:, :, 0, 0], dexpPx)
    # term12 = product(sq * gamma_inv[:, :, 0, 1], dexpPy)
    # term21 = product(sq * gamma_inv[:, :, 1, 0], dexpPx)
    # term22 = product(sq * gamma_inv[:, :, 1, 1], dexpPy)

    # direction = tensor_fn(term11 + term12, image_Ix) + tensor_fn(term21 + term22, image_Iy)
    direction = product( 1/sq , direction )

    if np.any(np.isnan(direction)):
        print("******************Direction create NaN")

    return direction


# # Forward Euler method for Lie algebra manifold:

def forword_euler_mathod_lie_alg(lie_alg_manifold,num_steps=20,dt=0.01):

    # Forward Euler integration in Lie algebra for SO(2)
    def lie_algebra_euler_integration(lie_alg_manifold, dt):
        return lie_alg_manifold + dt * lie_algebra_derivative(lie_alg_manifold)

    # Perform the integration over time
    current_lie_algebra_manifold = lie_alg_manifold

    for _ in tqdm(range(num_steps)):
        current_lie_algebra_manifold = lie_algebra_euler_integration(current_lie_algebra_manifold, dt)

    return current_lie_algebra_manifold

# g_n+1 = g_n @ exp(dt*a_n)
def forword_euler_mathod(g_manifold,num_steps=20,dt=0.01):

    # Forward Euler integration for Lie group SO(2)
    def euler_integration(g_manifold, dt):
        a_n = lie_algebra_derivative(g_manifold)
        exp_an = to_lie_group_manifold(dt * a_n)
        g_next = entrywise_matrix_product(g_manifold,exp_an)
        return g_next

    # Perform the integration over time
    current_g_manifold = g_manifold

    for _ in tqdm(range(num_steps)):
        current_g_manifold = euler_integration(current_g_manifold, dt)

    return current_g_manifold

