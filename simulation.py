import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import multiprocessing as mp

# class leaf():
#     def __init__(self, bbox=np.array([[-1, 1],[-1, 1]]), depth=1):
#         # if the leaf has a particle ... 
#         self.mass = None
#         self.com = None
#         # center position of quadrant 
#         self.center = np.array([bbox[0].sum() / 2, bbox[1].sum() / 2])
#         # length of bounding box, same for x- and y- axes 
#         self.length = bbox[0, 1] - bbox[0, 0]
#         self.depth = depth 

        
# class QuadTree():
#     # tree constructor .. 
#     def __init__(self, oleaf=leaf()):
#         # inherit from the original leaf node ... 
#         self.center = oleaf.center
#         self.length = oleaf.length
#         self.depth = oleaf.depth
#         self.mass = oleaf.mass 
#         self.com = oleaf.com 
        
#         # initialize leaf nodes ... 
#         self.ll = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0] - oleaf.length / 2, oleaf.center[0]], 
#                                  [oleaf.center[1] - oleaf.length / 2, oleaf.center[1]]]),)
#         self.ul = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0] - oleaf.length / 2, oleaf.center[0]], 
#                                  [oleaf.center[1], oleaf.center[1] + oleaf.length / 2]]))
#         self.ur = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0], oleaf.center[0] + oleaf.length / 2], 
#                                  [oleaf.center[1], oleaf.center[1] + oleaf.length / 2]]))
#         self.lr = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0], oleaf.center[0] + oleaf.length / 2], 
#                                  [oleaf.center[1] - oleaf.length / 2, oleaf.center[1]]]))
        
        
class leaf():
    def __init__(self, bbox=np.array([[-1, -1],[1, 1]]), depth=1):
        # if the leaf has a particle ... 
        self.mass = None
        self.com = None
        # center position of quadrant 
        self.center = np.array([bbox[:, 0].sum() / 2, bbox[:, 1].sum() / 2])
        # length of bounding box, same for x- and y- axes for regular tree code, not same for partitioned leaves
        self.length = bbox[1] - bbox[0]
        self.depth = depth 

        
class QuadTree():
    # tree constructor .. 
    def __init__(self, oleaf=leaf()):
        # inherit from the original leaf node ... 
        self.center = oleaf.center
        self.length = oleaf.length
        self.depth = oleaf.depth
        self.mass = oleaf.mass 
        self.com = oleaf.com 
        
        self.ll = leaf(depth=self.depth + 1, 
                       bbox = np.array([oleaf.center - oleaf.length / 2, oleaf.center]))
        self.ul = leaf(depth=self.depth + 1, 
                       bbox = np.array([oleaf.center + np.array([-1, 0]) * oleaf.length / 2, 
                                        oleaf.center + np.array([0, 1]) * oleaf.length / 2]))
        self.ur = leaf(depth=self.depth + 1, 
                       bbox = np.array([oleaf.center, oleaf.center + oleaf.length / 2]))
        self.lr = leaf(depth=self.depth + 1, 
                       bbox = np.array([oleaf.center + np.array([0, -1]) * oleaf.length / 2, 
                                        oleaf.center + np.array([1, 0]) * oleaf.length / 2]))
   
        
def quadAssign(p, node): 
    theta = np.arctan2(p[1] - node.center[1], p[0] - node.center[0]) * 180 / np.pi
    if theta < 90 and theta > 0: 
        node.ur = assignParticle(p, node.ur)
    elif theta < 180 and theta >= 90: 
        node.ul = assignParticle(p, node.ul)
    elif theta < 0 and theta >= -90:
        node.lr = assignParticle(p, node.lr)
    else: # theta < 360 and theta >= 270 
        node.ll = assignParticle(p, node.ll)
    return node
    
        
def assignParticle(p, node):
    # if a leaf has no particle, assign particle to leaf 
    if isinstance(node, leaf) and node.com is None:
        node.com = p
        node.mass = 1
    # if a leaf has a particle, make it a tree and re-assign particles
    elif isinstance(node, leaf) and node.com is not None: 
        p_old = node.com 
        node = QuadTree(node)
        node = assignParticle(p, node)  
        node = assignParticle(p_old, node)
        node.com = (p_old + p) / 2
        node.mass = 2
    # if tree, find out where in the tree the particle should live 
    else: 
        node = quadAssign(p, node)
        node.com = (node.com * node.mass + p) / (node.mass + 1)
        node.mass = node.mass + 1
        
    return node
    

def in_box(com, corners):
    # a point is in a box if it is gt the bottom left corner and lt top right corner 
    return (com[0] >= corners[0][0] and com[0] <= corners[1][0]) and (com[1] >= corners[0][1] and com[1] <= corners[1][1])


def solve_acc(p, node, L, epsilon=1e-3, G = 1e-6, m_scale=1e3, r_scale=1): 
    corners = np.array([[p[0] - L/2, p[1] - L/2], [p[0] + L/2, p[1] + L/2]])
    
    n_com = node.com 
    n_c = node.center
    n_l = node.length
    n_m = node.mass
    
    # check how many corners are in the effective interaction volume 
    t = [not in_box(np.array([n_c[0] - n_l/2, n_c[1] - n_l/2]), corners), 
          not in_box(np.array([n_c[0] + n_l/2, n_c[1] - n_l/2]), corners),
          not in_box(np.array([n_c[0] + n_l/2, n_c[1] + n_l/2]), corners),
          not in_box(np.array([n_c[0] - n_l/2, n_c[1] + n_l/2]), corners)]
    
    # if not all or no corners are in, it straddles the boundary and must be partitioned
    if sum(t) < 4 and sum(t) > 0: 

        return np.array([0, 0])
    
    # check if the node's center of mass is within the box. if it is, return standard force calculation 
    # if sum(t) == 0: 
    if in_box(n_com, corners) :
        return G * m_scale * node.mass * (n_com - p) / sum((n_com - p)**2 + epsilon)**(3 / 2)
    
    # we determine the node boundary is outside periodic boundary thus we need to figure out where to loop 
    n_com[n_com > p + L/2] = n_com[n_com > p + L/2] - L
    n_com[n_com < p -L/2] = n_com[n_com < p + -L/2] + L

    return G * m_scale * node.mass * (n_com - p) / sum((n_com - p)**2 + epsilon)**(3 / 2)



def f_multipole(p, node, L, theta=0.5):  
    # check to see if particle is looking at itself,return 0 
    if node.mass is not None and sum((node.com - p)**2)**0.5 < 1e-10:
        return np.zeros(2) 
    
    # if a leaf node and has a particle, directly solve for gravity 
    if isinstance(node, leaf) and node.mass is not None:
        return solve_acc(p, node, L)
    # if a leaf node and doesn't have a particle, return 0
    elif isinstance(node, leaf) and node.mass is None:
        return np.zeros(2)
    
    # if a quadtree that satisfies multipole condition, solve for gravity
    if isinstance(node, QuadTree) and node.length / sum((node.com - p)**2)**0.5 <= theta: 
        return solve_acc(p, node, L)
    # else if quadtree and hasn't satisfied, recursively search the tree's children 
    elif isinstance(node, QuadTree) and node.length / sum((node.com - p)**2)**0.5 > theta: 
        return sum(np.array([f_multipole(p, ch, L, theta=theta) for ch in [node.ll, node.ur, node.lr, node.ul]]))
        

def leapfrog(r, t_start=0, t_end=10, N=1e4, L=2, theta=0.5, multi=False):
    dt = (t_end - t_start)/N

    tpoints = np.arange(t_start, t_end, dt)
    xpoints = []
    trees = []
    
    acc = np.zeros((len(r), 2))
    pos = r[:, 0:2]
    vel = r[:, 2: ]
    
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=n_cpu)

    for t in tqdm(tpoints):
        # have to .copy() because we are updating the same array and that causes memory issues 
        xpoints.append(np.hstack((pos, vel, acc)))
        
        # use current acceleration to kick velocity a half time step
        vel = vel + acc * dt / 2 
        # drift the position using the kicked (half time step) velocity 
        pos = pos + vel * dt 
        
        # update acceleration block. construct the quadtree for force modeling 
        root = leaf()
        # which we then can immediately turn into a tree 
        for p in pos: 
            root = assignParticle(p, root)
        # calculate acceleration 
        
        # multiprocessing is good for large N (> 1000), can be optimized 
        # if multi: 
#         acc = np.vstack([pool.apply_async(f_multipole, args=(p, root, 0)).get() for p in pos])
        # else: 
        acc = np.array([f_multipole(p, root, root.length, theta=theta) for p in pos])
        
        # kick velocity a half time step using new updated acceleration  
        vel = vel + acc * dt / 2 
        
        # fix positions based on boundary conditions 
        pos[pos[:, 0] > L/2, 0] = pos[pos[:, 0] > L/2, 0] - L
        pos[pos[:, 0] < -L/2, 0] = pos[pos[:, 0] < -L/2, 0] + L
        pos[pos[:, 1] > L/2, 1] = pos[pos[:, 1] > L/2, 1] - L
        pos[pos[:, 1] < -L/2, 1] = pos[pos[:, 1] < -L/2, 1] + L
        
        trees.append(root)
        
    pool.close()
    pool.join()
        
    return tpoints, np.array(xpoints), np.array(trees)

def g(x):
    return x**2

if __name__ == "__main__":
    # some constants ... 
    L = 2
    n = 10
    theta = 0 
    softening = 0.01
    
    
    particles = np.random.random((n, 4)) * 0.5 - 0.5 / 2 # L - L / 2 
    particles[:, 2] = 0 # np.random.random(n)
    particles[:, 3] = 0 # np.random.random(n) - 0.5
    # particles = np.array([[0.15, 0, 0, 0], [-0.15, -0, 0, 0]])

    tpoints, particles_list, trees = leapfrog(particles.copy())