import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class leaf():
    def __init__(self, bbox=np.array([[-1, 1],[-1, 1]]), depth=1):
        # if the leaf has a particle ... 
        self.mass = None
        self.com = None
        # center position of quadrant 
        self.center = np.array([bbox[0].sum() / 2, bbox[1].sum() / 2])
        # length of bounding box, same for x- and y- axes 
        self.length = bbox[0, 1] - bbox[0, 0]
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
        
        # initialize leaf nodes ... 
        self.ll = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0] - oleaf.length / 2, oleaf.center[0]], 
                                 [oleaf.center[1] - oleaf.length / 2, oleaf.center[1]]]),)
        self.ul = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0] - oleaf.length / 2, oleaf.center[0]], 
                                 [oleaf.center[1], oleaf.center[1] + oleaf.length / 2]]))
        self.ur = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0], oleaf.center[0] + oleaf.length / 2], 
                                 [oleaf.center[1], oleaf.center[1] + oleaf.length / 2]]))
        self.lr = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0], oleaf.center[0] + oleaf.length / 2], 
                                 [oleaf.center[1] - oleaf.length / 2, oleaf.center[1]]]))
   
        
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
    

def solve_acc(p, node, epsilon=0.0, G = 1e-3, m_scale=1, r_scale=1): 
    # do psudo periodic boundary conditions here 
    return G * m_scale * node.mass * (node.com - p + epsilon) / sum((node.com - p + epsilon)**2)**(3 / 2)


def f_multipole(p, node, theta=0.0):  
    # check to see if particle is looking at itself,return 0 
    if node.mass is not None and sum((node.com - p)**2)**0.5 < 1e-10:
        return np.zeros(2) 
    
    # if a leaf node and has a particle, directly solve for gravity 
    if isinstance(node, leaf) and node.mass is not None:
        return solve_acc(p, node)
    # if a leaf node and doesn't have a particle, return 0
    elif isinstance(node, leaf) and node.mass is None:
        return np.zeros(2)
    
    # if a quadtree that satisfies multipole condition, solve for gravity
    if isinstance(node, QuadTree) and node.length / sum((node.com - p)**2)**0.5 <= theta: 
        return solve_acc(p, node)
    # else if quadtree and hasn't satisfied, recursively search the tree's children 
    elif isinstance(node, QuadTree) and node.length / sum((node.com - p)**2)**0.5 > theta: 
        return sum(np.array([f_multipole(p, ch, theta) for ch in [node.ll, node.ur, node.lr, node.ul]]))
        

def leapfrog(r, t_start=0, t_end=10, N=1e4, L=2):
    dt = 0.01 # (t_end - t_start)/N

    tpoints = np.arange(t_start, t_end, dt)
    xpoints = []
    trees = []
    
    acc = np.zeros((len(r), 2))
    pos = r[:, 0:2]
    vel = r[:, 2: ]

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
        acc = np.array([f_multipole(p, root, 0) for p in pos])
        
        # kick velocity a half time step using new updated acceleration  
        vel = vel + acc * dt / 2 
        
        # fix positions based on boundary conditions 
        pos[pos[:, 0] > L/2, 0] = pos[pos[:, 0] > L/2, 0] - L
        pos[pos[:, 0] < -L/2, 0] = pos[pos[:, 0] < -L/2, 0] + L
        pos[pos[:, 1] > L/2, 1] = pos[pos[:, 1] > L/2, 1] - L
        pos[pos[:, 1] < -L/2, 1] = pos[pos[:, 1] < -L/2, 1] + L
        
        trees.append(root)
        
    return tpoints, np.array(xpoints), np.array(trees)


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