import numpy as np
import matplotlib.pyplot as plt

class leaf():
    def __init__(self, bbox=np.array([[-1, 1],[-1, 1]]), depth=1):
        # if the leaf has a particle ... 
        self.particle = None 
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
        self.mass = 0
        self.com = 0 
        
        # initialize leaf nodes ... 
        self.ll = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0] - oleaf.length / 2, oleaf.center[0]], 
                                 [oleaf.center[1] - oleaf.length / 2, oleaf.center[1]]]),)
        self.ul = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0] - oleaf.length / 2, oleaf.center[0]], 
                                 [oleaf.center[1], oleaf.center[1] + oleaf.length / 2]]))
        self.ur = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0], oleaf.center[0] + oleaf.length / 2], 
                                 [oleaf.center[1], oleaf.center[1] + oleaf.length / 2]]))
        self.lr = leaf(depth = self.depth + 1, bbox=np.array([[oleaf.center[0], oleaf.center[0] + oleaf.length / 2], 
                                 [oleaf.center[1] - oleaf.length / 2, oleaf.center[1]]]))
                

def plotter(node, c, alpha=0.1):
    plt.plot(
    [node.center[0] + node.length / 2, node.center[0] - node.length / 2, node.center[0] - node.length / 2, 
     node.center[0] + node.length / 2, node.center[0] + node.length / 2],
    [node.center[1] + node.length / 2, node.center[1] + node.length / 2, node.center[1] - node.length / 2, 
     node.center[1] - node.length / 2, node.center[1] + node.length / 2], 
    alpha=alpha, c=c)
    
    return 0 


def drawTree(node): 
    if isinstance(node, leaf):
        if node.particle is not None: 
            plotter(node, 'c', 0.2)
            return 0 
    else: 
        plotter(node, 'white', 0.05)
        _ = [drawTree(n) for n in [node.ul, node.ur, node.ll, node.lr]]

        
def quadAssign(p, node): 
    # grab the angle to assign to a quadrant (in degrees). arctan2 returns [0, 180] and [-180, 0]
    # need to take the angle of the particle relative to the evaluated node
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
    if isinstance(node, leaf) and node.particle is None:
        node.particle = p 
    # if a leaf has a particle, make it a tree and re-assign particles
    elif isinstance(node, leaf) and node.particle is not None: 
        p_old = node.particle 
        node = QuadTree(node)
        node = assignParticle(p, node)  
        node = assignParticle(p_old, node)
    # if tree, find out where in the tree the particle should live 
    else: 
        node.mass = node.mass + 1 
        node = quadAssign(p, node)
        
    return node
    

# function holding our quantities we want to get d/dt for 
# so we can use them in RK function 
def f(r, t): 
    x = r[:, 0]
    y = r[:, 1]
    xdot = r[:, 2]
    ydot = r[:, 3]
    
    a = xdot
    b = ydot
    da = np.ones(len(x)) * 0
    db = np.ones(len(x)) * 0

    return np.array([a, b, da, db], float).T


def RK4(r, t_start=0, t_end=10, N=1e4):
    h = 0.1 # (t_end - t_start)/N

    tpoints = np.arange(t_start, t_end, h)
    xpoints = []
    trees = []

    for t in tpoints:
        # have to .copy() because we are updating the same array and that causes memory issues 
        xpoints.append(r.copy())
        
        # we start with a simple leaf (root) 
        root = leaf()
        # which we then can immediately turn into a tree 
        for p in r: 
            root = assignParticle(p[0:2], root)
        trees.append(root)

        k1 = h*f(r, t)
        k2 = h*f(r + 0.5*k1, t + 0.5*h)
        k3 = h*f(r + 0.5*k2, t + 0.5*h)
        k4 = h*f(r + k3, t + h)
        
        r += (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # boundary condition fixing 
        r[r[:, 0] > 1, 0] = r[r[:, 0] > 1, 0] - 2
        r[r[:, 0] < -1, 0] = r[r[:, 0] < -1, 0] + 2
        r[r[:, 1] > 1, 1] = r[r[:, 1] > 1, 1] - 2
        r[r[:, 1] < -1, 1] = r[r[:, 1] < -1, 1] + 2
        
    return tpoints, np.array(xpoints), trees


def leapfrog():
    return 0 