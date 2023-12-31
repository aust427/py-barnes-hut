import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle as pkl 

import argparse 

import os 
import sys 

import copy 


def genParticles(rand_type, sig=0, mu=0.1, L=1, N=10):
    if rand_type == 1: 
        particles = np.random.random((N, 4)) * L - L/2
    else: # if rand_type == 2 : 
        particles = np.random.normal(loc=sig, scale=mu, size=(N, 4)) 

    particles[:, 0:2] = -L/2 + (particles[:, 0:2] - -L/2) % L  

    # force velocity to 0 
    particles[:, 2:] = 0

    return particles 


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


def subnode(bbox=np.array([[-1, 1],[-1, 1]]), parent_area=1, parent_mass=1): 
    n = leaf(bbox=bbox )    
    n.com = n.center 
    n.mass = parent_mass * (n.length[0]*n.length[1]) / parent_area
    
    return n
  

def solve_acc(p, node, L, epsilon=1e-3, G = 1e-6, m_scale=1e3, r_scale=1): 
    # create the 'new' box surrounding the particle 
    bound_corners = np.array([[p[0] - L/2, p[1] - L/2],
                              [p[0] + L/2, p[1] - L/2],
                              [p[0] + L/2, p[1] + L/2],
                              [p[0] - L/2, p[1] + L/2]])
        
    n_com = copy.copy(node.com )
    n_c = copy.copy(node.center)
    n_l = copy.copy(node.length)
    n_m = copy.copy(node.mass)
    
    # check how many corners are in the effective interaction volume 
    t = [not in_box(np.array([n_c[0] - n_l[0]/2, n_c[1] - n_l[1]/2]), bound_corners[[0, 2]]), 
          not in_box(np.array([n_c[0] + n_l[0]/2, n_c[1] - n_l[1]/2]), bound_corners[[0, 2]]),
          not in_box(np.array([n_c[0] + n_l[0]/2, n_c[1] + n_l[1]/2]), bound_corners[[0, 2]]),
          not in_box(np.array([n_c[0] - n_l[0]/2, n_c[1] + n_l[1]/2]), bound_corners[[0, 2]])]
    
    # if not all or no corners are in, it straddles the boundary and must be subdivided
    # this is the subdivision block. hard to change ... 
    if sum(t) < 4 and sum(t) > 0:  
        # find the minimum distance to the cell walls
        lx = min(abs(np.array([n_l[0]/2 - (n_c[0] - n_com[0]), n_l[0]/2 + (n_c[0] - n_com[0]) ])))
        ly = min(abs(np.array([n_l[1]/2 - (n_c[1] - n_com[1]), n_l[1]/2 + (n_c[1] - n_com[1]) ])))
        
        # establish the boundaries of the new 'cell' with evenly distributed mass 
        re_corners = np.array([[n_com[0] - lx, n_com[1] - ly], 
                               [n_com[0] - lx, n_com[1] + ly],
                               [n_com[0] + lx, n_com[1] + ly],
                               [n_com[0] + lx, n_com[1] - ly]])
        
        # check how many corners are in the effective interaction volume 
        t = [in_box(corner, bound_corners[[0, 2]]) for corner in re_corners]
        
        # if one corner in the volume, need to subdivide into four cells 
        if sum(t) == 1: 
            # find which corner is in the node so we can use it to split it
            b_corner = bound_corners[[in_box(corner, re_corners[[0, 2]]) for corner in bound_corners]][0]
            
            n1 = subnode(bbox=np.array([re_corners[0], b_corner - 1e-9]), 
                        parent_mass=n_m, parent_area=4 * lx * ly)
            n2 = subnode(bbox=np.array([b_corner + 1e-9, re_corners[2]]), 
                        parent_mass=n_m, parent_area=4 * lx * ly)
            n3 = subnode(bbox=np.array([[b_corner[0] + 1e-9, re_corners[0][1] ], 
                                       [re_corners[2][0], b_corner[1] - 1e-9]]), 
                        parent_mass=n_m, parent_area=4 * lx * ly)
            n4 = subnode(bbox=np.array([[re_corners[0][0], b_corner[1] + 1e-9], 
                                        [b_corner[0] - 1e-9, re_corners[2][1]]]), 
                        parent_mass=n_m, parent_area=4 * lx * ly)
            
            return np.vstack([solve_acc(p, n_, L, epsilon=epsilon, G = G, m_scale=m_scale, r_scale=r_scale) 
                              for n_ in [n1, n2, n2, n3]]).sum(axis=0)
            
        # if two corners, need to split into half 
        elif sum(t) == 2:
            # establish empty leaves to hold new nonsense 
            n1 = leaf() 
            n2 = leaf() 
            
            # node is on the right edge of the new volume 
            if t[0] and t[1]:
                n1 = subnode(bbox=np.array([re_corners[0], [bound_corners[2][0] - 1e-9, re_corners[1][1]]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
                n2 = subnode(bbox=np.array([[bound_corners[2][0] + 1e-9 , re_corners[0][1]], re_corners[2]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
            # node is on the left edge of the new volume 
            elif t[2] and t[3]:
                n1 = subnode(bbox=np.array([re_corners[0], [bound_corners[0][0] - 1e-9, re_corners[1][1]]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
                n2 = subnode(bbox=np.array([[bound_corners[0][0] + 1e-9 , re_corners[0][1]], re_corners[2]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
            # node is on the top edge of the new volume 
            elif t[0] and t[3] :               
                n1 = subnode(bbox=np.array([re_corners[0], [re_corners[2][0] , bound_corners[2][1] - 1e-9 ]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
                n2 = subnode(bbox=np.array( [[re_corners[0][0], bound_corners[2][1] + 1e-9], re_corners[2]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
            # elif t[1] and [2]: node is on bottom edge of new volume 
            else:            
                n1 = subnode(bbox=np.array([re_corners[0], [re_corners[2][0] , bound_corners[0][1] - 1e-9 ]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)
                n2 = subnode(bbox=np.array( [[re_corners[0][0], bound_corners[0][1] + 1e-9], re_corners[2]]), 
                            parent_mass=n_m, parent_area=4 * lx * ly)

            return np.vstack([solve_acc(p, n_, L, epsilon=epsilon, G = G, m_scale=m_scale, r_scale=r_scale) 
                              for n_ in [n1, n2]]).sum(axis=0)
                
    
    # check if the node's center of mass is within the box. if it is, return standard force calculation 
    # if sum(t) == 0: 
    if in_box(n_com, bound_corners[[0, 2]]) :
        return G * m_scale * node.mass * (n_com - p) / sum((n_com - p)**2 + epsilon)**(3 / 2)
    
    # now we know for sure it's not in the effective volume. we need to then see if the node's box overlaps with our periodic boundary
    n_c[n_com > p + L/2] = n_c[n_com > p + L/2] - L
    n_c[n_com < p -L/2] = n_c[n_com < p + -L/2] + L
    
    n_com[n_com > p + L/2] = n_com[n_com > p + L/2] - L
    n_com[n_com < p -L/2] = n_com[n_com < p + -L/2] + L
    

    s = leaf()
    s.com = copy.copy(n_com)
    s.center = copy.copy(n_c)
    s.mass = copy.copy(node.mass )
    s.length = copy.copy(node.length)
    
    return solve_acc(p, s, L, epsilon=epsilon, G = G, m_scale=m_scale, r_scale=r_scale)


def f_multipole(p, node, L, theta=0.5, epsilon=1e-3, m_scale=1e-3):  
    # check to see if particle is looking at itself,return 0 
    if node.mass is not None and sum((node.com - p)**2)**0.5 < 1e-10:
        return np.zeros(2) 
    
    # if a leaf node and has a particle, directly solve for gravity 
    if isinstance(node, leaf) and node.mass is not None:
        return solve_acc(p, node, L, epsilon=epsilon, m_scale=m_scale)
    # if a leaf node and doesn't have a particle, return 0
    elif isinstance(node, leaf) and node.mass is None:
        return np.zeros(2)
    
    # if a quadtree that satisfies multipole condition, solve for gravity
    if isinstance(node, QuadTree) and node.length[0] / sum((node.com - p)**2)**0.5 <= theta: 
        return solve_acc(p, node, L, epsilon=epsilon, m_scale=m_scale)
    # else if quadtree and hasn't satisfied, recursively search the tree's children 
    elif isinstance(node, QuadTree) and node.length[0] / sum((node.com - p)**2)**0.5 > theta: 
        return sum(np.array([f_multipole(p, ch, L, theta=theta, epsilon=epsilon, m_scale=m_scale) for ch in [node.ll, node.ur, node.lr, node.ul]]))
         
        
def leapfrog(r, t_start=0, t_end=10, N=1e3, L=2, theta=0.5, multi=False, epsilon=1e-3, m_scale=1e3, 
            store=False, path='./data', fname=None):
    dt = (t_end - t_start)/N

    tpoints = np.arange(t_start, t_end + dt, dt)
    xpoints = []
    trees = []
    
    acc = np.zeros((len(r), 2))
    pos = r[:, 0:2]
    vel = r[:, 2: ]
    
    pos = -L/2 + (pos - -L/2) % L  
    
    for t in tqdm(tpoints):
        # have to .copy() because we are updating the same array and that causes memory issues 
        xpoints.append(np.hstack((pos, vel, acc)))
        
        # use current acceleration to kick velocity a half time step
        vel = vel + acc * dt / 2 
        # drift the position using the kicked (half time step) velocity 
        pos = pos + vel * dt 
        
        # update acceleration block. construct the quadtree for force modeling 
        root = leaf(bbox=np.array([[-L/2, -L/2], [L/2, L/2]]))
        # which we then can immediately turn into a tree 
        for p in pos: 
            root = assignParticle(p, root)
        trees.append(root)

        # calculate acceleration 
        acc = np.array([f_multipole(p, root, root.length[0], theta=theta, epsilon=epsilon, m_scale=m_scale) 
                    for p in pos])

        # kick velocity a half time step using new updated acceleration  
        vel = vel + acc * dt / 2 
        
        # fix positions based on boundary conditions 
        # can probably change this to a modulo based statement ... 
        # pos = -L/2 + (pos - -L/2) % L
        while(abs(pos[:, 0].max()) > L/2 and abs(pos[:, 1].max()) > L/2):
            pos[pos[:, 0] > L/2, 0] = pos[pos[:, 0] > L/2, 0] - L
            pos[pos[:, 0] < -L/2, 0] = pos[pos[:, 0] < -L/2, 0] + L
            pos[pos[:, 1] > L/2, 1] = pos[pos[:, 1] > L/2, 1] - L
            pos[pos[:, 1] < -L/2, 1] = pos[pos[:, 1] < -L/2, 1] + L
            
        if store: 
            np.savetxt(fname='{}/snap_{:.3f}.txt'.format(path, t), X=pos) 
            
    return tpoints
        

if __name__ == "__main__":
    sys.setrecursionlimit(10000) 
    
    parser = argparse.ArgumentParser(description='2D Barnes-Hut particle simulation code with quasi-periodic boundary conditions.')

    parser.add_argument("-f", "--file", help="File to particles containing positions and velocity, shape: (N, 4).", metavar="FILE") 
    parser.add_argument('-L', '--L', help="Side length of the box, spanning [-L/2, L/2].", type=float)
    parser.add_argument('-N', '--N', help="Number of particles in the simulation.", type=int)
    parser.add_argument('-rand_type','--rand_type', nargs='+', type=float, help='type of random distribution to use for creating the particle data. 1 - uniform, 2 - normal ')
    parser.add_argument('-opening_angle','--opening_angle', type=float, help='the opening angle which determines if the algorithm looks at cluster-level or individual-level masses. Fiducial value is 0.5')
    parser.add_argument('-softening','--softening', type=float, help='softening length of the simulation, limiting the max gravitational interaction. Recomended value is 1e-3')
    parser.add_argument('-mass_scale','--mass_scale', type=float, help='path to particle data')
    parser.add_argument('--store', action='store_true', help='boolean to store particle results results')

    args = parser.parse_args()
        
    L = args.L if args.L is not None else 1 
    rand_type = [int(args.rand_type[0])] if args.rand_type is not None else [1]
    sig= args.rand_type[1] if args.rand_type is not None and len(args.rand_type) > 1 else 0.1
    mu = args.rand_type[2] if args.rand_type is not None and len(args.rand_type) > 2 else 0.1
    N = args.N if args.N is not None else 3
    theta = args.opening_angle if args.opening_angle is not None else 0.5 # 
    epsilon = args.softening if args.softening is not None else 1e-3 # 
    m_scale = args.mass_scale if args.mass_scale is not None else 1e3 # 
    
    particle_path = args.file
    store = args.store 
    storedir = None 
    
    if (rand_type[0] > 2 or rand_type[0] < 1) and particle_path is None: 
        print('Random generation defaulting to uniform.')
        rand_type[0] = 1
        
    if particle_path is None: 
        if rand_type[0] == 1: 
            print('Particle data generated via uniform distribution.')
        elif rand_type[0] == 2: 
            print('Particle data generated via normal distribution with the following parameters: \nsigma = {}; mu = {}.'.format(sig, mu))
        particles = genParticles(rand_type, mu=mu, sig=sig, L=L, N=N)
    else: 
        print('Particle data read from the following directory: {}'.format(particle_path))
        particles = np.loadtxt(particle_path)
        N = particles.shape[0]
        
    if store:
        storedir = 'L{}n{}'.format(L, N)
        storedir = './data/{}'.format(storedir)
    
        if not os.path.isdir(storedir):
            os.mkdir(storedir)

    print('Simulation running with the following parameters:') 
    print('L = {}; N = {}; theta = {}; softening = {}; M_scale = {}.'.format(L, N, theta, epsilon, m_scale))
    if store:
        print('Particle data will be stored in {}\n'.format(storedir)) 
    
    tpoints = leapfrog(particles, L=L, theta=theta, epsilon=epsilon, m_scale=m_scale, store=store, path=storedir)
    
    parameters = {'N': N, 'L': L, 'theta': theta, 'path': storedir, 'tpoints': tpoints}
    if store: 
        print('Parameter file written to: {}'.format(storedir))
        with open('{}/param.pkl'.format(storedir), "wb") as file:
            pkl.dump(parameters, file)
        

