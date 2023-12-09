import numpy as np
import matplotlib.pyplot as plt 

import time 
from natsort import natsorted
import moviepy.video.io.ImageSequenceClip

from simulation import leaf, QuadTree
        
def plotter(ax, node, c, alpha=0.5, zorder=1, ls='-'):
    ax.plot(
    [node.center[0] + node.length[0] / 2, 
     node.center[0] - node.length[0] / 2, 
     node.center[0] - node.length[0] / 2, 
     node.center[0] + node.length[0] / 2, 
     node.center[0] + node.length[0] / 2],
    [node.center[1] + node.length[0] / 2, 
     node.center[1] + node.length[0] / 2, 
     node.center[1] - node.length[0] / 2, 
     node.center[1] - node.length[0] / 2, 
     node.center[1] + node.length[0] / 2], 
    alpha=alpha, c=c, zorder=zorder, ls=ls)
#     plt.scatter(node.com[0], node.com[1])
    
    return 0 


def drawTree(ax, node): 
    if isinstance(node, leaf):
        if node.com is not None: 
            plotter(ax, node, 'c', 0.2)
        return 0 
    else: 
        plotter(ax, node, 'white', 0.05)
        _ = [drawTree(ax, n) for n in [node.ul, node.ur, node.ll, node.lr]]
        


def renderParticles(particles_list, trees, tpoints, save_dir):
    for particles, tree, t in zip(particles_list, trees, tpoints): 
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_facecolor("black")
        ax.set_xlabel('x [Mpc]')
        ax.set_ylabel('y [Mpc]')

        drawTree(ax, tree)
        
        ax.scatter(particles[:, 0], particles[:, 1], c='white', s=0.5, zorder=10)
        
        plt.savefig('{}/frame_{}.png'.format(save_dir, t), bbox_inches='tight', pad_inches = 0, dpi=100)
        plt.close()
            
    return


if __name__ == "__main__": 
    tstart = time.time() 

    print('[{:.2f}] Creating plot directory ...'.format(time.time() - tstart))
    new_dir = '{}/{}'.format(os.path.abspath(os.getcwd()), 'L{}n{}'.format(L, n))
    if not os.path.isdir(new_dir):
        os.mkdir(os.path.join(new_dir))

    print('[{:.2f}] Initializing multiprocessing ...'.format(time.time() - tstart))
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=n_cpu)

    mp_trees = np.array_split(trees, n_cpu)
    mp_particles = np.array_split(particles_list, n_cpu)
    mp_tpoints = np.array_split(np.arange(len(tpoints)), n_cpu)

    print('[{:.2f}] Creating plots ...'.format(time.time() - tstart))
    for i in range(n_cpu):
        pool.apply_async(renderParticles, args=(mp_particles[i], mp_trees[i], mp_tpoints[i], new_dir))

    pool.close()
    pool.join()
    print('[{:.2f}] Multiprocess concluded ...'.format(time.time() - tstart))


    print('[{:.2f}] Creating video ...'.format(time.time() - tstart))

    fps=30 #number of frames per second
    image_files = natsorted([os.path.join(new_dir,img) for img in os.listdir(new_dir) if img.endswith(".png")], reverse=False)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('test.mp4')

    print('[{:.2f}] Video created ...'.format(time.time() - tstart))

    _ = [os.remove(image_file) for image_file in image_files]
    print('[{:.2f}] Directory cleaned ...'.format(time.time() - tstart))