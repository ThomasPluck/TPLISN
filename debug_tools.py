import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Assuming your data is shape (time_steps, height, width)
def arrays_to_mp4(data, filename, fps=10, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    
    # Set consistent color scale
    if vmin is None: vmin = -abs(data[0]).max()
    if vmax is None: vmax = abs(data[0]).max()
    
    im = ax.imshow(data[0], cmap='bwr', vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    
    def animate(frame):
        im.set_array(data[frame])
        out = data[frame][4,4] - data[frame][0,0]
        ax.set_title(f'Step {frame}, Out = {out:.3f}')
        return [im]
    
    ani = animation.FuncAnimation(fig, animate, frames=len(data), 
                                  interval=1000//fps, blit=True)
    ani.save(filename, writer='ffmpeg', fps=fps)
    plt.close()