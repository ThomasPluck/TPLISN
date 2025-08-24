import random
from tqdm import tqdm

import matplotlib.pyplot as plt

from data import *
from parallel_arch import ParallelIsingNodeEdgeCLLN
from summing_arch import ParallelSummingIsingNodeEdgeCLLN
from debug_tools import arrays_to_mp4

# Experiment Params
TRIALS = 500
STEPS_PER_TRIAL = 10000
nudging_factor = 0.5

model = ParallelSummingIsingNodeEdgeCLLN(
    HEIGHT=4,
    WIDTH=4,
    beta=1.0,
    lr=1e-5,
    sigma=1e-2,
    dt=0.001
)

accuracy = []

for t in tqdm(range(TRIALS)):
    
    index = random.choice([0,1,2,3])
    label = 2 * (index > 0 and index < 3) - 1
    
    unclamped_data = unclamped[index].copy()
    
    converged = False
    count = 0
    
    while not converged:
    
        oplus = model.free_node_states[2,2]
        ominus = model.free_node_states[0,0]
        output = oplus - ominus
        
        # Borrowing protocol from CLLN
        oplusc = oplus + nudging_factor/2 * (label - output)
        ominusc = ominus - nudging_factor/2 * (label - output)
        
        clamped_data = unclamped[index].copy()
        clamped_data[2,2] = oplusc
        clamped_data[0,0] = ominusc
        
        old_horiz = model.free_horiz_states.copy()
        old_verti = model.free_verti_states.copy()
        
        model.step(unclamped=unclamped_data,clamped=clamped_data)
        model.train(1)
        
        edge_change = max(np.max(np.abs(model.free_horiz_states - old_horiz)), 
                np.max(np.abs(model.free_verti_states - old_verti)))

        count += 1

        if edge_change < 1e-6 or count > STEPS_PER_TRIAL:
            converged = True
            
    accuracy.append(output * label > 0)
            
    if t == TRIALS-1:
        
        state_vid_data = []
        weight_vid_data = []
        
        for s in range(1000):
            
            model.step(unclamped=unclamped_data,clamped=clamped_data)
            state_vid_data.append(model.render_state())
            
        # arrays_to_mp4(state_vid_data,f"parallel_state_{t+1}.mp4")
        # print("Free process rendered...")
        fig, ax = plt.subplots()
        ax.imshow(model.render_state(), cmap='bwr')
        ax.set_title(f'Model State')
        plt.savefig(f'model_state_{t+1}.png')
        print("States plotted..")
        plt.close()
        
        fig, ax = plt.subplots()
        max = np.maximum(abs(model.horiz_weights),abs(model.verti_weights.T)).max()
        im = ax.imshow(model.render_weights(), cmap='bwr', vmin=-max, vmax=max)
        plt.colorbar(im)
        ax.set_title(f'Parallel Weights at trial {t+1}')
        plt.savefig(f'parallel_weights_{t+1}.png')
        print("Weights plotted...")
        plt.close()
        
        processed = []
        for i in range(len(accuracy)//50):
            processed.append(sum(accuracy[50*i:50*(i+1)])/50)
        
        fig, ax = plt.subplots()
        ax.plot(processed)
        ax.set_title(f'Training Accuracy')
        plt.savefig(f'parallel_accuracy_{t+1}.png')
        print("Accuracy plotted..")
        plt.close()