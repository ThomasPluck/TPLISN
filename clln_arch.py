import numpy as np

class ParallelIsingNodeEdgeCLLN:
    
    def __init__(
        self,
        HEIGHT,
        WIDTH,
        beta = 1.0,
        sigma = 0.1,
        lr = 1e-3,
        mean_field = True,
        dt = 0.1,
    ):
        
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        
        self.beta = beta
        self.mean_field = mean_field
        self.lr = lr
        self.dt = dt
        
        self.free_node_states = np.random.randn(HEIGHT,WIDTH) * sigma
        self.clamped_node_states = np.random.randn(HEIGHT,WIDTH) * sigma
        self.free_horiz_states = np.random.randn(HEIGHT,WIDTH-1) * sigma
        self.clamped_horiz_states = np.random.randn(HEIGHT,WIDTH-1) * sigma
        self.free_verti_states = np.random.randn(HEIGHT-1,WIDTH) * sigma
        self.clamped_verti_states = np.random.randn(HEIGHT-1,WIDTH) * sigma
        
        self.horiz_weights = np.random.randn(HEIGHT,WIDTH-1) * sigma
        self.verti_weights = np.random.randn(HEIGHT-1,WIDTH) * sigma
        
    def step(self, unclamped, clamped = None):
        
        # Node fields
        free_node_field = np.zeros_like(self.free_node_states)
        free_node_field[:,1:] += self.horiz_weights * self.free_horiz_states
        free_node_field[:,:-1] -= self.horiz_weights * self.free_horiz_states
        free_node_field[1:,:] += self.verti_weights * self.free_verti_states
        free_node_field[:-1,:] -= self.verti_weights * self.free_verti_states
        
        clamped_node_field = np.zeros_like(self.free_node_states)
        clamped_node_field[:,1:] += self.horiz_weights * self.clamped_horiz_states
        clamped_node_field[:,:-1] -= self.horiz_weights * self.clamped_horiz_states
        clamped_node_field[1:,:] += self.verti_weights * self.clamped_verti_states
        clamped_node_field[:-1,:] -= self.verti_weights * self.clamped_verti_states
        
        # Update nodes
        free_node_prob_plus = np.tanh(self.beta * free_node_field)
        clamped_node_prob_plus = np.tanh(self.beta * clamped_node_field)
        if self.mean_field:
            self.free_node_states += (free_node_prob_plus - self.free_node_states) * self.dt
            self.clamped_node_states += (clamped_node_prob_plus - self.clamped_node_states) * self.dt
        else:
            self.free_node_states = np.sign(free_node_prob_plus + np.random.uniform(-1,+1,self.free_node_states.shape))
            self.clamped_node_states = np.sign(clamped_node_prob_plus + np.random.uniform(-1,+1,self.clamped_node_states.shape))
        
        if clamped is not None:
            self.free_node_states = np.where(unclamped != 0.0, unclamped, self.free_node_states)
            self.clamped_node_states = np.where(clamped != 0.0, clamped, self.clamped_node_states)
            
        # np.clip(self.free_node_states, -5, 5, out=self.free_node_states)
        # np.clip(self.clamped_node_states, -5, 5, out=self.clamped_node_states)
        
        # Edge fields
        free_horiz_fields = self.horiz_weights * self.free_node_states[:,1:]
        free_horiz_fields -= self.horiz_weights * self.free_node_states[:,:-1]
        clamped_horiz_fields = self.horiz_weights * self.clamped_node_states[:,1:]
        clamped_horiz_fields -= self.horiz_weights * self.clamped_node_states[:,:-1]
        free_verti_fields = self.verti_weights * self.free_node_states[1:,:]
        free_verti_fields -= self.verti_weights * self.free_node_states[:-1,:]
        clamped_verti_fields = self.verti_weights * self.clamped_node_states[1:,:]
        clamped_verti_fields -= self.verti_weights * self.clamped_node_states[:-1,:]
        
        # Update edges
        free_horiz_prob_plus = np.tanh(self.beta * free_horiz_fields)
        clamped_horiz_prob_plus = np.tanh(self.beta * clamped_horiz_fields)
        free_verti_prob_plus = np.tanh(self.beta * free_verti_fields)
        clamped_verti_prob_plus = np.tanh(self.beta * clamped_verti_fields)
        if self.mean_field:
            self.free_horiz_states += (free_horiz_prob_plus - self.free_horiz_states) * self.dt
            self.clamped_horiz_states += (clamped_horiz_prob_plus - self.clamped_horiz_states) * self.dt
            self.free_verti_states += (free_verti_prob_plus - self.free_verti_states) * self.dt
            self.clamped_verti_states += (clamped_verti_prob_plus - self.clamped_verti_states) * self.dt
        else:
            self.free_horiz_states = np.sign(free_horiz_prob_plus + np.random.uniform(-1,+1,self.free_horiz_states.shape))
            self.clamped_horiz_states = np.sign(clamped_horiz_prob_plus + np.random.uniform(-1,+1,self.clamped_horiz_states.shape))
            self.free_verti_states = np.sign(free_verti_prob_plus + np.random.uniform(-1,+1,self.free_verti_states.shape))
            self.clamped_verti_states = np.sign(clamped_verti_prob_plus + np.random.uniform(-1,+1,self.clamped_verti_states.shape))
        
    def train(self,signal):
        
        # Compute correlations
        # free_horiz_corrs = self.free_node_states[:,1:] * self.free_horiz_states - self.free_node_states[:,:-1] * self.free_horiz_states
        # clamped_horiz_corrs = self.clamped_node_states[:,1:] * self.clamped_horiz_states - self.clamped_node_states[:,:-1] * self.clamped_horiz_states
        # free_verti_corrs = self.free_node_states[1:,:] * self.free_verti_states - self.free_node_states[:-1,:] * self.free_verti_states
        # clamped_verti_corrs = self.clamped_node_states[1:,:] * self.clamped_verti_states - self.clamped_node_states[:-1,:] * self.clamped_verti_states
        free_horiz_corrs = (self.free_node_states[:, :-1] - self.free_node_states[:, 1:]) ** 2
        clamped_horiz_corrs = (self.clamped_node_states[:, :-1] - self.clamped_node_states[:, 1:]) ** 2
        free_verti_corrs = (self.free_node_states[:-1, :] - self.free_node_states[1:, :]) ** 2
        clamped_verti_corrs = (self.clamped_node_states[:-1, :] - self.clamped_node_states[1:, :]) ** 2
        
        # Update the weights
        self.horiz_weights += signal * self.lr * (free_horiz_corrs - clamped_horiz_corrs)
        self.verti_weights += signal * self.lr * (free_verti_corrs - clamped_verti_corrs)
        self.horiz_weights *= 1 - self.lr
        self.verti_weights *= 1 - self.lr
        
    def render_state(self):
        
        canvas = np.zeros((2*self.HEIGHT-1,2*self.WIDTH-1))
        
        canvas += np.kron(self.free_node_states,np.array([[1,0],[0,0]]))[:-1,:-1]
        canvas[:,1:] += np.kron(self.free_horiz_states,np.array([[1,0],[0,0]]))[:-1,:]
        canvas[1:,:] += np.kron(self.free_verti_states,np.array([[1,0],[0,0]]))[:,:-1]
        
        return canvas
    
    def render_weights(self):
        
        canvas = np.zeros((2*self.HEIGHT-1,2*self.WIDTH-1))
        
        canvas[:,1:] += np.kron(self.horiz_weights,np.array([[1,0],[0,0]]))[:-1,:]
        canvas[1:,:] += np.kron(self.verti_weights,np.array([[1,0],[0,0]]))[:,:-1]
        
        return canvas