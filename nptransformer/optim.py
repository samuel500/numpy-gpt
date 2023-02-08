import numpy as np

class SGD:

    def __init__(self, params, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.params = params

    def step(self):
        for tensor in self.params:
            tensor.data -= self.learning_rate * tensor.grad


class Adam:
    
    def __init__(self, params, learning_rate=3e-4, betas=(0.9, 0.95), weight_decay=0.1):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas  # momentum, decay_rate

        self.params = [
            {
                'param': param,
                'velocities': np.zeros_like(param.grad),
                'mean_squares': np.zeros_like(param.grad),
                't': 0,
                'weight_decay': weight_decay if 'Embedding' not in param.name else 0
            }
            for param in params
        ]

    def step(self):

        for tensor in self.params:
            tensor['param'].data *= (1-self.learning_rate*tensor['weight_decay'])
            tensor['t'] += 1

            tensor['velocities'] = self.betas[0] * tensor['velocities'] + (1-self.betas[0]) * tensor['param'].grad

            velocities = tensor['velocities'] / (1 - self.betas[0]**tensor['t'])  # bias correction
            tensor['mean_squares'] = self.betas[1] * tensor['mean_squares'] + (1 - self.betas[1]) * (tensor['param'].grad**2) 
            mean_squares = tensor['mean_squares'] / (1 - self.betas[1]**tensor['t'])

            # tensor['param'].data -= self.learning_rate * np.clip(velocities / (np.sqrt(mean_squares)+1e-8), -1, 1)
            tensor['param'].data -= self.learning_rate * velocities / (np.sqrt(mean_squares)+1e-8)

