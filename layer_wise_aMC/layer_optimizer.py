import torch
import numpy as np

######
# Adaptive Layer wise MC optimizer
# Takes model parameters as well as the following hyperparameters:
# init_sigma = Initial value of the standard deviation of the move-proposal distribution
# epsilon = Rate at which parameters of the move-proposal distribution are modified
# beta = Inverse temperature for Metropolis acceptance probability. Fixed to infinity in the paper.
# n_reset = Number of consecutive rejected moves before a parameter's move-proposal distribution is reset to initial values
# sigma_decay = Scaling factor for sigma upon n_reset consecutive rejected moves. Fixed to 0.95 in the paper.
# minibatch = if using a single batch per epoch, speed up by storing the previous loss
######


class aMC(torch.optim.Optimizer):
    def __init__(self, params, init_sigma,epsilon = 0., beta = "inf", n_reset = 100, sigma_decay = .95, minibatch = False):
        

        defaults = dict(init_sigma = init_sigma, epsilon = epsilon, beta = beta, n_reset = n_reset ,sigma_decay = sigma_decay, minibatch = minibatch)
        super().__init__(params,defaults)

        self._params = self.param_groups[0]['params']
        if len(self.param_groups) != 1:
            raise ValueError("Currently doesn't support per-parameter options "
                             "(parameter groups)")

        #initialize optimizer parameters
        
        # NOTE: aMC currently has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        
        if not minibatch:
            state['best_loss'] = None
        
        state['minibatch'] = minibatch
        state['sigma'] = init_sigma
        state['n_cr'] = [0] * len(self._params)
        state['mus'] = []
        for p in self._params:
            state['mus'].append(torch.zeros_like(p))

    def sample_noise(self, i):
        state = self.state[self._params[0]]
        noise = torch.normal(mean =  state['mus'][i], std = state['sigma'])
        return noise

    def _add_noise(self, noise, i):
        self._params[i].add_(noise)

    def _subtract_noise(self, noise, i):
        self._params[i].subtract_(noise)

    @torch.no_grad()
    def step(self, closure):
        
        state = self.state[self._params[0]]
        group = self.param_groups[0]

        for i in range(len(self._params)):
            #Calculate initial loss
            if not state['minibatch']:
                if state['best_loss'] == None:
                    state['best_loss'] = closure()
                init_loss = state['best_loss']
            else:
                init_loss = closure()
            
            #sample noise, evaluate new loss function, and accept/reject move
            noise = self.sample_noise(i)

            self._add_noise(noise, i)

            new_loss = closure()


            move_accepted = False
            cost = new_loss - init_loss
            if cost <= 0:
                move_accepted = True
            elif group['beta'] != "inf":    
                if np.random.rand() < torch.exp(-cost * group['beta']):
                    move_accepted = True


            if move_accepted:
                state['n_cr'][i] = 0
                if not state['minibatch']:
                    state['best_loss'] = new_loss
                
                #Shift mean based on noise
                if group['epsilon'] != 0:
                    state['mus'][i] += group['epsilon'] * (noise - state['mus'][i])

            else:
                #reset net to previous state
                self._subtract_noise(noise, i)
                state['n_cr'][i] += 1

                if state['n_cr'][i] == group['n_reset']:
                    state['mus'][i].zero_()

                    state['n_cr'][i] = 0
        
        return state['best_loss']
                    