import numpy as np

"""
the replay buffer here is basically from the openai baselines code
"""
class replay_buffer:
    def __init__(self, agent_management, sample_func):
        self.agent_management = agent_management
        self.T = agent_management.max_timesteps
        self.size = agent_management.buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {
            'phase': np.empty([self.size, self.T + 1, self.agent_management.phase_dim]),        # phase
            'obs': np.empty([self.size, self.T + 1, self.agent_management.ob_length]),          # obs
            'ag': np.empty([self.size, self.T + 1, self.agent_management.phase_dim]),           # achieved_goal
            'g': np.empty([self.size, self.T + 1, self.agent_management.phase_dim]),            # desired_goal
            'action': np.empty([self.size, self.T + 1, self.agent_management.action_space.n]),  # action
            'label': np.empty([self.size, self.T + 1, 1])           # label : 有效和无效
        }

    def store_experiences(self, batch):
        ma_phase, mb_obs, mb_ag, mb_g, mb_actions, mb_label = batch
        batch_size = mb_obs.shape[0]    #TODO 修改

        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['phase'][idxs] = ma_phase
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.buffers['label'][idxs] = mb_label

        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)    # 抽样batch_size
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)  # inc 为batch_size
        elif self.current_size < self.size:  # 低于buffer size存取
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

class gan_sampler:
    def __init__(self):
        super(gan_sampler, self).__init__()

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        transitions = []
        return transitions
