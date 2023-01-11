import numpy as np
import torch
from collections import deque

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size, state_shape, action_shape, n):
        transition_dtype = np.dtype([('timestep', np.int32), ('state', np.float32, state_shape), ('action', np.float32, action_shape), ('reward', np.float32), ('done', np.bool_)])
        self.blank_trans = (0, np.zeros(state_shape, dtype=np.float32), 0, 0.0, False)

        self.n = n
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.zeros((size,), dtype=transition_dtype)  # Build structured array
        
        self.max = 1  # Initial max value to return (1 = 1^ω)

  # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data):
        self.data[self.index] = data  # Store data in underlying data structure
        if self.full or  self.index >= self.n:
            self._update_index(self.index - self.n + self.tree_start, self.max)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    

    def append(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state)) ###
    

    def sample(self, sample_size:int=32):
        states, actions, rewards, next_states, dones = list(), list(), list(), list(), list()
        
        sample_indices = np.random.choice(len(self.buffer), size=sample_size, replace=False)

        for index in sample_indices:
            sample = self.buffer[index]
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            dones.append(sample[3])
            next_states.append(sample[4])

        states, actions, rewards, next_states, dones = \
            (torch.from_numpy(np.array(datas)).float().to(self.device) for datas in (states, actions, rewards, next_states, dones))

        return states, actions, rewards, next_states, dones


class PrioritisedReplayMemory():
    def __init__(self, device, discount, multi_step, priority_weight, priority_exponent, capacity, state_shape, action_shape):
        self.blank_trans = (0, np.zeros(state_shape, dtype=np.float32), 0, 0.0, False)
        self.device = device
        self.capacity = capacity
        self.discount = discount
        self.n = multi_step
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
        self.transitions = SegmentTree(capacity, state_shape, action_shape, multi_step)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, done):
        self.transitions.append((self.t, state, action, reward, done))  # Store new transition with maximum priority
        self.t = 0 if done else self.t + 1  # Start new episodes with t = 0

    # Returns the transitions with blank states where appropriate
    def _get_transitions(self, idxs):
        transition_idxs = np.arange(0, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(1, 1 + self.n):  # e.g. 4 5 6
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
        transitions[blank_mask] = self.blank_trans
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
            if np.all(probs > 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        # Create un-discretised states and nth next states
        all_states = transitions['state']
        states = torch.tensor(all_states[:, 0].copy(), device=self.device, dtype=torch.float32)
        next_states = torch.tensor(all_states[:, self.n].copy(), device=self.device, dtype=torch.float32)
        # Discrete actions to be used as index
        actions = torch.tensor(transitions['action'][:, 0].copy(), dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = torch.tensor(transitions['reward'][:, 0:-1].copy(), dtype=torch.float32, device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        # Mask for non-terminal nth next states
        done = torch.tensor(transitions['done'][:, self.n].copy(), dtype=torch.float32, device=self.device)
        return probs, idxs, tree_idxs, states, actions, R, next_states, done

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, returns, next_states, done = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, done, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[self.current_idx]
        state = transitions['state']  # Agent will turn into batch
        self.current_idx += 1

        return state