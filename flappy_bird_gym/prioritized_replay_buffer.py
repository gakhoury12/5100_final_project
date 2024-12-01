# flappy_bird_gym/prioritized_replay_buffer.py

import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize the Prioritized Replay Buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Prioritization exponent (0 = uniform sampling, 1 = pure prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha

        # Pre-allocate memory for experiences
        self.states = [None] * capacity
        self.actions = [None] * capacity
        self.rewards = [None] * capacity
        self.next_states = [None] * capacity
        self.dones = [None] * capacity

        # Prioritization
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        # Buffer tracking
        self.current_index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        New experiences get the maximum priority to ensure they are sampled.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state.
            done (bool): Whether the episode is done.
        """
        index = self.current_index % self.capacity

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        # Assign max priority to new experiences
        self.priorities[index] = self.max_priority

        self.current_index += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        """
        Sample experiences from buffer with prioritized probability.

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Importance sampling weight annealing parameter.

        Returns:
            Tuple: Sampled experiences with importance sampling weights.
        """
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer")

        # Calculate sampling probabilities based on priorities
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sampling_probs = scaled_priorities / scaled_priorities.sum()

        # Sample indices with the calculated probabilities
        indices = np.random.choice(
            self.size,
            size=batch_size,
            p=sampling_probs,
            replace=False
        )

        # Calculate importance sampling weights
        weights = (self.size * sampling_probs[indices]) ** -beta
        weights /= weights.max()

        # Collect sampled experiences
        states = [self.states[idx] for idx in indices]
        actions = [self.actions[idx] for idx in indices]
        rewards = [self.rewards[idx] for idx in indices]
        next_states = [self.next_states[idx] for idx in indices]
        dones = [self.dones[idx] for idx in indices]

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights,
            indices
        )

    def update_priorities(self, indices, priorities):
        """
        Update priorities for the given indices.

        Args:
            indices (np.ndarray): Indices of experiences to update.
            priorities (np.ndarray): New priority values.
        """
        # Ensure non-zero priorities to prevent starvation
        priorities = np.abs(priorities) + 1e-6

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

        # Update max priority for new experiences
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return self.size
