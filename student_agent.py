import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np
from collections import deque
import random
import numpy as np
import math

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=1):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())
  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)
class DuelNoisyCategoricalQNet(nn.Module):
    def __init__(self, n_frames, n_actions, num_atoms=51, v_min=-1, v_max=1):
        super(DuelNoisyCategoricalQNet, self).__init__()
        
        self.n_actions = n_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)  # Discretized interval

        # Shared CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(n_frames, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        feature_dim = 64 * 7 * 7 
        
        # Noisy networks for value and advantage streams (each predicting num_atoms distributions)
        self.fc_value1 = NoisyLinear(feature_dim, 512)
        self.fc_value2 = NoisyLinear(512, num_atoms)
        
        self.fc_advantage1 = NoisyLinear(feature_dim, 512)
        self.fc_advantage2 =  NoisyLinear(512, n_actions * num_atoms)  # Advantage output

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.view(batch_size, -1)

        value = self.fc_value2(F.relu(self.fc_value1(x))).view(batch_size, 1, self.num_atoms)  # State value distribution
        advantage = self.fc_advantage2(F.relu(self.fc_advantage1(x))).view(batch_size, self.n_actions, self.num_atoms)  # Advantage distribution

        # Combining value and advantage distributions
        q_distribution = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return F.softmax(q_distribution, dim=2)  # Normalize as a probability distribution over atoms
    def reset_noise(self):
        self.fc_value1.reset_noise()
        self.fc_value2.reset_noise()
        self.fc_advantage1.reset_noise()
        self.fc_advantage2.reset_noise()

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.4, n_step=5, gamma=0.99,beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.n_step = n_step  # Multi-step return length
        self.gamma = gamma  # Discount factor
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Stores priorities
        self.n_step_buffer = deque(maxlen=n_step)  # Temporary buffer for multi-step transitions
        self.beta = beta
    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            state, action = self.n_step_buffer[0][:2]
            next_state, done = self.n_step_buffer[-1][3:]
            reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])

            # Default priority (until updated via TD error)
            priority = max(self.priorities, default=1.0)  
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
            if done: # finish the rest of the episode and clear
                self.n_step_buffer.popleft()
                while len(self.n_step_buffer) > 0:
                    state, action = self.n_step_buffer[0][:2]
                    reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
                    priority = max(self.priorities, default=1.0)  
                    self.buffer.append((state, action, reward, next_state, done))
                    self.priorities.append(priority)
                    self.n_step_buffer.popleft()

    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()  # Normalize probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        #indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        #weights = np.array([1] * batch_size)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        #for idx, error in zip(indices, td_errors):
        #    self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
        pass

    def __len__(self):
        return len(self.buffer)
# Do not modify the input of the 'act' function and the '__init__' function. 

def log_softmax(softmax_values):
    """Convert batched softmax values to log_softmax values."""
    log_sum_exp = torch.log(softmax_values.sum(dim=-1, keepdim=True))  # Compute log(sum(exp(x))) for each batch
    log_softmax_values = torch.log(softmax_values) - log_sum_exp  # Apply transformation per batch
    return log_softmax_values

class RainbowDQN:
    def __init__(self, n_frames, action_size, gamma=0.99, tau=8000, capacity=10000, lr=0.00025, steps=10, batch_size=128, n_step=3, beta=0.4,learn_start=50000,num_atoms=51):
        self.device = torch.device("cpu") 
        
        self.n_frames = n_frames
        self.action_size = action_size
        self.learn_start = learn_start
        self.n_step = n_step
        self.memory = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6, n_step=n_step, gamma=gamma, beta=beta)
        self.v_min = -1
        self.v_max = 20
        # Move networks to CUDA
        self.q_net = DuelNoisyCategoricalQNet(n_frames, action_size,num_atoms=num_atoms,v_min=self.v_min,v_max=self.v_max).to(self.device)
        self.target_net = DuelNoisyCategoricalQNet(n_frames, action_size,num_atoms=num_atoms,v_min=self.v_min,v_max=self.v_max).to(self.device)
        self.target_net.train()
        self.q_net.train()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr,eps=1e-4)
        self.gamma = gamma
        self.tau = tau
        self.train_steps = steps
        self.batch_size = batch_size
        self.counter = 0
        
        self.num_atoms = num_atoms
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)  # Atom support


    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.q_net(state)  # Get probability distributions over atoms
            q_values = torch.sum(action_probs * self.z_atoms, dim=-1)  # Compute expected Q-values
            action = q_values.argmax(dim=1).cpu().item()  # Select best action
        return action
    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
    
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = tuple(map(list, zip(*batch)))
    
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
    
        # **Distributional Bellman Update**
        with torch.no_grad():
            # Convert distribution to expected Q-values
            pns = self.q_net(next_states) # bin probabilities
            dns = self.z_atoms.expand_as(pns) * pns # distribution
            best_action = dns.sum(2).argmax(1) # calculate best action from expected value
            self.target_net.reset_noise() 
            pns_a = self.target_net(next_states)[range(self.batch_size),best_action]
            target_z = rewards + (self.gamma ** self.n_step) * self.z_atoms.unsqueeze(0) * (1 - dones)
            target_z = torch.clamp(target_z, self.v_min, self.v_max)
            # Compute categorical distribution targets using L2 projection
            b = (target_z - self.v_min) / self.delta_z  # Normalize
            l = b.floor().long()
            u = b.ceil().long()
            
            l = torch.clamp(l, 0, self.num_atoms - 1)
            u = torch.clamp(u, 0, self.num_atoms - 1)
            m = states.new_zeros(self.batch_size, self.num_atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.num_atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
    
        # Compute current Q-distribution
        current_dist = self.q_net(states)[range(self.batch_size), actions.squeeze()]
        # Compute KL-divergence loss
        loss = -(m * log_softmax(current_dist)).sum(dim=-1)  # Cross-entropy loss
        self.optimizer.zero_grad()
        (weights * loss).mean().backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1)
        self.optimizer.step()
        # Update priorities based on KL divergence
        td_errors = loss.detach().cpu().numpy().flatten()
        td_errors = np.clip(td_errors, -1.0, 1.0)
        self.memory.update_priorities(indices, td_errors)
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.counter += 1
        if self.counter % self.train_steps == 0:
            self.q_net.reset_noise()
        if self.counter >= self.learn_start:
            self.memory.beta = min(1,self.memory.beta + 0.6 / 300000) # anneal beta to 1
            if self.counter % self.train_steps == 0:
                self.train()
            if self.counter % self.tau == 0:
                self.update_target_network()
    def recover(self):
        self.q_net= torch.load('q_net.pth')
        self.target_net.load_state_dict(self.q_net.state_dict())
n_frames = 4
agent = RainbowDQN(n_frames, 12, tau=8000, gamma=0.99, lr=0.0000625, steps=4, capacity=100000, batch_size=32, n_step=5,learn_start=100000)
agent.recover()
agent.q_net.eval()
import cv2
def convert_image(image_in):
    """
    Convert a NumPy image array to 84x84 grayscale.
    Assumes the input image is either RGB or grayscale.
    """
    # Convert to grayscale if it's RGB
    if len(image_in.shape) == 3 and image_in.shape[2] == 3:
        image_in = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    image_resized = cv2.resize(image_in, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image_resized, axis=-1)  # Add a channel dimension

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = []

    def act(self, observation):
        observation = convert_image(observation).squeeze()
        if len(self.frames) == 0:
            self.frames = np.array([observation]*4)
        else:
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1] = observation
        #print(self.frames.shape)
        return agent.get_action(self.frames.copy())
if __name__ == "__main__":
    #torch.save(agent.q_net.cpu(), "q_net.pth")
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env.reset()
    done = False
    obs = None
    student = Agent()
    while not done:
        if obs is None :
            action = random.randrange(12)
        else:
            action = student.act(obs)
        obs, reward, done, info = env.step(action)
        env.render()
    print(f"Final score: {info['score']}")