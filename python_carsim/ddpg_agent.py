"""
深度确定性策略梯度(DDPG)智能体实现 (标准DDPG + 精英经验回放 + 梯度监控)

优化内容:
1. [Elite] 精英经验回放: 混合采样机制。
2. [Base] OU噪声、梯度裁剪。
3. [Monitor] 梯度监控: 返回梯度范数用于调试。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Optional, Any
import copy

from networks import PolicyNet, QValueNet
from replay_buffer import ReplayBuffer
import torch.nn.functional as F

# OU 噪声类 (保持不变)
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float = 1.0,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_capacity: int = 100000,
        batch_size: int = 128,
        device: str = None,
        # 精英回放参数
        elite_ratio: float = 0.4,
        elite_capacity: int = 5000
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.action_dim = action_dim
        
        # 精英回放设置
        self.elite_ratio = elite_ratio
        
        # 初始化网络 (Actor + Single Critic)
        self._initialize_networks(state_dim, action_dim, hidden_dim)
        self._initialize_optimizers(actor_lr, critic_lr)
        
        # 缓冲区
        self.buffer = ReplayBuffer(buffer_capacity)
        self.elite_buffer = ReplayBuffer(elite_capacity)
        
        # OU 噪声
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim), 
            std_deviation=float(0.2) * np.ones(action_dim)
        )
    
    def _initialize_networks(self, state_dim, action_dim, hidden_dim):
        # 策略网络 (Actor)
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, self.action_bound).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        
        # 价值网络 (Single Critic) - 回归单网络
        self.critic = QValueNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
    
    def _initialize_optimizers(self, actor_lr, critic_lr):
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def reset_noise(self):
        self.noise.reset()
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if noise_scale > 0.0:
            ou_noise = self.noise() * noise_scale
            action = np.clip(action + ou_noise, 0.0, self.action_bound)
        
        return action
    
    def push(self, state, action, reward, next_state, done):
        """存入普通缓冲区"""
        self.buffer.push(state, action, reward, next_state, done)

    def push_elite(self, state, action, reward, next_state, done):
        """存入精英缓冲区"""
        self.elite_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, net, target_net):
        with torch.no_grad():
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
    
    def train_step(self) -> Tuple[float, float, float, float]:
        """
        执行一步训练
        返回: (critic_loss, actor_loss, critic_grad_norm, actor_grad_norm)
        """
        # 1. 混合采样逻辑
        if len(self.elite_buffer) < self.batch_size or len(self.buffer) < self.batch_size:
            if len(self.buffer) < self.batch_size:
                return 0.0, 0.0, 0.0, 0.0
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        else:
            n_elite = int(self.batch_size * self.elite_ratio)
            n_normal = self.batch_size - n_elite
            
            s_e, a_e, r_e, ns_e, d_e = self.elite_buffer.sample(n_elite)
            s_n, a_n, r_n, ns_n, d_n = self.buffer.sample(n_normal)
            
            states = s_e + s_n
            actions = a_e + a_n
            rewards = r_e + r_n
            next_states = ns_e + ns_n
            dones = d_e + d_n
        
        # 转换为 Tensor
        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_tensor = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # ----------------------------
        # 2. 更新 Critic (Single Critic)
        # ----------------------------
        critic_loss, critic_grad = self._update_critic(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
        
        # ----------------------------
        # 3. 更新 Actor (标准DDPG每步更新)
        # ----------------------------
        actor_loss, actor_grad = self._update_actor(states_tensor)
        
        # 4. 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        
        return critic_loss, actor_loss, critic_grad, actor_grad
    
    def _update_critic(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor) -> Tuple[float, float]:
        with torch.no_grad():
            # DDPG: 这里的 next_actions 不需要加噪声
            next_actions = self.target_actor(next_states_tensor)
            
            # 计算目标 Q 值
            target_q = self.target_critic(next_states_tensor, next_actions)
            
            # Bellman 方程
            td_target = rewards_tensor + self.gamma * (1 - dones_tensor) * target_q
        
        current_q = self.critic(states_tensor, actions_tensor)
        
        critic_loss = nn.MSELoss()(current_q, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        
        # 梯度裁剪与监控
        norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        
        self.critic_opt.step()
        
        return critic_loss.item(), norm.item()
    
    def _update_actor(self, states_tensor) -> Tuple[float, float]:
        # Actor 目标：最大化 Critic 对自己动作的评分
        actor_actions = self.actor(states_tensor)
        actor_loss = -self.critic(states_tensor, actor_actions).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        
        # 梯度裁剪与监控
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        
        self.actor_opt.step()
        
        return actor_loss.item(), norm.item()
    
    def save_model(self, filepath: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt'])
        
        # 同步目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())