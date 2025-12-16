#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG训练脚本 - 精英经验回放版 (支持梯度监控)
"""
import numpy as np
import torch
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
from pycarsimlib.rl.ddpg_agent import DDPGAgent
from pycarsimlib.rl.env_carsim_simulink import CarsimSimulinkEnv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"随机种子已锁定为: {seed}")

def get_reference_torque(current_step, max_torque):
    """
    根据 current_step 返回 0 ~ max_torque 之间的扭矩
    """
    return np.array([0+((current_step*0.01)**0.5)*180, 0+((current_step*0.01)**0.5)*180, 500+((current_step*0.01)**1.2)*32, 500+((current_step*0.01)**1.2)*32])

def train_ddpg_simulink(
    max_episodes: int = 10000,
    max_torque: float = 1500.0,
    target_slip_ratio: float = 0.1,
    target_speed: float = 100.0,
    log_dir: str = "logs",
    pretrained_model_path: str = None 
):
    # --- 1. 奖励权重配置 ---
    reward_weights = {
        'w_speed': 0.015,
        'w_accel': 0.0,
        'w_energy': 0.3,
        'w_consistency': -0.0,
        'w_beta': -0.0,
        'w_slip': -3.0,
        'w_smooth': -0.0
    }
    
    hyperparams = {
        'Action Bound': 1.0,
        'Hidden Dim': 256,
        'Gamma': 0.99,
        'Tau': 5e-3,
        'Buffer Capacity': 10000,
        'Actor LR': 1e-5,
        'Critic LR': 1e-4,
        'Batch Size': 128,
        'Elite Ratio': 0.4,   
        'Elite Capacity': 50000,
        'Noise Scale': 0.5,
        'Min Noise': 0.05,
        'Noise Decay': 0.995,
        'Pretrained': pretrained_model_path if pretrained_model_path else "None"
    }
    
    # --- 2. 日志设置 ---
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"elite_ddpg_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    
    md_table = "### Reward Coefficients\n| Key | Value |\n|---|---|\n"
    for k, v in reward_weights.items():
        md_table += f"| {k} | {v} |\n"
    writer.add_text("Configuration", md_table, 0)

    md_table = "\n### Hyperparameters\n| Key | Value |\n|---|---|\n"
    for k, v in hyperparams.items():
        md_table += f"| {k} | {v} |\n"
        
    writer.add_text("Configuration/Parameters", md_table, 0)
    
    print(f"训练日志将保存至: {log_path}")

    # --- 3. 初始化环境 ---
    env = CarsimSimulinkEnv(
        sim_time_s=10.0,       
        delta_time_s=0.01,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        target_speed=target_speed,
        reward_weights=reward_weights, 
        send_port=9202,
        recv_port=8087
    )
    
    # --- 4. 初始化 Agent ---
    agent = DDPGAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        action_bound=hyperparams['Action Bound'],
        hidden_dim=hyperparams['Hidden Dim'],
        gamma = hyperparams['Gamma'],
        tau = hyperparams['Tau'],
        actor_lr=hyperparams['Actor LR'],
        critic_lr=hyperparams['Critic LR'],
        buffer_capacity = hyperparams['Buffer Capacity'],
        batch_size=hyperparams['Batch Size'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        elite_ratio=hyperparams['Elite Ratio'],
        elite_capacity=hyperparams['Elite Capacity']   
    )
    
    # 加载预训练模型
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n🔄 正在加载预训练模型: {pretrained_model_path}")
        try:
            agent.load_model(pretrained_model_path)
            print("✅ 模型加载成功！将基于此模型继续训练。")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return
    else:
        print("\n🆕 未指定预训练模型，将从头开始训练。")
    
    best_episode_reward = -float('inf') 
    noise_scale = hyperparams['Noise Scale']
    min_noise = hyperparams['Min Noise']
    noise_decay = hyperparams['Noise Decay']

    print("\n========== 开始训练 ==========")
    
    for episode in range(max_episodes):
        state, info = env.reset()
        agent.reset_noise() 
        
        episode_reward = 0  
        
        # 统计变量
        reward_stats = {
            "R_Speed": [], "R_Accel": [], "R_Energy": [],
            "R_Consis": [], "R_Slip": [], "R_Smooth": [], "R_Beta": []
        }
        current_episode_memory = []
        slip_errors = []
        speed_errors = []
        
        # 梯度统计
        critic_grads = []
        actor_grads = []
        
        while True:
            
            action = agent.select_action(state, noise_scale=noise_scale)
            next_state, reward, done, info = env.step(action)
            
            # 存入缓冲区
            agent.push(state, action, reward, next_state, done)
            current_episode_memory.append((state, action, reward, next_state, done))
            
            # 训练步
            c_loss, a_loss, c_grad, a_grad = agent.train_step()
            
            # 记录梯度 (仅当发生训练时)
            if c_loss != 0.0:
                critic_grads.append(c_grad)
                actor_grads.append(a_grad)
            
            state = next_state
            episode_reward += reward
            
            if "slip_error" in info: slip_errors.append(info["slip_error"])
            if "speed_error" in info: speed_errors.append(abs(info["speed_error"]))
            
            for key in reward_stats:
                if key in info:
                    reward_stats[key].append(info[key])
            
            if done:
                break
        
        # --- 回合结束 ---
        final_speed = info.get('vx', 0)

        # 统计分项
        sum_rewards = {k: np.sum(v) if v else 0.0 for k, v in reward_stats.items()}
        for k, v in sum_rewards.items():
            writer.add_scalar(f'Rewards_Details_Sum/{k}', v, episode)

        # 计算平均梯度
        avg_c_grad = np.mean(critic_grads) if critic_grads else 0.0
        avg_a_grad = np.mean(actor_grads) if actor_grads else 0.0

        # 精英回放逻辑
        is_elite = False
        if episode_reward > best_episode_reward*0.8 and episode_reward >=0:
            is_elite = True
            writer.add_scalar('Train/Is_Elite', 1, episode)
            print(f"🌟 [精英]! Reward: {episode_reward:.1f}")
            for trans in current_episode_memory:
                agent.push_elite(*trans)
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                agent.save_model(os.path.join("best_model_save", f"elite_ddpg_{current_time}.pt"))
                print(f"🌟 [新纪录] ! Reward: {episode_reward:.1f}")
        else:
            writer.add_scalar('Train/Is_Elite', 0, episode)

        # TensorBoard
        if c_loss != 0.0:
            writer.add_scalar('Loss/Critic', c_loss, episode)
            writer.add_scalar('Loss/Actor', a_loss, episode)
            writer.add_scalar('Grad/Critic', avg_c_grad, episode)
            writer.add_scalar('Grad/Actor', avg_a_grad, episode)
            
        writer.add_scalar('Train/Reward', episode_reward, episode)
        
        # 噪声衰减
        noise_scale = max(min_noise, noise_scale * noise_decay)
        writer.add_scalar('Train/Noise_Scale', noise_scale, episode)

        # 打印信息
        print(f"Ep {episode+1}/{max_episodes} | Total: {episode_reward:.0f} | "
            f"Best: {best_episode_reward:.0f} | "
            f"Spd: {sum_rewards['R_Speed']:.0f} | "
            #f"Accel: {sum_rewards['R_Accel']:.0f} | "
            f"Energy: {sum_rewards['R_Energy']:.0f} | "
            #f"Beta: {sum_rewards['R_Beta']:.0f} | "
            #f"Consis: {sum_rewards['R_Consis']:.0f} | "
            f"Slp: {sum_rewards['R_Slip']:.0f} | "
            #f"Smth: {sum_rewards['R_Smooth']:.0f} | "
            f"Speed: {final_speed:.1f} | "
            f"Grad(C/A): {avg_c_grad:.3f}/{avg_a_grad:.3f}")

    agent.save_model(os.path.join(log_path, "final_model.pt"))
    writer.close()
    env.close()
    print("训练完成")

if __name__ == "__main__":
    setup_seed(42)
    os.makedirs("logs", exist_ok=True)
    MODEL_PATH = "best_model_save/elite_ddpg_20251216_114644.pt"
    train_ddpg_simulink(
        pretrained_model_path=None
    )