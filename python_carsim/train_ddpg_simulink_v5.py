#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG-PC Training Script
Direct Python-CarSim DLL Link
"""
import numpy as np
import torch
import os
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 假设文件结构:
# ./train_ddpg_PC.py
# ./env_pc.py
# ./ddpg_agent.py
# ./pycarsimlib/ (库文件)
from ddpg_agent import DDPGAgent
from env_pc import PythonCarsimEnv  # 引用上面新写的类

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"随机种子已锁定为: {seed}")

def train_ddpg_PC(
    max_episodes: int = 200,
    max_torque: float = 1500.0,
    target_slip_ratio: float = 0.1,
    target_speed: float = 100.0,
    log_dir: str = "logs_PC",
    pretrained_model_path: str = None 
):
    # --- 1. 配置 ---
    reward_weights = {
        'w_speed': 0.5,        # 提高一点速度权重
        'w_accel': 0.0,
        'w_energy': 0.05,      # 能耗惩罚
        'w_consistency': 0.0, 
        'w_beta': 0.0,       
        'w_slip': -0.03,        # 强力惩罚滑移
        'w_smooth': -0
    }
    
    hyperparams = {
        'Action Bound': 1.0,   
        'Hidden Dim': 256,
        'Actor LR': 1e-5,      
        'Critic LR': 1e-4,
        'Batch Size': 128,
        'Elite Ratio': 0.3,    
        'Elite Capacity': 20000,
        'Noise Scale': 0.5,    
        'Min Noise': 0.05,
        'Noise Decay': 0.998,  
    }
    
    # 日志
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"Python_Carsim_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    print(f"训练日志: {log_path}")

    # --- 2. 初始化环境 ---
    # [关键修复] 这里必须是 CarSim 的 Database 路径 (包含 Runs, Data, Extensions 等文件夹)
    # 你之前写的 "Program Files" 路径通常是安装路径，不是数据路径。
    # 请检查 Public Documents 或者你自己的工作区
    CARSIM_DB_DIR = r"E:\CarSim2022\CarSim2022.1_Prog\RL" 
    
    # 检查路径是否存在
    if not os.path.exists(CARSIM_DB_DIR):
        print(f"❌ 错误: 找不到 CarSim 数据库路径: {CARSIM_DB_DIR}")
        print("请修改代码中的 CARSIM_DB_DIR 为包含 'Runs' 和 'Data' 文件夹的目录")
        return

    env = PythonCarsimEnv(
        carsim_db_dir=CARSIM_DB_DIR,
        sim_time_s=10.0,       
        delta_time_s=0.01,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        target_speed=target_speed,
        vehicle_type="normal_vehicle", # 确保与 pycarsimlib 里的配置一致
        reward_weights=reward_weights
    )
    
    # --- 3. 初始化 Agent ---
    agent = DDPGAgent(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        action_bound=hyperparams['Action Bound'],
        hidden_dim=hyperparams['Hidden Dim'],
        actor_lr=hyperparams['Actor LR'],
        critic_lr=hyperparams['Critic LR'],
        batch_size=hyperparams['Batch Size'],
        elite_ratio=hyperparams['Elite Ratio'],
        elite_capacity=hyperparams['Elite Capacity']   
    )
    
    # 加载预训练
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"🔄 加载预训练模型: {pretrained_model_path}")
        agent.load_model(pretrained_model_path)
        noise_scale = 0.1 
    else:
        print("🆕 从零开始训练")
        noise_scale = hyperparams['Noise Scale']

    best_episode_reward = -float('inf') 
    min_noise = hyperparams['Min Noise']
    noise_decay = hyperparams['Noise Decay']

    print("\n========== Start Pure DDPG Training ==========")
    
    try:
        for episode in range(max_episodes):
            # 1. Reset (这一步会重启 CarSim)
            state, info = env.reset()
            agent.reset_noise() 
            
            episode_reward = 0  
            reward_stats = { "R_Spd": [], "R_Slp": [], "R_Eng": [] }
            current_episode_memory = []
            
            critic_grads = []
            actor_grads = []
            
            while True:
                # 2. Select Action
                action = agent.select_action(state, noise_scale=noise_scale)

                # 3. Step
                next_state, reward, done, info = env.step(action)
                
                # 4. Push & Train
                agent.push(state, action, reward, next_state, done)
                current_episode_memory.append((state, action, reward, next_state, done))
                
                # DDPG
                c_loss, a_loss, c_grad, a_grad = agent.train_step()
                
                if c_loss != 0:
                    critic_grads.append(c_grad)
                    actor_grads.append(a_grad)
                
                state = next_state
                episode_reward += reward
                
                # Log Stats
                for k in reward_stats:
                    if k in info: reward_stats[k].append(info[k])
                
                if done: break
            
            # --- Episode End ---
            
            # Summary stats
            sum_rewards = {k: np.sum(v) for k, v in reward_stats.items()}
            avg_c = np.mean(critic_grads) if critic_grads else 0
            avg_a = np.mean(actor_grads) if actor_grads else 0

            # Tensorboard
            writer.add_scalar('Loss/Critic', c_loss, episode)
            writer.add_scalar('Loss/Actor', a_loss, episode)
            writer.add_scalar('Train/Reward', episode_reward, episode)
            writer.add_scalar('Train/Noise', noise_scale, episode)
            if avg_c > 0:
                writer.add_scalar('Grad/Critic', avg_c, episode)
                writer.add_scalar('Grad/Actor', avg_a, episode)

            # 打印 Summary (覆盖掉 step 的打印)
            print(f"Ep {episode}| Rw: {episode_reward:.0f} | Ns: {noise_scale:.2f} | "
                  f"Spd: {sum_rewards['R_Spd']:.0f} | Slp: {sum_rewards['R_Slp']:.0f} | "
                  f"Grad: {avg_c:.3f}/{avg_a:.3f}")

            # 精英策略
            is_elite = False
        if episode_reward > best_episode_reward*0.8 and episode_reward >=0:
            is_elite = True
            writer.add_scalar('Train/Is_Elite', 1, episode)
            print(f"🌟 [精英]! Reward: {episode_reward:.1f}")
            for trans in current_episode_memory:
                agent.push_elite(*trans)
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                agent.save_model(os.path.join("best_model_save", f"Python_Carsim_{current_time}.pt"))
                print(f"🌟 [新纪录] ! Reward: {episode_reward:.1f}")
        else:
            writer.add_scalar('Train/Is_Elite', 0, episode)
            
            # 噪声衰减
            noise_scale = max(min_noise, noise_scale * noise_decay)

    except KeyboardInterrupt:
        print("人为停止训练...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保关闭 CarSim，否则下次可能起不来
        env.close()
        agent.save_model(os.path.join(log_path, "final_model.pt"))
        print("资源已释放，训练结束。")

if __name__ == "__main__":
    setup_seed(42)
    # 注意：不需要传入 pretrained_model_path=None，因为这是默认值
    train_ddpg_PC()
