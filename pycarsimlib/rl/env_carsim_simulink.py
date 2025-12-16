#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
速度追踪版 (精简UDP通信版)
更新点：
1. UDP接收仅包含: [Vx, Ax, S_FL, S_FR, S_RL, S_RR, Yaw] 共7维数据。
2. 索引映射更新。
"""
import socket
import struct
import numpy as np
import time

class CarsimSimulinkEnv:
    def __init__(
        self,
        sim_time_s: float = 20.0,
        delta_time_s: float = 0.01,
        max_torque: float = 1500.0,
        target_slip_ratio: float = 0.1, 
        target_speed: float = 100.0,
        send_port: int = 9202,
        recv_port: int = 8087,
        ip_addr: str = "127.0.0.1",
        reward_weights: dict = None
    ):
        self.sim_time_s = sim_time_s
        self.delta_time_s = delta_time_s
        self.max_steps = int(sim_time_s / delta_time_s)
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.target_speed = target_speed
        
        # 默认权重配置
        default_weights = {
            'w_speed': 0.15,      # 速度奖励
            'w_accel': 2.0,       # 加速度奖励
            'w_energy': -0.0,     # 能耗
            'w_consistency': -0.0,# 一致性
            'w_yaw': -0.0,        # 横摆
            'w_slip': -0.1,       # 滑移率
            'w_smooth': -0.0      # 平滑
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        self.ip_addr = ip_addr
        self.send_port = send_port
        self.recv_port = recv_port
        
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.udp_socket.bind((self.ip_addr, self.recv_port))
        except OSError:
            print(f"端口 {self.recv_port} 被占用，尝试复用...")
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind((self.ip_addr, self.recv_port))
        
        self.udp_socket.settimeout(2.0)
        self.last_torque = np.zeros(4)
        self.filter_alpha = 0.6 
        
        # [修改] 状态维度现在是 7
        # 顺序: [Vx, Ax, S_FL, S_FR, S_RL, S_RR, Yaw]
        self.state_dim = 7 
        self.action_dim = 4
        self.current_step = 0
        self.comm_error_count = 0  # 通信错误计数器

    def reset(self):
        """ 重置环境：强制制动直到车辆完全静止 (分段制动策略) """
        self.current_step = 0
        self.last_torque = np.zeros(4)
        
        # 1. 清空 UDP 缓冲区
        self.udp_socket.setblocking(False)
        try:
            while True:
                self.udp_socket.recv(4096)
        except BlockingIOError:
            pass
        self.udp_socket.setblocking(True)
        self.udp_socket.settimeout(2.0)

        # 2. 发送探测包
        # 注意：如果你之前修改为了 6d (带step)，这里请记得保持一致。这里按你当前提供的 5d 写。
        probe_cmd = struct.pack("5d", 0.0, 0.0, 0.0, 0.0, 0.0)
        try:
            self.udp_socket.sendto(probe_cmd, (self.ip_addr, self.send_port))
            initial_state = self._receive_state_raw()
        except:
            initial_state = np.zeros(self.state_dim)

        current_vx_kmh = initial_state[0]
        
        # 如果本来就是停着的，直接返回
        if abs(current_vx_kmh) < 0.1: # 阈值设小一点
            return self._normalize_state(initial_state), {}

        # 3. 强制制动循环
        print(f"\n--- 执行强制制动 (当前车速: {current_vx_kmh:.1f} km/h) ---")
        
        # 安全计数器
        safety_max_steps = int(5000 * max(abs(current_vx_kmh), 1.0))

        # 阈值配置
        LOCK_SPEED_THRESHOLD = 3.0  # 低于 5km/h 进入“死踩模式”
        
        for i in range(safety_max_steps):
            try:
                # === 策略核心逻辑 ===
                
                # 1. 接收当前状态 (先收再发，或者是根据上一帧状态决定这一帧动作)
                cmd_data = None
                # --- 阶段判定 ---
                if abs(current_vx_kmh) > LOCK_SPEED_THRESHOLD:
                    # >>> 高速阶段：点刹 + 中等压力 <<<
                    brake_pressure = abs(10.0 -abs(current_vx_kmh)*0.1)
                    brake_duration = 30 
                    release_duration = 70
                    cycle_len = brake_duration + release_duration
                    step_in_cycle = i % cycle_len
                    if step_in_cycle < brake_duration:
                        # 刹车
                        cmd_data = struct.pack("7d", 0.0, 0.0, 0.0, 0.0, brake_pressure, 0.0, 0.0)
                    else:
                        # 松开
                        cmd_data = struct.pack("7d", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    # >>> 低速阶段：死踩到底 + 禁止松开 <<<
                    # 速度很低了，不会偏航了，直接给最大压力刹停，防止蠕行
                    brake_pressure = 13.0 -abs(current_vx_kmh)
                    brake_duration = 80 
                    release_duration = 20 
                    cycle_len = brake_duration + release_duration
                    step_in_cycle = i % cycle_len
                    if step_in_cycle < brake_duration:
                        # 刹车
                        cmd_data = struct.pack("7d", 0.0, 0.0, 0.0, 0.0, brake_pressure, 0.0, 0.0)
                    else:
                        # 松开
                        cmd_data = struct.pack("7d", 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0)
                
                # 发送指令
                self.udp_socket.sendto(cmd_data, (self.ip_addr, self.send_port))
                
                # 接收反馈
                state = self._receive_state_raw()
                current_vx_kmh = state[0] # 更新当前速度
                yaw = state[6]
                
                # 打印进度
                if i % 20 == 0:
                    status_str = "Pulsing" if abs(current_vx_kmh) > LOCK_SPEED_THRESHOLD else "LOCKING"
                    print(f"\r[{status_str}] V={current_vx_kmh:.2f} km/h | Yaw={yaw:.2f}", end="")
                
                # 判定刹停标准 (非常严格)
                if abs(current_vx_kmh) < 0.3: # 0.05 km/h 几乎就是静止
                    print(f"\n✅ 制动完成。最终车速: {current_vx_kmh:.4f} km/h")
                    break
            
            except socket.timeout:
                print("\n⚠️ [Warning] 连接超时")
                break
        else:
            print("\n⚠️ [Warning] 达到最大制动步数")
        
        # 4. 最后彻底松开，准备开始 Episode
        release_cmd = struct.pack("5d", 0.0, 0.0, 0.0, 0.0, 0.0)
        try:
            self.udp_socket.sendto(release_cmd, (self.ip_addr, self.send_port))
            time.sleep(0.05) 
            final_state = self._receive_state_raw()
        except:
            final_state = np.zeros(self.state_dim)
            
        return self._normalize_state(final_state), {}

    def step(self, action):
        #action = np.clip(action, -1.0, 1.0)
        #normalized_action = (action + 1.0) / 2.0
        #target_torque = normalized_action * self.max_torque 
        target_torque = action * self.max_torque 
        
        real_torque = target_torque 
        step_val = float(self.current_step)
        throttle = 1.0

        #test_torque = np.array([300+((step_val*0.01)**1)*30, 300+((step_val*0.01)**1)*30, 500+((step_val*0.01)**1.3)*30, 500+((step_val*0.01)**1.3)*30])
        send_data = struct.pack("7d", 
                                real_torque[0], 
                                real_torque[1], 
                                real_torque[2], 
                                real_torque[3], 
                                0.0, 
                                step_val,
                                throttle)
        
        try:
            self.udp_socket.sendto(send_data, (self.ip_addr, self.send_port))
            
            # 接收精简后的数据
            raw_state = self._receive_state_raw()
            next_state = self._normalize_state(raw_state)
            
            reward, reward_details = self._calculate_reward(raw_state, real_torque, self.last_torque)
            
            self.last_torque = real_torque
            self.current_step += 1

            details_str = " | ".join([f"{k}: {v:.3f}" for k, v in reward_details.items()])
            if self.current_step % 100 == 0:
                print(f"step: {self.current_step}, reward: {reward:.4f}, vx: {raw_state[0]:.2f} km/h || {details_str}")

            done = self.current_step >= self.max_steps
            
            # info 字典中的索引
            info = {
                "slip_error": np.mean(np.abs(raw_state[2:6] - self.target_slip_ratio)), 
                "vx": raw_state[0], 
                "ax": raw_state[1], 
                **reward_details
            }
            return next_state, reward, done, info
            
        except socket.error:
            self.comm_error_count += 1
            print(f"\n⚠️ [Warning] 通信错误，返回空状态。当前错误计数: {self.comm_error_count}/10")
            
            # 当错误次数达到10次时，终止训练
            if self.comm_error_count >= 10:
                print("\n❌ [Error] 通信错误次数达到10次，终止训练！")
                # 抛出异常以便训练脚本捕获并终止
                raise RuntimeError("通信错误次数达到上限，终止训练")
                
            return np.zeros(self.state_dim), 0, True, {}

    def _receive_state_raw(self):
        """
        [核心修改]
        接收 7 个 double，共 56 bytes
        顺序: Vx, Ax, S_FL, S_FR, S_RL, S_RR, Yaw
        """
        expected_bytes = 7 * 8 # 56
        try:
            data, _ = self.udp_socket.recvfrom(1024)
            if len(data) >= expected_bytes:
                # 仅解包前 56 字节
                return np.array(struct.unpack("7d", data[:expected_bytes]), dtype=np.float32)
            else:
                return np.zeros(self.state_dim, dtype=np.float32)
        except socket.timeout:
            return np.zeros(self.state_dim, dtype=np.float32)

    def _normalize_state(self, raw_state):
        """
        [修改] 针对 7维数据进行归一化
        raw_state: [Vx, Ax, S1, S2, S3, S4, Yaw]
        """
        norm_state = raw_state.copy()
        
        # 1. 速度 Vx (索引0) -> /100
        norm_state[0] = raw_state[0] / 100.0 
        
        # 2. 加速度 Ax (索引1) -> /5.0 (归一化到 -1~1 范围)
        norm_state[1] = raw_state[1] / 1.0    
        
        # 3. 滑移率 Slips (索引2-5) ->
        norm_state[2:6] = raw_state[2:6]/100
        
        # 4. 横摆角速度 Yaw (索引6) -> /50.0
        norm_state[6] = raw_state[6] / 10.0   
        
        return norm_state

    def _calculate_reward(self, state, current_torque, last_torque):
        if np.all(state == 0): return 0.0, {}
        
        # [修改] 基于新索引提取物理量
        vx = state[0]           # km/h
        ax = state[1]           # m/s^2
        slip = state[2:6] # 假设 Simulink 发来的是 %, 否则去掉 *0.01
        beta = state[6]     # deg偏航角
        
        w = self.weights 
        
        # 1. Speed
        r_speed = w['w_speed'] * (vx) 
        
        # 2. Accel
        if ax > 0.0:
            r_accel = w['w_accel'] * (ax)
        else:
            r_accel = 0.0
        
        # 3. Energy
        torque_norm = current_torque / self.max_torque
        r_energy = w['w_energy'] * np.sum(np.square(torque_norm))
        
        # 4. Slip (前后轮分离)
        # 提取前后轮滑移率
        slip_FL = slip[0] # FL
        slip_FR = slip[1] # FR
        slip_RL = slip[2] # RL
        slip_RR = slip[3] # RR
        
        # 设定阈值
        threshold_front = 0.10  # 前轮 10%
        threshold_rear  = 0.15  # 后轮 15%
        
        # 计算超出的部分 (小于阈值则为0)
        excess_FL = np.maximum(0.0, slip_FL - threshold_front)
        excess_FR = np.maximum(0.0, slip_FR - threshold_front)
        excess_RL = np.maximum(0.0, slip_RL - threshold_rear)
        excess_RR = np.maximum(0.0, slip_RR - threshold_rear)
        
        if vx > 3: # 只有车动起来才算滑移惩罚
            r_slip_FL = w['w_slip'] * excess_FL
            r_slip_FR = w['w_slip'] * excess_FR
            r_slip_RL = w['w_slip'] * excess_RL
            r_slip_RR = w['w_slip'] * excess_RR
            
            # 总滑移奖励
            r_slip = r_slip_FL + r_slip_FR + r_slip_RL + r_slip_RR
        else:
            r_slip = 0.0

        # 5. Consistency
        diff_front = abs(current_torque[0] - current_torque[1])
        diff_rear = abs(current_torque[2] - current_torque[3])
        r_consistency = w['w_consistency'] * ((diff_front + diff_rear) / self.max_torque)
        
        # 6. Smooth
        delta_torque = (current_torque - last_torque) / self.max_torque
        smooth_l2 = np.mean(np.square(delta_torque))
        r_smooth = w['w_smooth'] * smooth_l2 * 10.0

        # 7. Yaw
        r_beta = w['w_beta'] * abs(beta)

        total_reward = r_speed + r_accel + r_energy + r_consistency + r_slip + r_smooth + r_beta

        reward_details = {
            "R_Speed": r_speed,
            "R_Accel": r_accel,
            "R_Energy": r_energy,
            "R_Consis": r_consistency,
            "R_Slip": r_slip,
            "R_Smooth": r_smooth,
            "R_Beta": r_beta
        }

        return total_reward, reward_details
    
    # [修改] 维度更新
    def get_state_dim(self): return 7 
    def get_action_dim(self): return 4
    def close(self): self.udp_socket.close()