# env_pc.py
from attr import s
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional, Any
import os
from pycarsimlib.core import CarsimManager

class PythonCarsimEnv:
    """
    CarSim Python 直连
    """
    
    # ================= CarSim 变量名配置 =================
    # 1. 控制信号 (对应 CarSim Generic Import)
    IMP_THROTTLE = "IMP_THROTTLE_ENGINE" # 油门 (0-1)
    IMP_BRAKE = "IMP_PCON_BK"        # 制动压力 (MPa)
    
    # 4轮驱动扭矩 (Nm) - 你的控制目标
    IMP_TORQUE_L1 = "IMP_MY_OUT_D1_L" 
    IMP_TORQUE_R1 = "IMP_MY_OUT_D1_R" 
    IMP_TORQUE_L2 = "IMP_MY_OUT_D2_L" 
    IMP_TORQUE_R2 = "IMP_MY_OUT_D2_R" 

    # 2. 状态信号 (对应 CarSim Generic Export)
    EXP_VX = "Vx"         # 纵向车速 (km/h)
    EXP_AX = "Ax"         # 纵向加速度 (g, 需注意单位转换)
    EXP_AVZ = "AVz"       # 横摆角速度 (deg/s)
    
    # 轮速 (RPM) - 用于计算滑移率
    EXP_WHEEL_L1 = "AVy_L1"
    EXP_WHEEL_R1 = "AVy_R1"
    EXP_WHEEL_L2 = "AVy_L2"
    EXP_WHEEL_R2 = "AVy_R2"

    # ====================================================

    def __init__(
        self,
        carsim_db_dir: str,
        vehicle_type: str = "normal_vehicle", # 对应 pycarsimlib 配置的 key
        sim_time_s: float = 10.0,
        delta_time_s: float = 0.01,
        max_torque: float = 1500.0,
        target_slip_ratio: float = 0.15,
        target_speed: float = 100.0,
        reward_weights: dict = None
    ):
        self.carsim_db_dir = carsim_db_dir
        self.vehicle_type = vehicle_type
        self.sim_time_s = sim_time_s
        self.delta_time = timedelta(seconds=delta_time_s)
        self.max_steps = int(sim_time_s / delta_time_s)
        
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.target_speed = target_speed
        
        # 奖励权重
        default_weights = {
            'w_speed': 0.1, 'w_accel': 0.0, 'w_energy': 0.0,
            'w_consistency': 0.0, 'w_beta': 0.0, 'w_slip': -1.0, 'w_smooth': 0.0
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        # 物理常数
        self.wheel_radius = 0.362 # m (需根据车型修改)
        self.veh_bf = 1.600; # 前轮距 m
        self.veh_br = 1.740; # 后轮距 m

        # 内部变量
        self.cm: Optional[CarsimManager] = None
        self.current_step = 0
        self.last_torque = np.zeros(4)
        
        # 维度: [Vx, Ax, S_L1, S_R1, S_L2, S_R2, YawRate]
        self.state_dim = 7
        self.action_dim = 4

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        重启仿真环境
        """
        # 1. 关闭旧的 Solver
        self.close()
        
        # 2. 实例化新的 Manager (相当于点击 Run)
        # 注意: 这里会加载 DLL 并初始化
        try:
            self.cm = CarsimManager(
                carsim_db_dir=self.carsim_db_dir,
                vehicle_type=self.vehicle_type
            )
        except Exception as e:
            raise RuntimeError(f"无法启动 CarSim Solver, 请检查路径和License: {e}")

        self.current_step = 0
        self.last_torque = np.zeros(4)
        
        # 3. 运行第 0 步 (初始化状态)
        init_action = self._get_zero_action_dict()
        obs, _, _ = self.cm.step(action=init_action, delta_time=self.delta_time)
        
        # 4. 解析状态
        state = self._parse_observation(obs)
        norm_state = self._normalize_state(state)
        
        return norm_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步 RL
        """
        # 1. 动作处理 (Agent输出 [0, 1] -> 物理扭矩)
        action = np.clip(action, 0.0, 1.0)
        target_torque = action * self.max_torque
        
        # 2. 构造 CarSim 输入字典
        control_inputs = {
            self.IMP_THROTTLE: 0.0,      # 不踩油门踏板，直接控扭矩
            self.IMP_BRAKE: 0.0,
            
            # 4轮扭矩
            self.IMP_TORQUE_L1: target_torque[0],
            self.IMP_TORQUE_R1: target_torque[1],
            self.IMP_TORQUE_L2: target_torque[2],
            self.IMP_TORQUE_R2: target_torque[3],
        }
        
        # 3. 物理仿真一步
        # CarsimManager.step 返回: (observation_dict, terminated, time_sec)
        obs, terminated, _ = self.cm.step(action=control_inputs, delta_time=self.delta_time)
        
        # 4. 状态解析与计算
        raw_state = self._parse_observation(obs)
        next_state = self._normalize_state(raw_state)
        
        # 5. 奖励计算
        reward, r_details = self._calculate_reward(raw_state, target_torque, self.last_torque)
        
        self.last_torque = target_torque
        self.current_step += 1
        slip_error = np.mean(np.abs(raw_state[2:6] - self.target_slip_ratio))
        
        # 6. 结束判定
        done = (self.current_step >= self.max_steps) or terminated
        
        # 7. 构造详细 Info (用于看板显示)
        # raw_state 索引: 0:Vx, 1:Ax, 2:SL1, 3:SR1, 4:SL2, 5:SR2, 6:Yaw
        info = {
            # --- 车辆状态 ---
            "vx": raw_state[0],       # km/h
            "ax": raw_state[1],       # g
            "yaw": raw_state[6],      # deg/s
            
            # --- 滑移率 (原始值) ---
            "slip_L1": raw_state[2],
            "slip_R1": raw_state[3],
            "slip_L2": raw_state[4],
            "slip_R2": raw_state[5],
            
            # --- 动作 (扭矩 Nm) ---
            "trq_L1": target_torque[0],
            "trq_R1": target_torque[1],
            "trq_L2": target_torque[2],
            "trq_R2": target_torque[3],
            
            # --- 奖励细节 ---
            **r_details
        }
        # ================= [修改结束] =================
        
        # 注释掉原来的简单 print，交给外部看板处理
        # if self.current_step % 10 == 0: ...
        
        # 打印 (同行刷新)
        #if self.current_step % 10 == 0:
            #dtl_str = " | ".join([f"{k}: {v:.2f}" for k, v in r_details.items()])
            #print(f"\rStep: {self.current_step} | Rw: {reward:.4f} | Vx: {raw_state[0]:.1f} | SlipErr: {slip_error:.2f}|| {dtl_str}", end="", flush=True)
        
        if done: print() # 换行
            
        return next_state, reward, done, info

    def close(self):
        if self.cm:
            self.cm.close()
            self.cm = None

    # ================= 辅助函数 =================
    
    def _get_zero_action_dict(self):
        return {
            self.IMP_THROTTLE: 0.0, self.IMP_BRAKE: 0.0,
            self.IMP_TORQUE_L1: 0.0, self.IMP_TORQUE_R1: 0.0,
            self.IMP_TORQUE_L2: 0.0, self.IMP_TORQUE_R2: 0.0
        }

    def _parse_observation(self, obs: Dict[str, float]) -> np.ndarray:
        """从字典解析物理值并计算滑移率"""
        vx = obs.get(self.EXP_VX, 0.0) # km/h
        ax = obs.get(self.EXP_AX, 0.0) # g? 假设 CarSim 输出是 g
        avz = obs.get(self.EXP_AVZ, 0.0) * np.pi / 180
        
        # 轮速 RPM -> 线速度 m/s
        rpm_to_ms = (2 * np.pi / 60.0) * self.wheel_radius
        v_L1 = obs.get(self.EXP_WHEEL_L1, 0.0) * rpm_to_ms
        v_R1 = obs.get(self.EXP_WHEEL_R1, 0.0) * rpm_to_ms
        v_L2 = obs.get(self.EXP_WHEEL_L2, 0.0) * rpm_to_ms
        v_R2 = obs.get(self.EXP_WHEEL_R2, 0.0) * rpm_to_ms
        
        # 计算滑移率 S = (V_wheel - Vx) / max(Vx, 1.0)
        v_L1_c = vx/3.6  #轮心速度
        v_R1_c = vx/3.6  
        v_L2_c = vx/3.6  
        v_R2_c = vx/3.6 
        
        s_L1 = (v_L1 - v_L1_c) / max(abs(v_L1), abs(v_L1_c)) if abs(vx) > 3.0 else 0.0
        s_R1 = (v_R1 - v_R1_c) / max(abs(v_R1), abs(v_R1_c)) if abs(vx) > 3.0 else 0.0
        s_L2 = (v_L2 - v_L2_c) / max(abs(v_L2), abs(v_L2_c)) if abs(vx) > 3.0 else 0.0
        s_R2 = (v_R2 - v_R2_c) / max(abs(v_R2), abs(v_R2_c)) if abs(vx) > 3.0 else 0.0
        

        return np.array([vx, ax, s_L1, s_R1, s_L2, s_R2, avz], dtype=np.float32)

    def _normalize_state(self, raw_state):
        # 简单的归一化，方便神经网络吃
        n_s = raw_state.copy()
        n_s[0] = raw_state[0] / 100.0  # Vx
        n_s[1] = raw_state[1] / 1.0    # Ax
        n_s[2:6] = raw_state[2:6] / 1  # Slips
        n_s[6] = raw_state[6] / 50.0   # Avz
        return n_s

    def _calculate_reward(self, state, current_torque, last_torque):
        # 解包状态
        vx = state[0]
        ax = state[1]
        slips = state[2:6]
        # avz = state[6]
        
        w = self.weights
        
        # 1. Speed 奖励
        r_speed = w['w_speed'] * vx
        # 2. Acceleration 奖励
        r_accel = w['w_accel'] * ax
        
        # 2. Slip 惩罚 (核心)
        excess = 0.0
        thresholds = [0.04, 0.04, 0.06, 0.06]
        for i in range(4):
            excess += max(0.0, abs(slips[i]) - thresholds[i])
            
        r_slip = 0.0
        if vx > 3.0: # 车动起来再算
            r_slip = w['w_slip'] * excess # 放大惩罚
            
        # 3. Energy
        r_energy = w['w_energy'] * np.mean((current_torque/self.max_torque))
        
        # 4. Consistency & Smooth
        r_smooth = w['w_smooth'] * np.mean(((current_torque - last_torque)/self.max_torque)**2)
        
        total = r_speed + r_accel + r_slip + r_energy + r_smooth
        details = {"R_Spd": r_speed, "R_Acc": r_accel, "R_Slp": r_slip, "R_Eng": r_energy}
        
        return total, details

    def get_state_dim(self): return self.state_dim
    def get_action_dim(self): return self.action_dim