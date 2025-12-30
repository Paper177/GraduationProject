from attr import s
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional, Any
import os
import shutil
from pycarsimlib.core import CarsimManager

class PythonCarsimEnv:
    """
    CarSim Python 直连环境类
    用于管理与 CarSim 仿真软件的交互，包括初始化、重置、单步执行和关闭等操作。
    """
    
    # ================= CarSim 变量名配置 =================
    # 控制信号 (对应 CarSim Generic Import 中的变量名)
    IMP_THROTTLE = "IMP_THROTTLE_ENGINE" # 发动机油门开度 (0-1)
    IMP_BRAKE = "IMP_PCON_BK"        # 制动主缸压力 (MPa)
    
    # 4轮驱动/制动扭矩 (Nm)
    IMP_TORQUE_L1 = "IMP_MY_OUT_D1_L" # 左前轮扭矩
    IMP_TORQUE_R1 = "IMP_MY_OUT_D1_R" # 右前轮扭矩
    IMP_TORQUE_L2 = "IMP_MY_OUT_D2_L" # 左后轮扭矩
    IMP_TORQUE_R2 = "IMP_MY_OUT_D2_R" # 右后轮扭矩

    # 状态信号 (对应 CarSim Generic Export 中的变量名)
    EXP_VX = "Vx"         # 纵向车速 (km/h)
    EXP_VY = "Vy"         # 横向车速 (km/h)
    EXP_AX = "Ax"         # 纵向加速度 (g)
    EXP_AVZ = "AVz"       # 横摆角速度 (deg/s)
    EXP_STEER = "Steer_SW"       # 方向盘转角 (deg)

    # 轮速 (RPM)
    EXP_WHEEL_L1 = "AVy_L1" # 左前轮转速
    EXP_WHEEL_R1 = "AVy_R1" # 右前轮转速
    EXP_WHEEL_L2 = "AVy_L2" # 左后轮转速
    EXP_WHEEL_R2 = "AVy_R2" # 右后轮转速

    # ====================================================

    def __init__(
        self,
        carsim_db_dir: str,
        vehicle_type: str = "normal_vehicle", 
        sim_time_s: float = 10.0,
        delta_time_s: float = 0.01,
        max_torque: float = 1500.0,
        target_slip_ratio: float = 0.15,
        target_speed: float = 100.0,
        reward_weights: dict = None
    ):
        """
        初始化仿真环境
        
        Args:
            carsim_db_dir: CarSim 数据库目录路径
            vehicle_type: 车辆类型配置名称
            sim_time_s: 仿真总时长 (秒)
            delta_time_s: 仿真步长 (秒)
            max_torque: 最大扭矩限制 (Nm)
            target_slip_ratio: 目标滑移率
            target_speed: 目标速度 (km/h)
            reward_weights: 奖励函数权重字典
        """
        self.carsim_db_dir = carsim_db_dir
        self.vehicle_type = vehicle_type
        self.sim_time_s = sim_time_s
        self.delta_time = timedelta(seconds=delta_time_s)
        self.max_steps = int(sim_time_s / delta_time_s)
        
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.target_speed = target_speed
        
        # 初始化奖励权重
        default_weights = {
            'w_speed': 0.1, 'w_accel': 0.0, 'w_energy': 0.0,
            'w_consistency': 0.0, 'w_beta': 0.0, 'w_slip': -1.0, 'w_smooth': 0.0
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        # 车辆物理参数
        self.wheel_radius = 0.362 # 轮胎半径 m 
        self.veh_bf = 1.600; # 前轮距 m
        self.veh_br = 1.740; # 后轮距 m
        self.veh_l = 3.128   # 轴距 m
        self.veh_lf = 1.293  # 前轴到质心距离 m (满载状态)
        self.veh_lr = self.veh_l - self.veh_lf # 后轴到质心距离 m (满载状态)  

        # 内部状态变量
        self.cm: Optional[CarsimManager] = None
        self.current_step = 0
        self.last_torque = np.zeros(4)

        # 状态空间维度: [Vx, Ax, S_L1, S_R1, S_L2, S_R2, YawRate]
        self.state_dim = 7
        # 动作空间维度: [T_L1, T_R1, T_L2, T_R2]
        self.action_dim = 4

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        重置仿真环境到初始状态
        
        Returns:
            norm_state: 归一化后的初始状态
            info: 附加信息字典
        """
        # 1. 关闭旧的 Solver 实例
        self.close()
        
        # 2. 实例化新的 Manager 
        try:
            self.cm = CarsimManager(
                carsim_db_dir=self.carsim_db_dir,
                vehicle_type=self.vehicle_type
            )
        except Exception as e:
            raise RuntimeError(f"无法启动 CarSim Solver, 请检查路径和License: {e}")

        self.current_step = 0
        self.last_torque = np.zeros(4)
        
        # 3. 运行第 0 步以获取初始状态
        init_action = self._get_zero_action_dict()
        obs, _, _ = self.cm.step(action=init_action, delta_time=self.delta_time)
        
        # 4. 解析并归一化状态
        state = self._parse_observation(obs)
        norm_state = self._normalize_state(state)
        
        return norm_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步强化学习交互
        
        Args:
            action: 动作向量 (归一化到 [0, 1])
            
        Returns:
            next_state: 下一时刻状态
            reward: 当前步奖励
            done: 是否结束
            info: 附加信息
        """
        # 1. 动作处理 (Agent输出 [0, 1] -> 物理扭矩 Nm)
        action = np.clip(action, 0.0, 1.0)
        target_torque = action * self.max_torque
        # 2. 构造 CarSim 输入控制信号字典
        control_inputs = {
            self.IMP_THROTTLE: 0.0,
            self.IMP_BRAKE: 0.0,
            
            # 4轮扭矩分配
            self.IMP_TORQUE_L1: target_torque[0],
            self.IMP_TORQUE_R1: target_torque[1],
            self.IMP_TORQUE_L2: target_torque[2],
            self.IMP_TORQUE_R2: target_torque[3],
        }
        
        # 3. 执行一步物理仿真
        obs, terminated, _ = self.cm.step(action=control_inputs, delta_time=self.delta_time)
        
        # 4. 状态解析与归一化
        raw_state = self._parse_observation(obs)
        next_state = self._normalize_state(raw_state)
        
        # 5. 计算奖励
        reward, r_details = self._calculate_reward(raw_state, target_torque, self.last_torque)
        
        self.last_torque = target_torque
        self.current_step += 1
        slip_error = np.mean(np.abs(raw_state[2:6] - self.target_slip_ratio))
        
        # 6. 判断是否结束 (达到最大步数或仿真终止)
        done = (self.current_step >= self.max_steps) or terminated
        
        # 7. 构造详细 Info (用于看板显示和日志记录)
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
        
        if done: print() # 换行，用于控制台输出格式
            
        return next_state, reward, done, info

    def close(self):
        """关闭仿真环境，释放资源"""
        if self.cm:
            self.cm.close()
            self.cm = None

    def save_results_into_carsimdb(self, results_dir: str = "Results"):
        """
        将 CarSim 生成的 Results 文件夹内容复制到当前 simfile 所在的同级 Results 目录中
        用于保存仿真结果数据以便后续分析
        """
        if not self.cm:
            print("Error: CarSim simulation instance is not initialized.")
            return
        sim_dir = os.path.dirname(self.cm.simfile_path)
        source_results_path = os.path.join(sim_dir, results_dir)
        target_results_path = os.path.join(self.carsim_db_dir, results_dir)

        print(f"  正在保存仿真结果...")
        print(f"  源路径: {source_results_path}")
        print(f"  目标路径: {target_results_path}")

        if not os.path.exists(source_results_path):
            print(f"源结果目录不存在: {source_results_path}，可能仿真未生成结果或路径错误。")
            return

        try:
            if os.path.exists(target_results_path):
                pass
            shutil.copytree(source_results_path, target_results_path, dirs_exist_ok=True)
            print(f"  [Success] 结果已成功保存到 CarSim 数据库目录。")
        except Exception as e:
            print(f"  [Error] 保存结果失败: {e}")

    # ================= 辅助函数 =================
    
    def _get_zero_action_dict(self):
        """获取零输入动作字典"""
        return {
            self.IMP_THROTTLE: 0.0, self.IMP_BRAKE: 0.0,
            self.IMP_TORQUE_L1: 0.0, self.IMP_TORQUE_R1: 0.0,
            self.IMP_TORQUE_L2: 0.0, self.IMP_TORQUE_R2: 0.0
        }

    def _parse_observation(self, obs: Dict[str, float]) -> np.ndarray:
        """
        从 CarSim 输出字典解析物理值并计算滑移率
        
        Args:
            obs: CarSim 输出的观测字典
            
        Returns:
            state: 包含 [Vx, Ax, S_L1, S_R1, S_L2, S_R2, YawRate] 的 numpy 数组
        """
        vx = obs.get(self.EXP_VX, 0.0) # km/h
        vy = obs.get(self.EXP_VY, 0.0) # km/h
        ax = obs.get(self.EXP_AX, 0.0) # g (假设 CarSim 输出单位为 g)
        avz = obs.get(self.EXP_AVZ, 0.0) * np.pi / 180 # deg/s -> rad/s
        steer = obs.get(self.EXP_STEER, 0.0) * np.pi / 180 * 17.49 # 方向盘转角 -> 前轮转角
        vx_ms = vx / 3.6  # km/h -> m/s

        # 轮速 RPM -> 线速度 km/h
        rpm_to_ms = (2 * np.pi / 60.0) * self.wheel_radius
        v_L1 = obs.get(self.EXP_WHEEL_L1, 0.0) * rpm_to_ms * 3.6
        v_R1 = obs.get(self.EXP_WHEEL_R1, 0.0) * rpm_to_ms * 3.6
        v_L2 = obs.get(self.EXP_WHEEL_L2, 0.0) * rpm_to_ms * 3.6
        v_R2 = obs.get(self.EXP_WHEEL_R2, 0.0) * rpm_to_ms * 3.6
        
        # 计算轮心速度 (基于车辆运动学模型)
        # V_wheel_center = V_car + Omega x R
        v_L1_c = ((vx_ms - avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6 # km/h
        v_R1_c = ((vx_ms + avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6
        v_L2_c = ((vx_ms - avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6 
        v_R2_c = ((vx_ms + avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6

        # 计算滑移率 S = (V_wheel - V_wheel_center) / max(|V_wheel|, |V_wheel_center|, 0.1)
        # 仅当速度大于 3.0 km/h 时计算，否则置 0
        s_L1 = (v_L1 - v_L1_c) / max(abs(v_L1), abs(v_L1_c), 0.1) if max(abs(v_L1), abs(v_L1_c)) > 3.0 else 0.0
        s_R1 = (v_R1 - v_R1_c) / max(abs(v_R1), abs(v_R1_c), 0.1) if max(abs(v_R1), abs(v_R1_c)) > 3.0 else 0.0
        s_L2 = (v_L2 - v_L2_c) / max(abs(v_L2), abs(v_L2_c), 0.1) if max(abs(v_L2), abs(v_L2_c)) > 3.0 else 0.0
        s_R2 = (v_R2 - v_R2_c) / max(abs(v_R2), abs(v_R2_c), 0.1) if max(abs(v_R2), abs(v_R2_c)) > 3.0 else 0.0
        

        return np.array([vx, ax, s_L1, s_R1, s_L2, s_R2, avz], dtype=np.float32)

    def _normalize_state(self, raw_state):
        """
        状态归一化，便于神经网络处理
        """
        n_s = raw_state.copy()
        n_s[0] = raw_state[0] / 100.0  # Vx: 归一化基准 100 km/h
        n_s[1] = raw_state[1] / 1.0    # Ax: 归一化基准 1.0 g
        n_s[2:6] = raw_state[2:6] / 1.0  # Slips: 归一化基准 1.0
        n_s[6] = raw_state[6] / 1.5    # Avz: 归一化基准 1.5 rad/s
        return n_s

    def _calculate_reward(self, state, current_torque, last_torque):
        """
        计算奖励函数
        
        Args:
            state: 当前状态
            current_torque: 当前动作扭矩
            last_torque: 上一步动作扭矩
            
        Returns:
            total: 总奖励值
            details: 各分项奖励详情字典
        """
        # 解包状态
        vx = state[0]
        ax = state[1]
        slips = state[2:6]
        # avz = state[6]
        
        w = self.weights
        
        # 1. 速度奖励 (鼓励高速)
        r_speed = w['w_speed'] * vx
        # 2. 加速度奖励 (鼓励加速)
        r_accel = w['w_accel'] * ax
        
        # 3. 滑移率惩罚 (核心约束)
        # 计算超过阈值 (0.08) 的滑移率总和
        excess = 0.0
        thresholds = [0.08, 0.08, 0.08, 0.08]
        for i in range(4):
            excess += max(0.0, abs(slips[i]) - thresholds[i])
            
        r_slip = 0.0
        if vx > 3.0: # 仅在车辆运动时计算滑移率惩罚
            r_slip = w['w_slip'] * excess # 放大惩罚
            
        # 4. 能耗惩罚 (惩罚高扭矩使用)
        r_energy = w['w_energy'] * np.mean(np.abs(current_torque/self.max_torque))
        
        # 5. 一致性惩罚 (惩罚左右轮扭矩不一致)
        r_consistency = w['w_consistency'] * (abs(current_torque[0] - current_torque[1])+abs(current_torque[2] - current_torque[3]))/self.max_torque

        # 6. 平滑性惩罚 (惩罚扭矩剧烈变化)
        r_smooth = w['w_smooth'] * np.mean(((current_torque - last_torque)/self.max_torque)**2)

        total = r_speed + r_accel + r_slip + r_energy + r_consistency + r_smooth
        details = {"R_Spd": r_speed, "R_Acc": r_accel, "R_Slp": r_slip, "R_Eng": r_energy, "R_Cns": r_consistency}
        
        return total, details

    def get_state_dim(self): return self.state_dim
    def get_action_dim(self): return self.action_dim