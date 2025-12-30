# env_pc.py
from attr import s
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Tuple, Optional, Any
import os
import shutil
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
    EXP_VY = "Vy"         # 横向车速 (km/h)
    EXP_AX = "Ax"         # 纵向加速度 (g, 需注意单位转换)
    EXP_AVZ = "AVz"       # 横摆角速度 (deg/s)
    EXP_STEER = "Steer_SW"       # 转向角 (deg)
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
        ref_speed_path: str = "Vx.xlsx",
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
        
        # 加载参考速度曲线
        self.ref_speeds = []
        if os.path.exists(ref_speed_path):
            try:
                if ref_speed_path.endswith('.xlsx') or ref_speed_path.endswith('.xls'):
                    df = pd.read_excel(ref_speed_path, header=None)
                    # 假设数据在第一列，不需要列名
                    self.ref_speeds = df.iloc[:, 0].tolist()
                else:
                    with open(ref_speed_path, 'r') as f:
                        for line in f:
                            try:
                                self.ref_speeds.append(float(line.strip()))
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Warning: Failed to load reference speed from {ref_speed_path}: {e}")
        else:
            print(f"Warning: Reference speed file {ref_speed_path} not found.")
        
        # 如果没有数据或长度不够，用 target_speed 填充
        if not self.ref_speeds:
            self.ref_speeds = [target_speed] * (self.max_steps + 1)
            
        # 奖励权重
        default_weights = {
            'w_speed': 0.1, 
            'w_accel': 0.0, 
            'w_energy': 0.0,
            'w_consistency': 0.0, 
            'w_beta': 0.0, 
            'w_slip': -1.0, 
            'w_smooth': 0.0,
            'w_ref_bonus': 1.0 # 新增：超越参考曲线的奖励权重
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        # 物理常数
        self.wheel_radius = 0.362 # m (需根据车型修改)
        self.veh_bf = 1.600; # 前轮距 m
        self.veh_br = 1.740; # 后轮距 m
        self.veh_l = 3.128
        self.veh_lf = 1.293  #前轴到质心距离 m (满载时)
        self.veh_lr = self.veh_l - self.veh_lf # 后轴到质心距离 m (满载时)  

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
        
        # 获取当前参考速度 (用于显示)
        ref_v = self.ref_speeds[min(self.current_step, len(self.ref_speeds)-1)]
        
        # 7. 构造详细 Info (用于看板显示)
        # raw_state 索引: 0:Vx, 1:Ax, 2:SL1, 3:SR1, 4:SL2, 5:SR2, 6:Yaw
        info = {
            # --- 车辆状态 ---
            "vx": raw_state[0],       # km/h
            "ref_vx": ref_v,          # 参考速度
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

    def save_results_into_carsimdb(self, results_dir: str = "Results"):
        """
        将CarSim生成的 Results 文件夹内容复制到当前simfile所在的同级Results目录中,
        以便在CarSim软件中查看动画和绘图。
        
        Args:
            results_dir: 默认为 "Results" (CarSim标准输出目录名)
        """
        if not self.cm:
            print("Error: CarSim simulation instance is not initialized.")
            return

        # 1. 获取本次仿真临时工作目录 (例如 C:\Users\...\AppData\Local\Temp\pycarsimlib_xxxxx\)
        #    在 pycarsimlib 中，self.cm.sim_path 指向的是 simfile (simfile.sim) 的路径
        sim_dir = os.path.dirname(self.cm.simfile_path)
        
        # 2. 构造源目录路径: 临时目录下的 Results 文件夹
        source_results_path = os.path.join(sim_dir, results_dir)
        
        # 3. 构造目标目录路径: carsim_db_dir 下的 Results 文件夹
        #    注意：CarSim工程通常要求 Results 文件夹与 .sim 文件引用路径一致
        #    这里我们简单地将其放回数据库根目录下的 Results 文件夹
        #    如果数据库结构不同，需要调整此处逻辑
        target_results_path = os.path.join(self.carsim_db_dir, results_dir)

        print(f"[Save Results] 正在保存仿真结果...")
        print(f"  源路径: {source_results_path}")
        print(f"  目标路径: {target_results_path}")

        if not os.path.exists(source_results_path):
            print(f"  [Warning] 源结果目录不存在: {source_results_path}，可能仿真未生成结果或路径错误。")
            return

        try:
            # 如果目标目录存在，CarSim可能正在占用或需要覆盖
            # copytree 默认要求目标目录不存在 (dirs_exist_ok=True 在 Python 3.8+ 可用)
            # 为兼容性，建议使用 copytree(..., dirs_exist_ok=True)
            
            if os.path.exists(target_results_path):
                # 简单策略：覆盖同名文件
                pass
            
            shutil.copytree(source_results_path, target_results_path, dirs_exist_ok=True)
            print(f"  [Success] 结果已成功保存到 CarSim 数据库目录。")
        except Exception as e:
            print(f"  [Error] 保存结果失败: {e}")

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
        vy = obs.get(self.EXP_VY, 0.0) # km/h
        ax = obs.get(self.EXP_AX, 0.0) # g? 假设 CarSim 输出是 g
        avz = obs.get(self.EXP_AVZ, 0.0) * np.pi / 180
        steer = obs.get(self.EXP_STEER, 0.0) * np.pi / 180 * 17.49
        vx_ms = vx / 3.6  # m/s

        # 轮速 RPM -> 线速度 kph
        rpm_to_ms = (2 * np.pi / 60.0) * self.wheel_radius
        v_L1 = obs.get(self.EXP_WHEEL_L1, 0.0) * rpm_to_ms * 3.6
        v_R1 = obs.get(self.EXP_WHEEL_R1, 0.0) * rpm_to_ms * 3.6
        v_L2 = obs.get(self.EXP_WHEEL_L2, 0.0) * rpm_to_ms * 3.6
        v_R2 = obs.get(self.EXP_WHEEL_R2, 0.0) * rpm_to_ms * 3.6
        
        # 计算滑移率 S = (V_wheel - Vx) / max(Vx, 1.0)
        v_L1_c = ((vx_ms - avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6 #轮心速度kph
        v_R1_c = ((vx_ms + avz*0.5*self.veh_bf)*np.cos(steer)  + (vy+avz*self.veh_lf)*np.sin(steer)) * 3.6
        v_L2_c = ((vx_ms - avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6 
        v_R2_c = ((vx_ms + avz*0.5*self.veh_br)*np.cos(steer) ) * 3.6

        s_L1 = (v_L1 - v_L1_c) / max(abs(v_L1), abs(v_L1_c)) if max(abs(v_L1), abs(v_L1_c)) > 3.0 else 0.0
        s_R1 = (v_R1 - v_R1_c) / max(abs(v_R1), abs(v_R1_c)) if max(abs(v_R1), abs(v_R1_c)) > 3.0 else 0.0
        s_L2 = (v_L2 - v_L2_c) / max(abs(v_L2), abs(v_L2_c)) if max(abs(v_L2), abs(v_L2_c)) > 3.0 else 0.0
        s_R2 = (v_R2 - v_R2_c) / max(abs(v_R2), abs(v_R2_c)) if max(abs(v_R2), abs(v_R2_c)) > 3.0 else 0.0
        

        return np.array([vx, ax, s_L1, s_R1, s_L2, s_R2, avz], dtype=np.float32)

    def _normalize_state(self, raw_state):
        # 简单的归一化，方便神经网络吃
        n_s = raw_state.copy()
        n_s[0] = raw_state[0] / 100.0  # Vx
        n_s[1] = raw_state[1] / 1.0    # Ax
        n_s[2:6] = raw_state[2:6] / 1.0  # Slips (0-1 typically)
        n_s[6] = raw_state[6] / 1.5    # Avz (rad/s), 1.5 rad/s is approx 86 deg/s, reasonable max
        return n_s

    def _calculate_reward(self, state, current_torque, last_torque):
        # 解包状态
        vx = state[0]
        ax = state[1]
        slips = state[2:6]
        # avz = state[6]
        
        w = self.weights
        
        # 获取当前步的参考速度
        ref_v = self.ref_speeds[min(self.current_step, len(self.ref_speeds)-1)]
        
        # 1. Speed 奖励
        # 基础速度奖励
        r_speed = w['w_speed'] * vx
        
        # 新增: 超越参考速度奖励
        # 逻辑: 鼓励 vx > ref_v。如果 vx > ref_v，差值为正，获得正奖励；反之获得负奖励（惩罚落后）
        r_ref = w.get('w_ref_bonus', 0.0) * (vx - ref_v)
        
        # 2. Acceleration 奖励
        r_accel = w['w_accel'] * ax
        
        # 2. Slip 惩罚 (核心)
        excess = 0.0
        thresholds = [0.08, 0.08, 0.08, 0.08]
        for i in range(4):
            excess += max(0.0, abs(slips[i]) - thresholds[i])
            
        r_slip = 0.0
        if vx > 3.0: # 车动起来再算
            r_slip = w['w_slip'] * excess # 放大惩罚
            
        # 3. Energy (Penalty for high torque usage)
        # Assuming w_energy is positive in config, we subtract it or handle sign there.
        # But typically energy is a cost. Let's make it consistent with standard RL practices.
        # If weight is positive, this term adds to reward. User's config had positive weight.
        # We should probably change the formula to reflect penalty if weight is positive, OR change weight to negative.
        # Here we keep formula simple and rely on weight being negative, OR change formula to abs().
        # Let's use absolute value of torque ratio, as torque can be negative (though in this env torque seems 0-1 mapped from action?)
        # IMP_TORQUE inputs are usually absolute torque requests.
        r_energy = w['w_energy'] * np.mean(np.abs(current_torque/self.max_torque))
        r_consistency = w['w_consistency'] * (abs(current_torque[0] - current_torque[1])+abs(current_torque[2] - current_torque[3]))/self.max_torque

        # 5. Smooth
        r_smooth = w['w_smooth'] * np.mean(((current_torque - last_torque)/self.max_torque)**2)

        total = r_speed + r_ref + r_accel + r_slip + r_energy + r_consistency + r_smooth
        details = {"R_Spd": r_speed, "R_Ref": r_ref, "R_Acc": r_accel, "R_Slp": r_slip, "R_Eng": r_energy, "R_Cns": r_consistency}
        
        return total, details
        
        return total, details

    def get_state_dim(self): return self.state_dim
    def get_action_dim(self): return self.action_dim