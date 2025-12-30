# python_carsim/dashboard.py
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.live import Live
from rich.console import Console
from datetime import datetime
from collections import deque
from rich.text import Text

class TrainingDashboard:
    def __init__(self):
        self.logs = deque(maxlen=8)  # 保存最近8条日志
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="logs", size=10), # 新增日志区域
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="state_panel"),
            Layout(name="wheel_panel"),
        )

    def log(self, message: str):
        """添加一条日志到面板"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def generate_table(self, title, data_dict, style="cyan"):
        table = Table(title=title, expand=True, border_style=style)
        table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", justify="left", style="white")
        
        for k, v in data_dict.items():
            if isinstance(v, float):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)
            table.add_row(k, val_str)
        return table

    def update(self, episode, step, info, reward, noise, elite_flag):
        # 1. Header: 全局状态
        status_str = f"[bold green]Running[/bold green]" if not elite_flag else f"[bold yellow]🌟 ELITE MODE 🌟[/bold yellow]"
        header_text = f"DDPG Training Monitor | {datetime.now().strftime('%H:%M:%S')} | {status_str}"
        self.layout["header"].update(Panel(Align.center(header_text), style="white"))

        # 2. Left Panel: 车辆整体状态 & 训练指标
        veh_data = {
            "Episode": episode,
            "Step": step,
            "Speed (km/h)": info.get('vx', 0),
            "Ref Speed": info.get('ref_vx', 0),
            "Accel (g)": info.get('ax', 0),
            "Yaw Rate": info.get('yaw', 0),
            "Step Reward": reward,
            "Noise Scale": noise,
        }
        # 加入奖励细节
        for k, v in info.items():
            if k.startswith("R_"):
                veh_data[k] = v
                
        self.layout["state_panel"].update(
            Panel(self.generate_table("Vehicle & Agent State", veh_data), title="Global Info")
        )

        # 3. Right Panel: 四轮详细状态 (滑移率 & 扭矩)
        # 构造一个 4x2 的表格或者分别显示
        wheel_table = Table(title="Wheel Dynamics (FL/FR/RL/RR)", expand=True)
        wheel_table.add_column("Wheel", justify="center")
        wheel_table.add_column("Slip Ratio", justify="center")
        wheel_table.add_column("Torque (Nm)", justify="center")
        
        wheels = [("Front Left", "L1"), ("Front Right", "R1"), ("Rear Left", "L2"), ("Rear Right", "R2")]
        
        for name, suffix in wheels:
            slip = info.get(f"slip_{suffix}", 0.0)
            trq = info.get(f"trq_{suffix}", 0.0)
            
            # 滑移率颜色警告
            s_style = "green"
            if abs(slip) > 0.1: s_style = "yellow"
            if abs(slip) > 0.15: s_style = "red bold"
            
            wheel_table.add_row(
                name, 
                f"[{s_style}]{slip:.4f}[/{s_style}]", 
                f"{trq:.1f}"
            )

        self.layout["wheel_panel"].update(Panel(wheel_table, title="Actuation & Contact"))

        # 4. Logs Panel
        log_text = "\n".join(self.logs)
        self.layout["logs"].update(Panel(Text.from_markup(log_text), title="Training Logs", style="white"))
        
        # 5. Footer
        self.layout["footer"].update(Panel(Align.center("Press Ctrl+C to Stop Training safely"), style="grey50"))

        return self.layout

# 上下文管理器封装，方便调用
def create_live_monitor():
    dashboard = TrainingDashboard()
    return Live(dashboard.layout, refresh_per_second=10, screen=True), dashboard