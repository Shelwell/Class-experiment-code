import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# -------------------------- Matplotlib中文设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# -------------------------- 全局默认参数 --------------------------
DEFAULT_M = 1.0       # 质量 (kg)
DEFAULT_K = 100.0     # 弹簧刚度 (N/m)
DEFAULT_C = 5.0       # 阻尼系数 (N·s/m)
DEFAULT_OMEGA = 10.0  # 激励角频率 (rad/s)
DEFAULT_F0 = 10.0     # 激励幅值 (N)
T_END = 20.0          # 仿真总时间 (s)
N_POINTS = 2000       # 时间采样点数

# -------------------------- 系统微分方程定义 --------------------------
def undamped_system(y, t, m, k, F_func):
    """无阻尼受迫振动系统：m x'' + k x = F(t)"""
    x, x_dot = y
    dx_dt = x_dot
    dx_dot_dt = (F_func(t) - k * x) / m
    return [dx_dt, dx_dot_dt]

def parallel_damper_system(y, t, m, k, c, F_func):
    """并联阻尼受迫振动系统：m x'' + c x' + k x = F(t)"""
    x, x_dot = y
    dx_dt = x_dot
    dx_dot_dt = (F_func(t) - c * x_dot - k * x) / m
    return [dx_dt, dx_dot_dt]

def series_damper_system(y, t, m, k, c, F_func, F_prime_func):
    """串联阻尼受迫振动系统：c m x''' + k m x'' + k c x' = c F'(t) + k F(t)"""
    x, x_dot, x_ddot = y
    dx_dt = x_dot
    dx_dot_dt = x_ddot
    dx_ddot_dt = (c * F_prime_func(t) + k * F_func(t) - k * m * x_ddot - k * c * x_dot) / (c * m)
    return [dx_dt, dx_dot_dt, dx_ddot_dt]

# -------------------------- 主应用类 --------------------------
class VibrationSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("无阻尼/并联阻尼/串联阻尼受迫振动对比仿真")
        self.root.geometry("1200x750")
        
        # 创建左右分栏
        self.left_frame = ttk.Frame(root, width=250, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame = ttk.Frame(root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 初始化参数变量
        self.m_var = tk.DoubleVar(value=DEFAULT_M)
        self.k_var = tk.DoubleVar(value=DEFAULT_K)
        self.c_var = tk.DoubleVar(value=DEFAULT_C)
        self.omega_var = tk.DoubleVar(value=DEFAULT_OMEGA)
        self.freq_hz_var = tk.DoubleVar(value=DEFAULT_OMEGA/(2*np.pi))
        self.f0_var = tk.DoubleVar(value=DEFAULT_F0)
        
        # 曲线显示控制变量
        self.show_undamped = tk.BooleanVar(value=True)
        self.show_parallel = tk.BooleanVar(value=True)
        self.show_series = tk.BooleanVar(value=True)
        
        # 固有频率显示变量
        self.omega_n_var = tk.StringVar()
        self.fn_var = tk.StringVar()
        
        # 绑定频率同步事件
        self.omega_var.trace_add('write', self.omega_to_hz)
        self.freq_hz_var.trace_add('write', self.hz_to_omega)
        
        # 构建参数输入面板
        self.build_param_panel()
        
        # 构建绘图区域
        self.build_plot_area()
        
        # 初始计算固有频率并绘图
        self.update_natural_frequency()
        self.update_plot()

    def build_param_panel(self):
        """构建左侧参数输入面板"""
        # 系统参数
        ttk.Label(self.left_frame, text="系统参数设置", font=("Arial", 12, "bold")).pack(pady=10)
        
        ttk.Label(self.left_frame, text="质量 m (kg):").pack(anchor=tk.W, pady=2)
        ttk.Entry(self.left_frame, textvariable=self.m_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(self.left_frame, text="弹簧刚度 k (N/m):").pack(anchor=tk.W, pady=2)
        ttk.Entry(self.left_frame, textvariable=self.k_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(self.left_frame, text="阻尼系数 c (N·s/m):").pack(anchor=tk.W, pady=2)
        ttk.Entry(self.left_frame, textvariable=self.c_var).pack(fill=tk.X, pady=2)
        
        # 固有频率显示区域
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.left_frame, text="系统固有频率", font=("Arial", 12, "bold")).pack(pady=5)
        
        ttk.Label(self.left_frame, text="角频率 ωₙ (rad/s):").pack(anchor=tk.W, pady=2)
        ttk.Label(self.left_frame, textvariable=self.omega_n_var, foreground="blue", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        
        ttk.Label(self.left_frame, text="频率 fₙ (Hz):").pack(anchor=tk.W, pady=2)
        ttk.Label(self.left_frame, textvariable=self.fn_var, foreground="blue", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        
        # 激励参数
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.left_frame, text="激励参数设置", font=("Arial", 12, "bold")).pack(pady=10)
        
        ttk.Label(self.left_frame, text="激励角频率 ω (rad/s):").pack(anchor=tk.W, pady=2)
        ttk.Entry(self.left_frame, textvariable=self.omega_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(self.left_frame, text="激励频率 f (Hz):").pack(anchor=tk.W, pady=2)
        ttk.Entry(self.left_frame, textvariable=self.freq_hz_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(self.left_frame, text="激励幅值 F0 (N):").pack(anchor=tk.W, pady=2)
        ttk.Entry(self.left_frame, textvariable=self.f0_var).pack(fill=tk.X, pady=2)
        
        # 曲线显示控制
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(self.left_frame, text="曲线显示控制", font=("Arial", 12, "bold")).pack(pady=5)
        
        ttk.Checkbutton(self.left_frame, text="显示无阻尼系统", variable=self.show_undamped).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(self.left_frame, text="显示并联阻尼系统", variable=self.show_parallel).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(self.left_frame, text="显示串联阻尼系统", variable=self.show_series).pack(anchor=tk.W, pady=2)
        
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # 加载绘图按钮
        self.plot_btn = ttk.Button(
            self.left_frame, 
            text="加载参数并重新绘图", 
            command=self.update_all,
            style="Accent.TButton"
        )
        self.plot_btn.pack(fill=tk.X, pady=10)
        
        # 说明文本
        ttk.Label(
            self.left_frame, 
            text="说明：\n1. 修改参数后点击按钮更新\n2. 勾选/取消勾选控制曲线显示\n3. ω和f会自动同步转换\n4. 串联系统阻尼比与c成反比",
            wraplength=230,
            justify=tk.LEFT
        ).pack(pady=20)

    def build_plot_area(self):
        """构建右侧绘图区域"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加matplotlib工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def omega_to_hz(self, *args):
        """角频率转Hz"""
        try:
            omega = self.omega_var.get()
            freq_hz = omega / (2 * np.pi)
            self.freq_hz_var.set(round(freq_hz, 4))
        except:
            pass

    def hz_to_omega(self, *args):
        """Hz转角频率"""
        try:
            freq_hz = self.freq_hz_var.get()
            omega = freq_hz * 2 * np.pi
            self.omega_var.set(round(omega, 4))
        except:
            pass

    def update_natural_frequency(self):
        """计算并更新系统固有频率"""
        try:
            m = self.m_var.get()
            k = self.k_var.get()
            omega_n = np.sqrt(k / m)
            fn = omega_n / (2 * np.pi)
            self.omega_n_var.set(f"{omega_n:.4f}")
            self.fn_var.set(f"{fn:.4f}")
        except:
            self.omega_n_var.set("计算错误")
            self.fn_var.set("计算错误")

    def update_all(self):
        """更新固有频率并重新绘图"""
        self.update_natural_frequency()
        self.update_plot()

    def update_plot(self):
        """读取参数并更新绘图"""
        try:
            # 读取参数
            m = self.m_var.get()
            k = self.k_var.get()
            c = self.c_var.get()
            omega = self.omega_var.get()
            F0 = self.f0_var.get()
            
            # 生成时间数组
            t = np.linspace(0, T_END, N_POINTS)
            
            # 定义简谐激励及其导数
            def F(t):
                return F0 * np.sin(omega * t)
            
            def F_prime(t):
                return F0 * omega * np.cos(omega * t)
            
            # 初始条件
            y0_undamped = [0.0, 0.0]    # x(0)=0, x_dot(0)=0
            y0_parallel = [0.0, 0.0]    # x(0)=0, x_dot(0)=0
            y0_series = [0.0, 0.0, 0.0] # x(0)=0, x_dot(0)=0, x_ddot(0)=0
            
            # 清除旧图
            self.ax.clear()
            
            # 根据复选框状态绘制曲线
            if self.show_undamped.get():
                sol_undamped = odeint(undamped_system, y0_undamped, t, args=(m, k, F))
                x_undamped = sol_undamped[:, 0]
                self.ax.plot(t, x_undamped, 'b--', label='无阻尼系统', linewidth=1.2, alpha=0.8)
            
            if self.show_parallel.get():
                sol_parallel = odeint(parallel_damper_system, y0_parallel, t, args=(m, k, c, F))
                x_parallel = sol_parallel[:, 0]
                self.ax.plot(t, x_parallel, 'g-', label='并联阻尼系统', linewidth=1.5)
            
            if self.show_series.get():
                sol_series = odeint(series_damper_system, y0_series, t, args=(m, k, c, F, F_prime))
                x_series = sol_series[:, 0]
                self.ax.plot(t, x_series, 'r-', label='串联阻尼系统', linewidth=1.5)
            
            # 设置图表属性
            self.ax.set_title(
                f'受迫振动位移响应对比 (m={m}kg, k={k}N/m, c={c}N·s/m, ω={omega:.2f}rad/s, F0={F0}N)',
                fontsize=12
            )
            self.ax.set_xlabel('时间 t (s)', fontsize=10)
            self.ax.set_ylabel('位移 x (m)', fontsize=10)
            
            # 只有当有曲线显示时才显示图例
            if self.ax.get_legend_handles_labels()[0]:
                self.ax.legend(fontsize=10)
            
            self.ax.grid(True, alpha=0.3)
            self.ax.tick_params(axis='both', labelsize=9)
            
            # 刷新画布
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("参数错误", f"请输入有效的数值！\n错误信息：{str(e)}")

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = VibrationSimulator(root)
    root.mainloop()
