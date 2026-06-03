import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import math

# ==================== 常数与物理定义 ====================
CONSTANTS = {
    "SHEET_COUNT": 5,
    "SHEET_WIDTH": 1.25,
    "SHEET_END_MARGIN": 0.05,
    "MAX_VISUAL_FIBERS": 300,      # 仅用于信息显示，不做视觉模拟
    "PIPE_OFFSET": 0.45,
    "EFFECT_DECAY_RATE": 14,
    "MAX_BASE_AMPLITUDE": 0.045,
    "MAX_BUBBLE_PUSH": 0.012,
    "MAX_TOTAL_SHIFT": 0.06,
    "BUBBLE_FIBER_COUPLING": 0.025,
    "SHEAR_SCALE": 120,
    "SHEAR_VELOCITY_SCALE": 8.0,
    "SLUDGE_LAYER_MAX_HEIGHT": 0.6,      # m
    "SLUDGE_SETTLE_RATE": 0.0002,        # m/s per g/L factor
    "SLUDGE_DISCHARGE_RATE": 0.02,       # m per click (对应一次排泥）
    "UNIF_PENALTY_FACTOR": 0.4,
    "PULSE_POWER_BOOST": 1.4,
    "OFF_PHASE_POWER": 0.05,
    "MONITOR_UPDATE_HZ": 8,
    "FPS_UPDATE_INTERVAL": 500,
    "MAX_DELTA_TIME": 0.05,
    "GEOMETRY_REBUILD_DEBOUNCE": 120,
    "STATS_UPDATE_HZ": 4
}

# 预设场景 (与原有保持一致)
PRESETS = {
    "eco": {
        "intensity": 60,
        "pulse_period": 6.0,
        "p_pitch": 250,
        "s_pitch": 100,
        "h_size": 6.0,
        "f_len": 2.0,
        "slack": 0.008,
        "mode": "cont",
        "fiber_diameter": 2.8,
        "thickness": 30,
        "mlss": 6000,
        "srt": 20,
        "settling_rate": 2.0,
        "return_ratio": 80
    },
    "balanced": {
        "intensity": 80,
        "pulse_period": 4.0,
        "p_pitch": 100,
        "s_pitch": 80,
        "h_size": 4.0,
        "f_len": 2.0,
        "slack": 0.015,
        "mode": "cont",
        "fiber_diameter": 1.65,
        "thickness": 30,
        "mlss": 8000,
        "srt": 15,
        "settling_rate": 2.5,
        "return_ratio": 100
    },
    "flush": {
        "intensity": 110,
        "pulse_period": 3.0,
        "p_pitch": 50,
        "s_pitch": 50,
        "h_size": 2.5,
        "f_len": 2.0,
        "slack": 0.025,
        "mode": "pulse",
        "fiber_diameter": 1.65,
        "thickness": 30,
        "mlss": 10000,
        "srt": 12,
        "settling_rate": 3.0,
        "return_ratio": 150
    }
}

class MBRSimulator:
    """MBR工艺仿真核心逻辑（移植自原JavaScript版本）"""
    def __init__(self):
        # 工艺参数
        self.intensity = 110.0          # 曝气强度 Nm³/m²/h
        self.pulse_period = 3.0         # 脉冲周期 (s)
        self.p_pitch = 50               # 曝气管间距 mm
        self.s_pitch = 50               # 膜片间距 mm
        self.h_size = 2.5               # 曝气孔径 mm
        self.f_len = 2.0                # 膜丝长度 m
        self.slack = 0.025              # 松弛度 (无单位比例)
        self.fiber_diameter = 1.65      # 膜丝外径 mm
        self.thickness = 30             # 膜片厚度 mm
        self.mlss = 10000               # mg/L
        self.srt = 12                   # 污泥龄 d
        self.settling_rate = 3.0        # 沉降速率 m/h
        self.return_ratio = 150         # 回流比 %
        self.mode = "pulse"             # "cont" 或 "pulse"
        
        # 状态变量
        self.sludge_level = 0.15        # 污泥层高度 (m)
        self.is_discharging = False
        self.sim_time = 0.0             # 模拟时间 (s)
        
    def apply_preset(self, name: str):
        """应用预设场景"""
        preset = PRESETS[name]
        self.intensity = preset["intensity"]
        self.pulse_period = preset["pulse_period"]
        self.p_pitch = preset["p_pitch"]
        self.s_pitch = preset["s_pitch"]
        self.h_size = preset["h_size"]
        self.f_len = preset["f_len"]
        self.slack = preset["slack"]
        self.mode = preset["mode"]
        self.fiber_diameter = preset["fiber_diameter"]
        self.thickness = preset["thickness"]
        self.mlss = preset["mlss"]
        self.srt = preset["srt"]
        self.settling_rate = preset["settling_rate"]
        self.return_ratio = preset["return_ratio"]
        
    def get_sheet_area(self) -> float:
        """计算单片膜面积 (m²)"""
        base_area = 40.0 if self.fiber_diameter <= 1.8 else 25.0
        # 厚度影响 (30mm为基准) 和长度影响 (2m为基准)
        area = base_area * (self.thickness / 30.0) * (self.f_len / 2.0)
        return round(area, 2)
    
    def get_total_area(self) -> float:
        """总膜面积 (m²)"""
        return self.get_sheet_area() * CONSTANTS["SHEET_COUNT"]
    
    def calculate_fiber_count(self) -> Tuple[int, int]:
        """计算实际纤维数量和可视化纤维数量 (仅用于信息显示)"""
        sheet_area = self.get_sheet_area()
        diameter_m = self.fiber_diameter / 1000.0
        area_per_fiber = np.pi * diameter_m * self.f_len
        real_count = max(1, int(sheet_area / area_per_fiber))
        visual_count = min(real_count, CONSTANTS["MAX_VISUAL_FIBERS"])
        return real_count, visual_count
    
    def get_effective_intensity_factor(self, burst_active: bool = True) -> float:
        """计算曝气强度归一化因子 (0~1之间，影响剪切力与能耗)"""
        norm_intensity = (self.intensity - 50.0) / 100.0   # 50~150 -> 0~1
        if self.mode == "pulse":
            # 脉冲模式：占空比 = min(1/period, 0.5)
            duty_cycle = min(1.0 / self.pulse_period, 0.5)
            if burst_active:
                pwr = norm_intensity * CONSTANTS["PULSE_POWER_BOOST"]
            else:
                pwr = CONSTANTS["OFF_PHASE_POWER"]
            # 最终平均系数 = 占空比 * 开启功率 + (1-占空比)*关闭功率
            effective = duty_cycle * max(0.0, min(1.5, pwr)) + (1 - duty_cycle) * CONSTANTS["OFF_PHASE_POWER"]
            return np.clip(effective, 0.05, 1.2)
        else:
            return np.clip(norm_intensity, 0.1, 1.0)
    
    def calculate_shear_stress(self) -> Tuple[float, float]:
        """计算平均剪切力 和 最大剪切力 (Pa) - 基于原JS物理模型简化回归"""
        # 强度因子
        intensity_factor = self.get_effective_intensity_factor(burst_active=True)
        # 松弛度影响
        slack_factor = 1.0 + self.slack * 12.0   # slack 0.008~0.05 -> 1.1~1.6
        # 孔径影响：小孔径增加局部剪切
        orifice_factor = (4.0 / max(1.5, self.h_size)) ** 0.4
        # 曝气管间距影响 (间距越小，气泡覆盖越密，剪切增大)
        pipe_pitch_factor = np.exp(-self.p_pitch / 180.0)   # 50->0.76, 300->0.19
        # 膜片间距影响 (间距越小，膜丝间扰动增强)
        sheet_pitch_factor = np.exp(-self.s_pitch / 100.0)   # 50->0.606, 100->0.367
        # 综合基础剪切力 (基准0.3Pa)
        base_shear = 0.3 * intensity_factor * slack_factor * orifice_factor * (pipe_pitch_factor + 0.3) * (sheet_pitch_factor + 0.5)
        # 平均剪切力 (0.1~3.5 Pa 范围)
        avg_shear = np.clip(base_shear, 0.1, 3.5)
        # 最大剪切力与曝气脉冲相关，脉冲模式下峰值更高
        if self.mode == "pulse":
            max_multiplier = 1.8
        else:
            max_multiplier = 1.2
        max_shear = np.clip(avg_shear * max_multiplier * (1.0 + self.slack * 3.0), 0.2, 6.0)
        return round(avg_shear, 3), round(max_shear, 3)
    
    def calculate_sec(self) -> float:
        """能耗 SEC (kWh/m³)"""
        norm_intensity = (self.intensity - 50.0) / 100.0
        base_sec = 0.12 + norm_intensity * 0.35
        # 脉冲模式平均能耗略低
        if self.mode == "pulse":
            duty = min(1.0 / self.pulse_period, 0.5)
            base_sec *= (duty * 1.2 + (1-duty)*0.3)
        return round(base_sec, 3)
    
    def calculate_uniformity(self) -> float:
        """覆盖均匀度 (%) 基于曝气管与膜片间距差异"""
        diff = abs(self.p_pitch - self.s_pitch)
        uniformity = max(0.0, 100.0 - diff * CONSTANTS["UNIF_PENALTY_FACTOR"])
        return round(uniformity, 1)
    
    def calculate_risk_level(self, max_shear: float) -> Tuple[str, str]:
        """积垢风险等级及CSS类名"""
        if max_shear < 0.8:
            return "HIGH", "risk-high"
        elif max_shear < 1.8:
            return "MEDIUM", "risk-medium"
        else:
            return "LOW", "risk-low"
    
    def calculate_svi(self) -> float:
        """污泥体积指数 SVI (mL/g)"""
        svi = 200.0 - self.srt * 3.0 + (self.mlss / 10000.0) * 50.0
        return round(np.clip(svi, 50.0, 280.0), 1)
    
    def calculate_tss(self, avg_shear: float) -> float:
        """污泥浓度 TSS (mg/L) 动态估算"""
        base_tss = 5.0 + (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 25.0
        shear_effect = max(0.0, avg_shear * 1.5)
        tss = base_tss + shear_effect * 2.0
        return round(np.clip(tss, 4.0, 35.0), 1)
    
    def update_sludge_level(self, dt: float = 1.0):
        """更新污泥层高度 (dt: 模拟时间步长, 秒)"""
        if self.is_discharging:
            # 排泥状态：高度降低
            reduction = CONSTANTS["SLUDGE_DISCHARGE_RATE"] * dt
            self.sludge_level = max(0.0, self.sludge_level - reduction)
        else:
            # 污泥积累：受MLSS、回流比、沉降速率影响
            mlss_norm = (self.mlss - 2000.0) / 13000.0  # 2000~15000 -> 0~1
            return_factor = np.clip(self.return_ratio / 100.0, 0.5, 3.0)
            settling_effect = self.settling_rate / 2.5
            accumulation_rate = CONSTANTS["SLUDGE_SETTLE_RATE"] * mlss_norm * return_factor * settling_effect
            self.sludge_level += accumulation_rate * dt
            self.sludge_level = min(self.sludge_level, CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])
    
    def discharge_sludge(self):
        """手动排泥 (模拟一次大剂量排泥)"""
        self.sludge_level = max(0.0, self.sludge_level - 0.08)
        self.is_discharging = False   # 单次脉冲排泥后自动关闭排泥标志
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取全部实时监控指标"""
        avg_shear, max_shear = self.calculate_shear_stress()
        risk_text, risk_class = self.risk_level = self.calculate_risk_level(max_shear)
        uniformity = self.calculate_uniformity()
        sec = self.calculate_sec()
        svi = self.calculate_svi()
        tss = self.calculate_tss(avg_shear)
        total_area = self.get_total_area()
        real_fibers, _ = self.calculate_fiber_count()
        return {
            "sec": sec,
            "shear_avg": avg_shear,
            "shear_max": max_shear,
            "uniformity": uniformity,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "svi": svi,
            "tss": tss,
            "total_area": total_area,
            "fiber_count": real_fibers * CONSTANTS["SHEET_COUNT"],
            "sludge_level_mm": self.sludge_level * 1000.0,
            "sludge_percent": (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 100.0
        }

# ==================== Streamlit UI ====================
st.set_page_config(page_title="MBR 工业仿真系统 v14.4", layout="wide", page_icon="💧")

# 初始化会话状态中的仿真核心
if "sim" not in st.session_state:
    st.session_state.sim = MBRSimulator()
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0
if "auto_update" not in st.session_state:
    st.session_state.auto_update = False

sim = st.session_state.sim

# 侧边栏: 控制面板
with st.sidebar:
    st.markdown("## 🧪 MBR 系统控制")
    st.markdown("---")
    
    # 预设场景
    st.markdown("### 场景预设 Presets")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌿 节能模式", use_container_width=True):
            sim.apply_preset("eco")
            st.session_state.sim_time = 0.0
            st.rerun()
    with col2:
        if st.button("⚖️ 均衡模式", use_container_width=True):
            sim.apply_preset("balanced")
            st.session_state.sim_time = 0.0
            st.rerun()
    with col3:
        if st.button("💨 高冲刷模式", use_container_width=True):
            sim.apply_preset("flush")
            st.session_state.sim_time = 0.0
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🌊 曝气参数 Aeration")
    sim.mode = st.selectbox("曝气模式", ["cont", "pulse"], format_func=lambda x: "连续曝气" if x=="cont" else "脉冲式曝气", index=0 if sim.mode=="cont" else 1)
    sim.intensity = st.slider("曝气强度 (Nm³/m²/h)", 50, 150, int(sim.intensity), step=5)
    if sim.mode == "pulse":
        sim.pulse_period = st.slider("脉冲周期 (s)", 3.0, 6.0, sim.pulse_period, step=0.5)
    sim.h_size = st.slider("曝气孔径 (mm)", 1.0, 15.0, sim.h_size, step=0.5)
    sim.p_pitch = st.slider("曝气管间距 (mm)", 50, 300, sim.p_pitch, step=10)
    
    st.markdown("---")
    st.markdown("### 🧬 膜片参数 Membrane")
    fd_opt = {1.65:"1.65 mm → 40 m²", 2.8:"2.8 mm → 25 m²"}
    sim.fiber_diameter = st.selectbox("膜丝外径", options=[1.65, 2.8], format_func=lambda x: fd_opt[x], index=0 if sim.fiber_diameter==1.65 else 1)
    sim.thickness = st.slider("膜片厚度 (mm)", 10, 100, sim.thickness, step=5)
    sim.s_pitch = st.slider("膜片排列间距 (mm)", 50, 100, sim.s_pitch, step=5)
    sim.f_len = st.slider("膜丝长度 (m)", 0.1, 3.0, sim.f_len, step=0.1)
    sim.slack = st.slider("膜丝松弛度 (%)", 0.2, 5.0, sim.slack*100.0, step=0.2) / 100.0
    
    st.markdown("---")
    st.markdown("### 🧫 污泥参数 Sludge")
    sim.mlss = st.slider("MLSS 浓度 (mg/L)", 2000, 15000, sim.mlss, step=500)
    sim.srt = st.slider("污泥龄 SRT (d)", 5, 40, sim.srt, step=1)
    sim.settling_rate = st.slider("沉降速率 (m/h)", 0.5, 6.0, sim.settling_rate, step=0.5)
    sim.return_ratio = st.slider("污泥回流比 (%)", 50, 300, sim.return_ratio, step=10)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("⬇️ 排泥", use_container_width=True):
            sim.discharge_sludge()
            st.rerun()
    with col_d2:
        if st.button("⏱️ 模拟运行 1h", use_container_width=True):
            sim.update_sludge_level(dt=3600.0)
            st.session_state.sim_time += 3600
            st.rerun()
    
    # 污泥层可视化条
    sludge_percent = (sim.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 100.0
    st.markdown(f"**污泥层高度**: {sim.sludge_level*1000:.0f} mm")
    st.progress(min(100, int(sludge_percent)), text="污泥层占比")

# 主区域: 实时监控与指标卡片
st.title("💧 MBR 工业仿真系统 v14.4")
st.caption("基于物理模型的计算引擎 | 白色膜丝柔性动力学等效 | 气-固-液三相耦合")

# 获取当前所有指标
metrics = sim.get_metrics()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("能耗 SEC", f"{metrics['sec']} kWh/m³", delta=None)
with col2:
    st.metric("平均剪切力 τ̄", f"{metrics['shear_avg']} Pa", delta=None)
with col3:
    st.metric("最大剪切力 τmax", f"{metrics['shear_max']} Pa", delta=None)
with col4:
    st.metric("覆盖均匀度", f"{metrics['uniformity']} %", delta=None)

col5, col6, col7, col8 = st.columns(4)
with col5:
    risk_color = "🔴" if metrics['risk_text']=="HIGH" else ("🟡" if metrics['risk_text']=="MEDIUM" else "🟢")
    st.metric("积垢风险", f"{risk_color} {metrics['risk_text']}", delta=None)
with col6:
    st.metric("SVI 污泥指数", f"{metrics['svi']} mL/g")
with col7:
    st.metric("污泥浓度 TSS", f"{metrics['tss']} mg/L")
with col8:
    st.metric("总膜面积", f"{metrics['total_area']} m²")

# 第二行附加信息
st.markdown("---")
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.info(f"🧵 纤维数量: **{metrics['fiber_count']:,}** 根")
with info_col2:
    st.info(f"🌊 模拟时间: **{st.session_state.sim_time/3600:.1f} h**")
with info_col3:
    st.info(f"🧫 污泥层高度: **{metrics['sludge_level_mm']:.0f} mm**")

# 图表: 剪切力与污泥高度动态 (使用plotly模拟实时变化趋势)
if st.button("🔄 刷新趋势图 (基于当前参数模拟)"):
    st.session_state.sim_time += 1800   # 模拟推进半小时
    sim.update_sludge_level(dt=1800.0)
    st.rerun()

# 生成趋势数据 (基于参数敏感性模拟)
time_points = np.linspace(0, 12, 50)  # 模拟12小时趋势
shear_vals = []
sludge_vals = []
temp_sim = MBRSimulator()
temp_sim.intensity = sim.intensity
temp_sim.mode = sim.mode
temp_sim.pulse_period = sim.pulse_period
temp_sim.p_pitch = sim.p_pitch
temp_sim.s_pitch = sim.s_pitch
temp_sim.h_size = sim.h_size
temp_sim.slack = sim.slack
temp_sim.mlss = sim.mlss
temp_sim.settling_rate = sim.settling_rate
temp_sim.return_ratio = sim.return_ratio
temp_sim.sludge_level = sim.sludge_level  # 起始值
for t in time_points:
    avg_s, _ = temp_sim.calculate_shear_stress()
    shear_vals.append(avg_s)
    sludge_vals.append(temp_sim.sludge_level * 1000)
    # 污泥随时间的缓慢累积（不排泥时）
    temp_sim.update_sludge_level(dt=3600.0 * 0.24)  # 每步约0.24h

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("平均剪切力 (Pa) 预测趋势", "污泥层高度 (mm) 累积趋势"))
fig.add_trace(go.Scatter(x=time_points, y=shear_vals, mode='lines+markers', name='剪切力', line=dict(color='#00f2ff')), row=1, col=1)
fig.add_trace(go.Scatter(x=time_points, y=sludge_vals, mode='lines', name='污泥高度', line=dict(color='#d4a84b')), row=2, col=1)
fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
fig.update_xaxes(title_text="模拟时间 (小时)", row=2, col=1)
fig.update_yaxes(title_text="剪切力 (Pa)", row=1, col=1)
fig.update_yaxes(title_text="污泥高度 (mm)", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# 显示关键物理影响参数 (帮助说明)
with st.expander("🔍 当前关键物理影响参数 (基于移植后的物理模型)"):
    st.write(f"""
    - **曝气强度系数**: {sim.get_effective_intensity_factor(burst_active=True):.2f}
    - **松弛度影响系数**: {1.0 + sim.slack * 12.0:.2f}
    - **孔径影响因子**: {(4.0 / max(1.5, sim.h_size)) ** 0.4:.2f}
    - **曝气管间距因子**: {np.exp(-sim.p_pitch / 180.0):.2f}
    - **膜片间距因子**: {np.exp(-sim.s_pitch / 100.0):.2f}
    - **污泥积累速率因子**: { (sim.mlss-2000)/13000 * (sim.return_ratio/100) * (sim.settling_rate/2.5):.4f} (mm/s)
    """)

# 底部注记
st.markdown("---")
st.caption("⚙️ MBR 仿真内核 v14.4 | 保留原始剪切力/能耗/均匀度算法，污泥层动态基于MLSS/回流比/沉降速率，支持脉冲曝气与连续曝气模式。")
