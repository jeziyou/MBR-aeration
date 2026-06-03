import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import copy

# ==================== 常数与物理定义 ====================
CONSTANTS = {
    "SHEET_COUNT": 5,
    "SHEET_WIDTH": 1.25,
    "SHEET_END_MARGIN": 0.05,
    "PIPE_OFFSET": 0.45,
    "EFFECT_DECAY_RATE": 14,
    "SLUDGE_LAYER_MAX_HEIGHT": 0.6,
    "SLUDGE_DISCHARGE_RATE": 0.02,
    "UNIF_PENALTY_FACTOR": 0.4,
    "PULSE_POWER_BOOST": 1.4,
    "OFF_PHASE_POWER": 0.05,
}

ENHANCED_CONSTANTS = {
    "BUBBLE_DRAG_COEFF": 0.44,
    "BUBBLE_DIAMETER_m": 0.002,
    "GAS_HOLDUP_CORRECTION": 0.8,
    "VESILIND_V0": 7.0,
    "VESILIND_K": 0.6,
    "COMPRESSION_INDEX": 0.2,
    "MEMBRANE_RESISTANCE": 2.0e11,
    "FOULING_RATE_CONST": 1.0e-5,
    "BACKWASH_EFFICIENCY": 0.9,
    "TMP_MAX": 60.0,
}

PRESETS = {
    "eco": {"intensity": 60, "pulse_period": 6.0, "p_pitch": 250, "s_pitch": 100,
            "h_size": 6.0, "f_len": 2.0, "slack": 0.008, "mode": "cont",
            "fiber_diameter": 2.8, "thickness": 30, "mlss": 6000, "srt": 20,
            "settling_rate": 2.0, "return_ratio": 80},
    "balanced": {"intensity": 80, "pulse_period": 4.0, "p_pitch": 100, "s_pitch": 80,
                 "h_size": 4.0, "f_len": 2.0, "slack": 0.015, "mode": "cont",
                 "fiber_diameter": 1.65, "thickness": 30, "mlss": 8000, "srt": 15,
                 "settling_rate": 2.5, "return_ratio": 100},
    "flush": {"intensity": 110, "pulse_period": 3.0, "p_pitch": 50, "s_pitch": 50,
              "h_size": 2.5, "f_len": 2.0, "slack": 0.025, "mode": "pulse",
              "fiber_diameter": 1.65, "thickness": 30, "mlss": 10000, "srt": 12,
              "settling_rate": 3.0, "return_ratio": 150}
}

# ==================== 增强物理模型类 ====================
class MembraneFouling:
    def __init__(self, Rm: float = ENHANCED_CONSTANTS["MEMBRANE_RESISTANCE"]):
        self.Rm = Rm
        self.Rc = 0.0
        self.Rp = 0.0
        self.TMP = 0.0
        self.flux = 15.0

    def update(self, mlss: float, shear_pa: float, dt: float, backwash: bool = False) -> float:
        k_f = ENHANCED_CONSTANTS["FOULING_RATE_CONST"]
        dRc_dt = k_f * mlss / (1.0 + shear_pa) * (1.0 - self.Rc / 5e13)
        if backwash:
            dRc_dt = -ENHANCED_CONSTANTS["BACKWASH_EFFICIENCY"] * abs(dRc_dt)
            self.Rp *= 0.99
        self.Rc += dRc_dt * dt
        self.Rc = max(0.0, min(self.Rc, 5e13))
        self.Rp += 1e-8 * mlss * dt
        self.Rp = min(self.Rp, 5e12)
        mu = 0.001
        J_ms = self.flux / 3600.0
        self.TMP = mu * J_ms * (self.Rm + self.Rc + self.Rp) / 1000.0
        self.TMP = min(self.TMP, ENHANCED_CONSTANTS["TMP_MAX"])
        return self.TMP

class SludgeCompression:
    @staticmethod
    def settling_velocity(mlss: float, v0: float = 7.0, k: float = 0.6) -> float:
        return v0 * np.exp(-k * mlss / 1000.0)

    @staticmethod
    def compression_factor(sludge_level: float, max_height: float) -> float:
        return (sludge_level / max_height) ** ENHANCED_CONSTANTS["COMPRESSION_INDEX"]

class Hydraulics:
    @staticmethod
    def bubble_terminal_velocity(d_bubble: float = 0.002) -> float:
        g = 9.81
        Cd = ENHANCED_CONSTANTS["BUBBLE_DRAG_COEFF"]
        return np.sqrt(2.14 * Cd * g * d_bubble + 0.505 * g * d_bubble)

    @staticmethod
    def shear_from_bubbles(bubble_vel: float, gas_hold_up: float, density: float = 998) -> float:
        return 0.5 * density * bubble_vel ** 2 * gas_hold_up * ENHANCED_CONSTANTS["GAS_HOLDUP_CORRECTION"]

    @staticmethod
    def gas_hold_up(aeration_intensity: float, h_size: float, p_pitch: float, s_pitch: float) -> float:
        orifice_factor = (4.0 / max(1.5, h_size)) ** 0.3
        spacing_factor = np.exp(-abs(p_pitch - s_pitch) / 150.0)
        intensity_norm = (aeration_intensity - 50) / 100.0
        base_hold_up = 0.02 + 0.1 * intensity_norm
        return base_hold_up * orifice_factor * spacing_factor

class EnhancedMBRSimulator:
    def __init__(self):
        self.intensity = 110.0
        self.pulse_period = 3.0
        self.p_pitch = 50
        self.s_pitch = 50
        self.h_size = 2.5
        self.f_len = 2.0
        self.slack = 0.025
        self.fiber_diameter = 1.65
        self.thickness = 30
        self.mlss = 10000
        self.srt = 12
        self.settling_rate = 3.0
        self.return_ratio = 150
        self.mode = "pulse"
        self.sludge_level = 0.15
        self.is_discharging = False
        self.sim_time = 0.0
        self.fouling = MembraneFouling()
        self.sludge_compressor = SludgeCompression()
        self.hydraulics = Hydraulics()
        self.TMP_history = []
        self.backwash_flag = False
        self.avg_gas_hold_up = 0.05

    def apply_preset(self, name: str):
        preset = PRESETS[name]
        for k, v in preset.items():
            setattr(self, k, v)

    def get_sheet_area(self) -> float:
        base_area = 40.0 if self.fiber_diameter <= 1.8 else 25.0
        return round(base_area * (self.thickness / 30.0) * (self.f_len / 2.0), 2)

    def get_total_area(self) -> float:
        return self.get_sheet_area() * CONSTANTS["SHEET_COUNT"]

    def calculate_fiber_count(self):
        sheet_area = self.get_sheet_area()
        diameter_m = self.fiber_diameter / 1000.0
        area_per_fiber = np.pi * diameter_m * self.f_len
        real_count = max(1, int(sheet_area / area_per_fiber))
        return real_count, min(real_count, 300)

    def calculate_shear_stress(self) -> tuple:
        gas_hold_up = self.hydraulics.gas_hold_up(self.intensity, self.h_size, self.p_pitch, self.s_pitch)
        self.avg_gas_hold_up = gas_hold_up
        bubble_vel = self.hydraulics.bubble_terminal_velocity()
        intensity_norm = (self.intensity - 50.0) / 100.0
        bubble_vel *= (1 + 0.8 * intensity_norm)
        avg_shear = self.hydraulics.shear_from_bubbles(bubble_vel, gas_hold_up)
        slack_factor = 1.0 + self.slack * 12.0
        if self.mode == "pulse":
            max_shear = avg_shear * slack_factor * CONSTANTS["PULSE_POWER_BOOST"] * 1.5
        else:
            max_shear = avg_shear * slack_factor * 1.2
        avg_shear = np.clip(avg_shear, 0.1, 5.0)
        max_shear = np.clip(max_shear, 0.2, 8.0)
        return round(avg_shear, 3), round(max_shear, 3)

    def calculate_sec(self) -> float:
        delta_p = 50e3
        q_air = self.intensity * self.get_total_area() / 3600
        power = delta_p * q_air / 0.7
        flow_rate = 1.0
        sec = power / flow_rate / 1000
        return round(np.clip(sec, 0.05, 1.5), 3)

    def calculate_uniformity(self) -> float:
        diff = abs(self.p_pitch - self.s_pitch)
        return round(max(0.0, 100.0 - diff * CONSTANTS["UNIF_PENALTY_FACTOR"]), 1)

    def calculate_risk_level(self, max_shear: float) -> str:
        if max_shear < 0.8:
            return "HIGH"
        elif max_shear < 1.8:
            return "MEDIUM"
        else:
            return "LOW"

    def calculate_svi(self) -> float:
        svi = 200.0 - self.srt * 3.0 + (self.mlss / 10000.0) * 50.0
        return round(np.clip(svi, 50.0, 280.0), 1)

    def calculate_tss(self, avg_shear: float) -> float:
        base = 5.0 + (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 30.0
        shear_effect = max(0.0, avg_shear * 0.5)
        tss = base - shear_effect
        return round(np.clip(tss, 3.0, 35.0), 1)

    def update_sludge_level(self, dt: float = 1.0):
        if self.is_discharging:
            self.sludge_level = max(0.0, self.sludge_level - CONSTANTS["SLUDGE_DISCHARGE_RATE"] * dt)
        else:
            v_settle = self.sludge_compressor.settling_velocity(self.mlss) / 3600.0
            compress = self.sludge_compressor.compression_factor(self.sludge_level,
                                                                  CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])
            mlss_norm = (self.mlss - 2000.0) / 13000.0
            return_factor = np.clip(self.return_ratio / 100.0, 0.5, 3.0)
            net_settle = v_settle * (1 - compress) * mlss_norm * return_factor
            self.sludge_level += net_settle * dt
            self.sludge_level = min(self.sludge_level, CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])

    def update_fouling(self, dt: float) -> float:
        avg_shear, _ = self.calculate_shear_stress()
        tmp = self.fouling.update(self.mlss, avg_shear, dt, backwash=self.backwash_flag)
        self.TMP_history.append(tmp)
        if len(self.TMP_history) > 3600:
            self.TMP_history.pop(0)
        self.backwash_flag = False
        return tmp

    def get_current_tmp(self) -> float:
        mu = 0.001
        J_ms = self.fouling.flux / 3600.0
        tmp = mu * J_ms * (self.fouling.Rm + self.fouling.Rc + self.fouling.Rp) / 1000.0
        return min(tmp, ENHANCED_CONSTANTS["TMP_MAX"])

    def step_simulation(self, dt_hours: float):
        dt_sec = dt_hours * 3600.0
        self.update_sludge_level(dt_sec)
        self.update_fouling(dt_sec)
        self.sim_time += dt_sec

    def perform_backwash(self):
        self.backwash_flag = True
        avg_shear, _ = self.calculate_shear_stress()
        self.fouling.update(self.mlss, avg_shear, 0.1, backwash=True)

    def discharge_sludge(self):
        self.sludge_level = max(0.0, self.sludge_level - 0.08)
        self.is_discharging = False

    def get_metrics(self):
        avg_shear, max_shear = self.calculate_shear_stress()
        risk_text = self.calculate_risk_level(max_shear)
        return {
            "sec": self.calculate_sec(),
            "shear_avg": avg_shear,
            "shear_max": max_shear,
            "uniformity": self.calculate_uniformity(),
            "risk_text": risk_text,
            "svi": self.calculate_svi(),
            "tss": self.calculate_tss(avg_shear),
            "total_area": self.get_total_area(),
            "fiber_count": self.calculate_fiber_count()[0] * CONSTANTS["SHEET_COUNT"],
            "sludge_level_mm": self.sludge_level * 1000.0,
            "sludge_percent": (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 100.0,
            "tmp": round(self.get_current_tmp(), 2),
            "gas_hold_up": round(self.avg_gas_hold_up * 100, 1)
        }


# ==================== 3D HTML 生成器（保持原样）====================
# ... 此处省略，使用您之前提供的完整 generate_3d_html 函数 ...
# 为了节省篇幅，我假设您已保留该函数。实际运行时请将原函数复制至此。


# ==================== Streamlit UI ====================
st.set_page_config(page_title="MBR 工程级仿真系统 v2.0", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = EnhancedMBRSimulator()
sim = st.session_state.sim

with st.sidebar:
    st.header("🧪 MBR 系统控制")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🌿 节能模式", use_container_width=True):
            sim.apply_preset("eco")
            st.rerun()
    with c2:
        if st.button("⚖️ 均衡模式", use_container_width=True):
            sim.apply_preset("balanced")
            st.rerun()
    with c3:
        if st.button("💨 高冲刷模式", use_container_width=True):
            sim.apply_preset("flush")
            st.rerun()

    st.divider()
    st.subheader("🌊 曝气参数")
    sim.mode = st.selectbox("曝气模式", ["cont", "pulse"], format_func=lambda x: "连续曝气" if x == "cont" else "脉冲式曝气")
    sim.intensity = st.slider("曝气强度 (Nm³/m²/h)", 50, 150, int(sim.intensity), step=5)
    if sim.mode == "pulse":
        sim.pulse_period = st.slider("脉冲周期 (s)", 3.0, 6.0, float(sim.pulse_period), step=0.5)
    sim.h_size = st.slider("曝气孔径 (mm)", 1.0, 15.0, float(sim.h_size), step=0.5)
    sim.p_pitch = st.slider("曝气管间距 (mm)", 50, 300, int(sim.p_pitch), step=10)

    st.divider()
    st.subheader("🧬 膜片参数")
    sim.fiber_diameter = st.selectbox("膜丝外径", [1.65, 2.8], format_func=lambda x: f"{x} mm → {'40' if x==1.65 else '25'} m²")
    sim.thickness = st.slider("膜片厚度 (mm)", 10, 100, int(sim.thickness), step=5)
    sim.s_pitch = st.slider("膜片间距 (mm)", 50, 100, int(sim.s_pitch), step=5)
    sim.f_len = st.slider("膜丝长度 (m)", 0.1, 3.0, float(sim.f_len), step=0.1)
    sim.slack = st.slider("松弛度 (%)", 0.2, 5.0, float(sim.slack * 100), step=0.2) / 100.0

    st.divider()
    st.subheader("🧫 污泥与清洗")
    sim.mlss = st.slider("MLSS (mg/L)", 2000, 15000, int(sim.mlss), step=500)
    sim.srt = st.slider("污泥龄 (d)", 5, 40, int(sim.srt), step=1)
    sim.settling_rate = st.slider("沉降速率 (m/h)", 0.5, 6.0, float(sim.settling_rate), step=0.5)
    sim.return_ratio = st.slider("回流比 (%)", 50, 300, int(sim.return_ratio), step=10)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬇️ 排泥", use_container_width=True):
            sim.discharge_sludge()
            st.rerun()
    with col_b2:
        if st.button("🧼 反洗", use_container_width=True):
            sim.perform_backwash()
            st.rerun()
    if st.button("⏱️ 模拟1小时", use_container_width=True):
        sim.step_simulation(1.0)
        st.rerun()
    sludge_percent = sim.sludge_level / 0.6 * 100
    st.progress(min(100, int(sludge_percent)), text=f"污泥层 {sim.sludge_level*1000:.0f} mm")

st.title("💧 MBR 工程级仿真系统 v2.0")
st.caption("增强物理模型：Vesilind沉降 | 气泡剪切 | 膜污染(TMP) | 气含率 | 曝气能耗")

metrics = sim.get_metrics()

col1, col2, col3, col4 = st.columns(4)
col1.metric("能耗 SEC", f"{metrics['sec']} kWh/m³")
col2.metric("平均剪切力", f"{metrics['shear_avg']} Pa")
col3.metric("最大剪切力", f"{metrics['shear_max']} Pa")
col4.metric("覆盖均匀度", f"{metrics['uniformity']} %")

col5, col6, col7, col8 = st.columns(4)
col5.metric("积垢风险", metrics['risk_text'])
col6.metric("SVI", f"{metrics['svi']} mL/g")
col7.metric("TSS", f"{metrics['tss']} mg/L")
col8.metric("总膜面积", f"{metrics['total_area']} m²")

col9, col10, col11 = st.columns(3)
col9.metric("TMP 跨膜压力", f"{metrics['tmp']} kPa", delta=">35需反洗" if metrics['tmp'] > 35 else None)
col10.metric("气含率", f"{metrics['gas_hold_up']} %")
col11.metric("膜丝数量", f"{metrics['fiber_count']:,} 根")

st.markdown("### 🖥️ 3D 可视化视图")
# 假设 generate_3d_html 函数已定义，此处省略具体代码（请保留您原有的该函数）
# html_code = generate_3d_html(sim)
# st.components.v1.html(html_code, height=650, scrolling=False)

# 趋势图（基于当前状态的副本）
with st.expander("📈 12小时趋势预测 (剪切力 & 污泥层 & TMP)"):
    times = np.linspace(0, 12, 50)
    shear_vals = []
    sludge_vals = []
    tmp_vals = []
    # 创建临时仿真器并从主仿真器复制状态
    temp = EnhancedMBRSimulator()
    temp.intensity = sim.intensity
    temp.mode = sim.mode
    temp.pulse_period = sim.pulse_period
    temp.p_pitch = sim.p_pitch
    temp.s_pitch = sim.s_pitch
    temp.h_size = sim.h_size
    temp.slack = sim.slack
    temp.mlss = sim.mlss
    temp.settling_rate = sim.settling_rate
    temp.return_ratio = sim.return_ratio
    temp.sludge_level = sim.sludge_level
    temp.sim_time = sim.sim_time
    # 复制污染状态
    temp.fouling.Rm = sim.fouling.Rm
    temp.fouling.Rc = sim.fouling.Rc
    temp.fouling.Rp = sim.fouling.Rp
    temp.fouling.TMP = sim.fouling.TMP
    temp.fouling.flux = sim.fouling.flux
    temp.is_discharging = False
    step_hours = 12.0 / (len(times) - 1)
    for _ in times:
        avg, _ = temp.calculate_shear_stress()
        shear_vals.append(avg)
        sludge_vals.append(temp.sludge_level * 1000)
        tmp_vals.append(temp.get_current_tmp())
        temp.step_simulation(step_hours)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("剪切力 (Pa)", "污泥层高度 (mm)", "跨膜压力 TMP (kPa)"))
    fig.add_trace(go.Scatter(x=times, y=shear_vals, mode='lines+markers', name='剪切力', line=dict(color='#00f2ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=sludge_vals, mode='lines', name='污泥层', line=dict(color='#d4a84b')), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=tmp_vals, mode='lines', name='TMP', line=dict(color='#ff6644')), row=3, col=1)
    fig.update_layout(height=550, template="plotly_dark")
    fig.update_xaxes(title_text="模拟时间 (小时)", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
