"""
MBR 工程级仿真系统 v3.0
=========================
增强物理模型：Vesilind 沉降 | 气泡剪切 | 膜污染(TMP) | 气含率 | 曝气能耗
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ==================== 枚举定义 ====================
class AerationMode(Enum):
    """曝气模式枚举"""
    CONTINUOUS = "cont"
    PULSE = "pulse"

    @property
    def label(self) -> str:
        return "连续曝气" if self == AerationMode.CONTINUOUS else "脉冲式曝气"


# ==================== 物理常数 ====================
@dataclass(frozen=True)
class PhysicsConstants:
    """统一物理常数"""
    # 膜架几何
    sheet_count: int = 5
    sheet_width: float = 1.25
    sheet_end_margin: float = 0.05
    pipe_offset: float = 0.45

    # 气泡/流体
    bubble_drag_coeff: float = 0.44
    bubble_diameter_m: float = 0.002
    gas_holdup_correction: float = 0.8
    effect_decay_rate: float = 14.0

    # 剪切
    uniform_penalty_factor: float = 0.4
    pulse_power_boost: float = 1.4
    off_phase_power: float = 0.05

    # 污泥
    sludge_layer_max_height: float = 0.6
    sludge_discharge_rate: float = 0.02

    # Vesilind 沉降模型
    vesilind_v0: float = 7.0
    vesilind_k: float = 0.6
    compression_index: float = 0.2

    # 膜污染
    membrane_resistance: float = 2.0e11
    fouling_rate_const: float = 1.0e-5
    backwash_efficiency: float = 0.9
    tmp_max: float = 60.0
    dynamic_viscosity: float = 0.001


PHYS = PhysicsConstants()


# ==================== 预设配置 ====================
@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    icon: str
    intensity: float
    pulse_period: float
    p_pitch: int
    s_pitch: int
    h_size: float
    f_len: float
    slack: float
    mode: AerationMode
    fiber_diameter: float
    thickness: int
    mlss: int
    srt: int
    settling_rate: float
    return_ratio: int
    pipe_to_membrane_gap: int


PRESETS: Dict[str, Preset] = {
    "eco": Preset(
        name="eco", label="节能模式", icon="🌿",
        intensity=60, pulse_period=6.0, p_pitch=250, s_pitch=100,
        h_size=6.0, f_len=2.0, slack=0.008, mode=AerationMode.CONTINUOUS,
        fiber_diameter=2.8, thickness=30, mlss=6000, srt=20,
        settling_rate=2.0, return_ratio=80,
        pipe_to_membrane_gap=400,
    ),
    "balanced": Preset(
        name="balanced", label="均衡模式", icon="⚖️",
        intensity=80, pulse_period=4.0, p_pitch=100, s_pitch=80,
        h_size=4.0, f_len=2.0, slack=0.015, mode=AerationMode.CONTINUOUS,
        fiber_diameter=1.65, thickness=30, mlss=8000, srt=15,
        settling_rate=2.5, return_ratio=100,
        pipe_to_membrane_gap=250,
    ),
    "flush": Preset(
        name="flush", label="高冲刷模式", icon="💨",
        intensity=110, pulse_period=3.0, p_pitch=50, s_pitch=50,
        h_size=2.5, f_len=2.0, slack=0.025, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=30, mlss=10000, srt=12,
        settling_rate=3.0, return_ratio=150,
        pipe_to_membrane_gap=150,
    ),
    "low_fouling": Preset(
        name="low_fouling", label="低污染模式", icon="🛡️",
        intensity=90, pulse_period=5.0, p_pitch=80, s_pitch=60,
        h_size=3.0, f_len=2.0, slack=0.02, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=40, mlss=6000, srt=25,
        settling_rate=2.0, return_ratio=120,
        pipe_to_membrane_gap=300,
    ),
    "high_flux": Preset(
        name="high_flux", label="高通量模式", icon="⚡",
        intensity=130, pulse_period=2.5, p_pitch=40, s_pitch=40,
        h_size=2.0, f_len=2.5, slack=0.03, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=25, mlss=12000, srt=10,
        settling_rate=3.5, return_ratio=200,
        pipe_to_membrane_gap=150,
    ),
}


# ==================== 数据类 ====================
@dataclass
class SimulationMetrics:
    """仿真指标汇总"""
    sec: float = 0.0
    shear_avg: float = 0.0
    shear_max: float = 0.0
    uniformity: float = 0.0
    risk_text: str = "N/A"
    svi: float = 0.0
    tss: float = 0.0
    total_area: float = 0.0
    fiber_count: int = 0
    sludge_level_mm: float = 0.0
    sludge_percent: float = 0.0
    tmp: float = 0.0
    gas_hold_up: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ==================== 物理模型组件 ====================
class MembraneFouling:
    """膜污染模型：跟踪滤饼层阻力、孔堵阻力和跨膜压力(TMP)。"""

    def __init__(self, Rm: float = PHYS.membrane_resistance) -> None:
        self.Rm: float = Rm
        self.Rc: float = 0.0
        self.Rp: float = 0.0
        self.TMP: float = 0.0
        self.flux: float = 15.0  # LMH

    def update(self, mlss: float, shear_pa: float, dt: float, backwash: bool = False) -> float:
        """更新污染状态，返回当前 TMP (kPa)。"""
        k_f = PHYS.fouling_rate_const
        dRc_dt = k_f * mlss / (1.0 + shear_pa) * (1.0 - self.Rc / 5e13)

        if backwash:
            dRc_dt = -PHYS.backwash_efficiency * abs(dRc_dt)
            self.Rp *= 0.99

        self.Rc += dRc_dt * dt
        self.Rc = max(0.0, min(self.Rc, 5e13))
        self.Rp += 1e-8 * mlss * dt
        self.Rp = min(self.Rp, 5e12)

        J_ms: float = self.flux / 3600.0
        self.TMP = PHYS.dynamic_viscosity * J_ms * (self.Rm + self.Rc + self.Rp) / 1000.0
        self.TMP = min(self.TMP, PHYS.tmp_max)
        return self.TMP

    def get_state(self) -> Dict:
        """导出当前污染状态。"""
        return {"Rm": self.Rm, "Rc": self.Rc, "Rp": self.Rp, "TMP": self.TMP, "flux": self.flux}

    def set_state(self, state: Dict) -> None:
        """恢复污染状态。"""
        for k, v in state.items():
            setattr(self, k, v)


class SludgeCompression:
    """污泥压缩与沉降模型（Vesilind）。"""

    @staticmethod
    def settling_velocity(mlss: float, v0: float = PHYS.vesilind_v0,
                          k: float = PHYS.vesilind_k) -> float:
        """Vesilind 沉降速度 (m/h)。"""
        return v0 * np.exp(-k * mlss / 1000.0)

    @staticmethod
    def compression_factor(sludge_level: float, max_height: float = PHYS.sludge_layer_max_height) -> float:
        """污泥压缩因子。"""
        return (sludge_level / max_height) ** PHYS.compression_index


class Hydraulics:
    """水力与气泡动力学模型。"""

    @staticmethod
    def bubble_terminal_velocity(d_bubble: float = PHYS.bubble_diameter_m) -> float:
        """气泡终端速度 (m/s)。"""
        g = 9.81
        Cd = PHYS.bubble_drag_coeff
        return float(np.sqrt(2.14 * Cd * g * d_bubble + 0.505 * g * d_bubble))

    @staticmethod
    def shear_from_bubbles(bubble_vel: float, gas_hold_up: float, density: float = 998.0) -> float:
        """气泡引起的剪切应力 (Pa)。"""
        return 0.5 * density * bubble_vel ** 2 * gas_hold_up * PHYS.gas_holdup_correction

    @staticmethod
    def gas_hold_up(intensity: float, h_size: float, p_pitch: float, s_pitch: float) -> float:
        """气含率计算。"""
        orifice_factor: float = (4.0 / max(1.5, h_size)) ** 0.3
        spacing_factor: float = float(np.exp(-abs(p_pitch - s_pitch) / 150.0))
        intensity_norm: float = (intensity - 50.0) / 100.0
        base_hold_up: float = 0.02 + 0.1 * intensity_norm
        return base_hold_up * orifice_factor * spacing_factor


# ==================== 核心仿真器 ====================
class EnhancedMBRSimulator:
    """增强型 MBR 仿真器，集成物理模型与实时状态。"""

    def __init__(self) -> None:
        self.intensity: float = 110.0
        self.pulse_period: float = 3.0
        self.p_pitch: int = 50
        self.s_pitch: int = 50
        self.h_size: float = 2.5
        self.f_len: float = 2.0
        self.slack: float = 0.025
        self.fiber_diameter: float = 1.65
        self.thickness: int = 30
        self.mlss: int = 10000
        self.srt: int = 12
        self.settling_rate: float = 3.0
        self.return_ratio: int = 150
        self.mode: AerationMode = AerationMode.PULSE
        self.pipe_to_membrane_gap: int = 200  # 曝气管到膜片底部距离 (mm)

        self.sludge_level: float = 0.15
        self.is_discharging: bool = False
        self.sim_time: float = 0.0

        self.fouling: MembraneFouling = MembraneFouling()
        self.sludge_compressor: SludgeCompression = SludgeCompression()
        self.hydraulics: Hydraulics = Hydraulics()

        self.TMP_history: List[float] = []
        self.backwash_flag: bool = False
        self.avg_gas_hold_up: float = 0.05

    # ----- 预设 -----
    def apply_preset(self, name: str) -> None:
        """应用预设配置。"""
        preset = PRESETS[name]
        for key in ("intensity", "pulse_period", "p_pitch", "s_pitch", "h_size",
                     "f_len", "slack", "mode", "fiber_diameter", "thickness",
                     "mlss", "srt", "settling_rate", "return_ratio",
                     "pipe_to_membrane_gap"):
            setattr(self, key, getattr(preset, key))

    # ----- 几何计算 -----
    def get_sheet_area(self) -> float:
        """单张膜片面积 (m²)。"""
        base_area: float = 40.0 if self.fiber_diameter <= 1.8 else 25.0
        return round(base_area * (self.thickness / 30.0) * (self.f_len / 2.0), 2)

    def get_total_area(self) -> float:
        """总膜面积 (m²)。"""
        return self.get_sheet_area() * PHYS.sheet_count

    def calculate_fiber_count(self) -> Tuple[int, int]:
        """计算膜丝数量：(实际数量, 显示上限)。"""
        sheet_area: float = self.get_sheet_area()
        diameter_m: float = self.fiber_diameter / 1000.0
        area_per_fiber: float = np.pi * diameter_m * self.f_len
        real_count: int = max(1, int(sheet_area / area_per_fiber))
        return real_count, min(real_count, 300)

    # ----- 剪切与能耗 -----
    def calculate_shear_stress(self) -> Tuple[float, float]:
        """计算剪切应力 (平均, 最大) Pa。"""
        gas_hold_up: float = self.hydraulics.gas_hold_up(
            self.intensity, self.h_size, self.p_pitch, self.s_pitch)
        self.avg_gas_hold_up = gas_hold_up

        bubble_vel: float = self.hydraulics.bubble_terminal_velocity()
        intensity_norm: float = (self.intensity - 50.0) / 100.0
        bubble_vel *= (1.0 + 0.8 * intensity_norm)

        avg_shear: float = self.hydraulics.shear_from_bubbles(bubble_vel, gas_hold_up)
        slack_factor: float = 1.0 + self.slack * 12.0

        if self.mode == AerationMode.PULSE:
            max_shear: float = avg_shear * slack_factor * PHYS.pulse_power_boost * 1.5
        else:
            max_shear = avg_shear * slack_factor * 1.2

        avg_shear = float(np.clip(avg_shear, 0.1, 5.0))
        max_shear = float(np.clip(max_shear, 0.2, 8.0))
        return round(avg_shear, 3), round(max_shear, 3)

    def calculate_sec(self) -> float:
        """计算比能耗 SEC (kWh/m³)。"""
        delta_p: float = 50e3  # Pa
        q_air: float = self.intensity * self.get_total_area() / 3600.0
        power: float = delta_p * q_air / 0.7
        flow_rate: float = 1.0  # m³/h 基准
        sec: float = power / flow_rate / 1000.0
        return round(float(np.clip(sec, 0.05, 1.5)), 3)

    # ----- 均匀度与风险 -----
    def calculate_uniformity(self) -> float:
        """曝气覆盖均匀度 (%)。"""
        diff: float = abs(self.p_pitch - self.s_pitch)
        return round(max(0.0, 100.0 - diff * PHYS.uniform_penalty_factor), 1)

    @staticmethod
    def calculate_risk_level(max_shear: float) -> str:
        """根据最大剪切力评估积垢风险。"""
        if max_shear < 0.8:
            return "HIGH"
        elif max_shear < 1.8:
            return "MEDIUM"
        return "LOW"

    # ----- 污泥指标 -----
    def calculate_svi(self) -> float:
        """污泥体积指数 SVI (mL/g)。"""
        svi: float = 200.0 - self.srt * 3.0 + (self.mlss / 10000.0) * 50.0
        return round(float(np.clip(svi, 50.0, 280.0)), 1)

    def calculate_tss(self, avg_shear: float) -> float:
        """出水总悬浮固体 TSS (mg/L)。"""
        base: float = 5.0 + (self.sludge_level / PHYS.sludge_layer_max_height) * 30.0
        shear_effect: float = max(0.0, avg_shear * 0.5)
        tss: float = base - shear_effect
        return round(float(np.clip(tss, 3.0, 35.0)), 1)

    # ----- 状态更新 -----
    def update_sludge_level(self, dt: float = 1.0) -> None:
        """更新污泥层高度。"""
        if self.is_discharging:
            self.sludge_level = max(0.0, self.sludge_level - PHYS.sludge_discharge_rate * dt)
            return

        v_settle: float = self.sludge_compressor.settling_velocity(self.mlss) / 3600.0
        compress: float = self.sludge_compressor.compression_factor(
            self.sludge_level, PHYS.sludge_layer_max_height)
        mlss_norm: float = (self.mlss - 2000.0) / 13000.0
        return_factor: float = float(np.clip(self.return_ratio / 100.0, 0.5, 3.0))

        net_settle: float = v_settle * (1.0 - compress) * mlss_norm * return_factor
        self.sludge_level += net_settle * dt
        self.sludge_level = min(self.sludge_level, PHYS.sludge_layer_max_height)

    def update_fouling(self, dt: float) -> float:
        """更新膜污染，返回当前 TMP。"""
        avg_shear, _ = self.calculate_shear_stress()
        tmp: float = self.fouling.update(self.mlss, avg_shear, dt, backwash=self.backwash_flag)
        self.TMP_history.append(tmp)
        if len(self.TMP_history) > 3600:
            self.TMP_history.pop(0)
        self.backwash_flag = False
        return tmp

    def get_current_tmp(self) -> float:
        """获取当前跨膜压力 (kPa)。"""
        J_ms: float = self.fouling.flux / 3600.0
        tmp: float = PHYS.dynamic_viscosity * J_ms * (
            self.fouling.Rm + self.fouling.Rc + self.fouling.Rp) / 1000.0
        return round(min(tmp, PHYS.tmp_max), 2)

    def step_simulation(self, dt_hours: float) -> None:
        """推进仿真 dt_hours 小时。"""
        dt_sec: float = dt_hours * 3600.0
        self.update_sludge_level(dt_sec)
        self.update_fouling(dt_sec)
        self.sim_time += dt_sec

    # ----- 操作 -----
    def perform_backwash(self) -> None:
        """触发反洗。"""
        self.backwash_flag = True
        avg_shear, _ = self.calculate_shear_stress()
        self.fouling.update(self.mlss, avg_shear, 0.1, backwash=True)

    def discharge_sludge(self) -> None:
        """排泥操作。"""
        self.sludge_level = max(0.0, self.sludge_level - 0.08)
        self.is_discharging = False

    def reset(self) -> None:
        """重置仿真器到默认状态。"""
        self.__init__()

    # ----- 指标汇总 -----
    def get_metrics(self) -> SimulationMetrics:
        """汇总所有仿真指标。"""
        avg_shear, max_shear = self.calculate_shear_stress()
        real_count, _ = self.calculate_fiber_count()
        return SimulationMetrics(
            sec=self.calculate_sec(),
            shear_avg=avg_shear,
            shear_max=max_shear,
            uniformity=self.calculate_uniformity(),
            risk_text=self.calculate_risk_level(max_shear),
            svi=self.calculate_svi(),
            tss=self.calculate_tss(avg_shear),
            total_area=self.get_total_area(),
            fiber_count=real_count * PHYS.sheet_count,
            sludge_level_mm=self.sludge_level * 1000.0,
            sludge_percent=(self.sludge_level / PHYS.sludge_layer_max_height) * 100.0,
            tmp=self.get_current_tmp(),
            gas_hold_up=round(self.avg_gas_hold_up * 100, 1),
        )

    # ----- 状态快照 -----
    def get_state_snapshot(self) -> Dict:
        """导出完整状态快照，用于趋势预测或保存。"""
        return {
            "intensity": self.intensity,
            "mode": self.mode.value,
            "pulse_period": self.pulse_period,
            "p_pitch": self.p_pitch,
            "s_pitch": self.s_pitch,
            "h_size": self.h_size,
            "slack": self.slack,
            "mlss": self.mlss,
            "settling_rate": self.settling_rate,
            "return_ratio": self.return_ratio,
            "sludge_level": self.sludge_level,
            "sim_time": self.sim_time,
            "fouling": self.fouling.get_state(),
            "is_discharging": self.is_discharging,
            "pipe_to_membrane_gap": self.pipe_to_membrane_gap,
        }

    def load_state_snapshot(self, snap: Dict) -> None:
        """从状态快照恢复。"""
        for key in ("intensity", "pulse_period", "p_pitch", "s_pitch", "h_size",
                     "slack", "mlss", "settling_rate", "return_ratio",
                     "sludge_level", "sim_time", "is_discharging",
                     "pipe_to_membrane_gap"):
            if key in snap:
                setattr(self, key, snap[key])
        if "mode" in snap:
            self.mode = AerationMode(snap["mode"])
        if "fouling" in snap:
            self.fouling.set_state(snap["fouling"])

    # ----- 趋势预测 -----
    def predict_trend(self, hours: float = 12.0, steps: int = 50) -> pd.DataFrame:
        """
        基于当前状态进行趋势预测。
        返回包含 shear_avg, sludge_mm, tmp_kpa 各时间点的 DataFrame。
        """
        state = self.get_state_snapshot()
        temp = EnhancedMBRSimulator()
        temp.load_state_snapshot(state)
        temp.is_discharging = False

        times = np.linspace(0, hours, steps)
        shear_vals: List[float] = []
        sludge_vals: List[float] = []
        tmp_vals: List[float] = []

        step_hours = hours / (steps - 1)
        for _ in times:
            avg, _ = temp.calculate_shear_stress()
            shear_vals.append(avg)
            sludge_vals.append(temp.sludge_level * 1000.0)
            tmp_vals.append(temp.get_current_tmp())
            temp.step_simulation(step_hours)

        return pd.DataFrame({
            "time_h": times,
            "shear_pa": shear_vals,
            "sludge_mm": sludge_vals,
            "tmp_kpa": tmp_vals,
        })


# ==================== 3D 可视化生成器 ====================
def generate_3d_html(sim: EnhancedMBRSimulator) -> str:
    """
    生成 MBR 帘式膜组件 3D 可视化 HTML。
    参考实物：水平放置的 PVDF 中空纤维膜模块，白色外壳，膜丝束垂直悬挂，
    两侧各有产水集水管，底部曝气管。
    """
    sheet_count: int = PHYS.sheet_count
    sheet_width: float = PHYS.sheet_width    # 膜架宽度（X）
    sheet_spacing: float = sim.s_pitch / 1000.0  # 帘间距（Z）
    fiber_len: float = sim.f_len            # 膜丝有效长度（Y）
    fiber_diameter_m: float = sim.fiber_diameter / 1000.0
    sludge_height: float = sim.sludge_level
    sludge_mm: float = sim.sludge_level * 1000.0
    pipe_gap: float = sim.pipe_to_membrane_gap / 1000.0  # mm -> m
    pipe_offset: float = PHYS.pipe_offset
    slack_amount: float = sim.slack * 1.5   # 膜丝松弛程度

    # 膜组件壳体尺寸（参考实物比例）
    module_height: float = fiber_len + 0.3  # 总高（含集水管）
    module_depth: float = 0.08              # 壳体厚度（Z）
    header_height: float = 0.05             # 集水管高度
    header_width: float = 0.04               # 集水管宽度（两侧）

    # 帘间距（沿 Z 轴）
    total_depth: float = sheet_count * module_depth + (sheet_count - 1) * sheet_spacing

    # 相机位置
    cam_x: float = sheet_width * 0.5
    cam_y: float = module_height * 0.5
    cam_z: float = total_depth + 3.0
    slack_amount: float = sim.slack * 1.5

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<style>
  body {{ margin: 0; overflow: hidden; background: #0d1117; font-family: sans-serif; }}
  #info {{ position: absolute; top: 10px; left: 20px; color: #c9d1d9; font-size: 13px; line-height: 1.6; z-index: 10; }}
  .legend {{ position: absolute; bottom: 20px; right: 20px; color: #8b949e; font-size: 12px; background: rgba(13,17,23,0.7); padding: 8px 12px; border-radius: 4px; z-index: 10; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; border-radius: 2px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <b>MBR 帘式膜组件 3D 视图</b><br>
  膜丝外径: {sim.fiber_diameter} mm | 膜丝长: {fiber_len:.2f} m | 帘数: {sheet_count} | 间距: {sim.s_pitch} mm<br>
  污泥层: {sludge_mm:.0f} mm | MLSS: {sim.mlss} mg/L | SRT: {sim.srt} d
</div>
<div class="legend">
  <span style="background:#eeeeee"></span> 膜壳 &nbsp;
  <span style="background:#4499ff"></span> 膜丝 &nbsp;
  <span style="background:#66aadd"></span> 产水集水管 &nbsp;
  <span style="background:#ff8844"></span> 曝气管
</div>
<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const SW      = {sheet_width:.3f};
const FL      = {fiber_len:.3f};
const SS      = {sheet_spacing:.3f};
const SC      = {sheet_count};
const PO      = {pipe_offset:.3f};
const TD      = {total_depth:.3f};
const MH      = {module_height:.3f};
const MD      = {module_depth:.3f};
const HH      = {header_height:.3f};
const HW      = {header_width:.3f};
const SH      = {sludge_height:.3f};
const PG      = {pipe_gap:.3f};
const SLACK   = {slack_amount:.3f};
const FD      = {fiber_diameter_m:.5f};
const VFC     = 6;   // X 方向膜丝数（示意）
const VFC_Z   = 3;   // Z 方向膜层数（示意）

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 8, 30);

const bottomY = -(PG + 0.3);  // 池底坐标

const camera = new THREE.PerspectiveCamera(45, 1.6, 0.1, 50);
camera.position.set(SW * 0.3, MH * 0.5, TD + 3.5);
camera.lookAt(SW * 0.5, FL * 0.3, TD * 0.5);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = false;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(SW * 0.5, FL * 0.3, TD * 0.5);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 1.0;
controls.maxDistance = 20;
controls.update();

// ---- 光照 ----
scene.add(new THREE.AmbientLight(0x8090b0, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(TD + 4, MH + 3, TD + 3);
scene.add(sun);
const fillLight = new THREE.PointLight(0x4488cc, 1.0, 10);
fillLight.position.set(SW * 0.5, FL * 0.5, TD * 0.5);
scene.add(fillLight);

// ---- 网格地面（池底）----
const grid = new THREE.GridHelper(Math.max(SW, TD) + 2, 24, 0x334455, 0x1e2a38);
grid.position.set(SW * 0.5, bottomY - 0.01, TD * 0.5);
scene.add(grid);

// ---- 材质定义 ----
const caseMat   = new THREE.MeshStandardMaterial({{ color: 0xdddddd, metalness: 0.05, roughness: 0.6 }});
const caseDarkMat = new THREE.MeshStandardMaterial({{ color: 0xcccccc, metalness: 0.1, roughness: 0.5 }});
const fiberMat  = new THREE.MeshStandardMaterial({{ color: 0x4499ff, metalness: 0.02, roughness: 0.6, transparent: true, opacity: 0.82 }});
const headerMat = new THREE.MeshStandardMaterial({{ color: 0x66aadd, metalness: 0.4, roughness: 0.3 }});
const pipeMat   = new THREE.MeshStandardMaterial({{ color: 0xff8844, metalness: 0.4, roughness: 0.4 }});
const sludgeMat = new THREE.MeshStandardMaterial({{ color: 0x775522, metalness: 0, roughness: 1.0, transparent: true, opacity: 0.45 }});

// 预生成膜丝几何体（复用 InstancedMesh）
const segs = 4;
const segGeo = new THREE.CylinderGeometry(FD * 0.5, FD * 0.5, FL / segs, 4);

// ---- 逐个膜帘组件 ----
for (let si = 0; si < SC; si++) {{
  const cz = si * (MD + SS);  // 当前帘的 Z 中心
  const yTop = FL + 0.15;     // 上集水管中心 Y

  // ===== 白色膜壳（前后两块侧板 + 顶部封板，底部开放浸入污泥中）====
  // 前板
  const frontGeo = new THREE.BoxGeometry(SW, FL, 0.006);
  const front = new THREE.Mesh(frontGeo, caseMat);
  front.position.set(SW * 0.5, FL * 0.5, cz + MD * 0.5);
  front.castShadow = true; front.receiveShadow = true;
  scene.add(front);
  // 后板
  const back = new THREE.Mesh(frontGeo, caseMat);
  back.position.set(SW * 0.5, FL * 0.5, cz - MD * 0.5);
  back.castShadow = true; back.receiveShadow = true;
  scene.add(back);
  // 顶部封板
  const topGeo = new THREE.BoxGeometry(SW, 0.012, MD);
  const top = new THREE.Mesh(topGeo, caseMat);
  top.position.set(SW * 0.5, yTop, cz);
  scene.add(top);

  // ===== 上部产水集水管（水平，X 方向，两侧突出壳体）====
  const topHeaderGeo = new THREE.CylinderGeometry(HH * 0.5, HH * 0.5, SW + HW * 2, 14);
  const topHeader = new THREE.Mesh(topHeaderGeo, headerMat);
  topHeader.rotation.z = Math.PI / 2;
  topHeader.position.set(SW * 0.5, yTop, cz);
  topHeader.castShadow = true;
  scene.add(topHeader);

  // ===== 下部集水管（连接膜丝底部，Y=0 处）====
  const botHeaderGeo = new THREE.CylinderGeometry(HH * 0.4, HH * 0.4, SW + HW * 0.5, 14);
  const botHeader = new THREE.Mesh(botHeaderGeo, headerMat);
  botHeader.rotation.z = Math.PI / 2;
  botHeader.position.set(SW * 0.5, 0.04, cz);
  scene.add(botHeader);

  // ===== 两侧产水连接管（Z 方向，通向端部集管）====
  const sideTubeGeo = new THREE.CylinderGeometry(HH * 0.35, HH * 0.35, MD * 0.5, 10);
  for (let side = -1; side <= 1; side += 2) {{
    const st = new THREE.Mesh(sideTubeGeo, headerMat);
    st.rotation.x = Math.PI / 2;
    st.position.set(side > 0 ? SW * 0.98 : SW * 0.02, yTop - HH * 0.3, cz + side * MD * 0.25);
    scene.add(st);
  }}

  // ===== 中空纤维膜丝（使用 InstancedMesh，大幅减少对象）====
  const fiberSegCount = VFC * VFC_Z * segs;
  const fiberMesh = new THREE.InstancedMesh(segGeo, fiberMat, fiberSegCount);
  const dummy = new THREE.Object3D();
  let idx = 0;
  for (let fi = 0; fi < VFC; fi++) {{
    const fx = (fi + 0.5) / VFC * (SW - HW * 1.5) + HW * 0.75;
    const bx = (Math.random() - 0.5) * SLACK;
    const bz = (Math.random() - 0.5) * SLACK * 0.4;
    for (let fz_i = 0; fz_i < VFC_Z; fz_i++) {{
      const fz = cz - MD * 0.45 + (fz_i + 0.5) / VFC_Z * MD * 0.9;
      for (let s = 0; s < segs; s++) {{
        const t = (s + 0.5) / segs;
        const curve = Math.sin(t * Math.PI);
        dummy.position.set(fx + bx * curve, s * (FL / segs) + (FL / segs) * 0.5, fz + bz * curve);
        dummy.updateMatrix();
        fiberMesh.setMatrixAt(idx++, dummy.matrix);
      }}
    }}
  }}
  fiberMesh.instanceMatrix.needsUpdate = true;
  scene.add(fiberMesh);
}}

// ---- 曝气管（膜片下方，距离可调）----
const pipeR = 0.018;
const pipeGeo = new THREE.CylinderGeometry(pipeR, pipeR, SW * 0.85, 10);
const pipeY = -PG;  // 膜片底部(Y=0)下方 pipe_gap 处
for (let si = 0; si < SC; si++) {{
  const cz = si * (MD + SS);
  // 主管
  const mainPipe = new THREE.Mesh(pipeGeo, pipeMat);
  mainPipe.rotation.z = Math.PI / 2;
  mainPipe.position.set(SW * 0.5, pipeY, cz);
  scene.add(mainPipe);
  // 曝气支管（从主管向上喷向膜丝底部）
  const brGeo = new THREE.CylinderGeometry(pipeR * 0.5, pipeR * 0.5, PG * 0.7, 6);
  for (let pi = 0; pi < 4; pi++) {{
    const px = SW * 0.18 + pi * SW * 0.18;
    const br = new THREE.Mesh(brGeo, pipeMat);
    br.position.set(px, pipeY + PG * 0.35, cz);
    scene.add(br);
  }}
}}

// ---- 污泥层（从池底向上堆至泥水分界面）----
const sludgeTopY = bottomY + SH;  // 污泥层顶面
const midY = (bottomY + sludgeTopY) / 2;
const sludgeGeo = new THREE.BoxGeometry(SW * 1.1, SH, TD * 1.1);
const sludge = new THREE.Mesh(sludgeGeo, sludgeMat);
sludge.position.set(SW * 0.5, midY, TD * 0.5);
sludge.receiveShadow = true;
scene.add(sludge);

// 污泥-水交界线
const waterLineGeo = new THREE.BoxGeometry(SW * 1.1, 0.005, TD * 1.1);
const waterMat = new THREE.MeshBasicMaterial({{ color: 0x1a4a6e, transparent: true, opacity: 0.3 }});
const waterLine = new THREE.Mesh(waterLineGeo, waterMat);
waterLine.position.set(SW * 0.5, sludgeTopY, TD * 0.5);
scene.add(waterLine);

// ---- 气泡（从曝气管喷出，穿过膜丝间上升）----
const bGeo = new THREE.SphereGeometry(0.008, 5, 5);
const bMat = new THREE.MeshBasicMaterial({{ color: 0xaaddff, transparent: true, opacity: 0.5 }});
const bubbles = [];
const bubbleMaxY = FL + 0.15;  // 气泡升到膜片顶部以上
const bubbleMinY = pipeY;      // 气泡从曝气管处产生
for (let i = 0; i < 50; i++) {{
  const b = new THREE.Mesh(bGeo, bMat);
  const si = Math.floor(Math.random() * SC);
  b.position.set(
    SW * 0.1 + Math.random() * SW * 0.8,
    bubbleMinY + Math.random() * (bubbleMaxY - bubbleMinY),
    si * (MD + SS) + Math.random() * MD
  );
  b.userData = {{ speed: 0.004 + Math.random() * 0.008, ox: b.position.x, oz: b.position.z, phase: Math.random() * Math.PI * 2 }};
  scene.add(b);
  bubbles.push(b);
}}

// 动画
function animate(time) {{
  requestAnimationFrame(animate);
  const t = time * 0.001;
  bubbles.forEach(b => {{
    b.position.y += b.userData.speed;
    b.position.x = b.userData.ox + Math.sin(t * 2 + b.userData.phase) * 0.003;
    b.position.z = b.userData.oz + Math.cos(t * 1.5 + b.userData.phase) * 0.002;
    if (b.position.y > bubbleMaxY) {{
      b.position.y = bubbleMinY;
      const si = Math.floor(Math.random() * SC);
      b.userData.oz = si * (MD + SS) + Math.random() * MD;
      b.userData.ox = SW * 0.1 + Math.random() * SW * 0.8;
      b.position.x = b.userData.ox;
      b.position.z = b.userData.oz;
    }}
  }});
  controls.update();
  renderer.render(scene, camera);
}}
requestAnimationFrame(animate);

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


# ==================== Streamlit UI ====================
def init_session() -> None:
    """初始化 session state。"""
    if "sim" not in st.session_state:
        st.session_state.sim = EnhancedMBRSimulator()
    if "preset_key" not in st.session_state:
        st.session_state.preset_key = "flush"


def render_sidebar(sim: EnhancedMBRSimulator) -> None:
    """渲染侧边栏控制面板。"""
    with st.sidebar:
        st.header("🔬 MBR 系统控制")

        # 预设按钮
        st.subheader("📋 预设方案")
        preset_names = list(PRESETS.keys())
        cols = st.columns(len(preset_names))
        for i, (key, preset) in enumerate(PRESETS.items()):
            with cols[i]:
                if st.button(f"{preset.icon} {preset.label}", key=f"preset_{key}",
                             use_container_width=True,
                             type="secondary" if st.session_state.get("preset_key") != key else "primary"):
                    sim.apply_preset(key)
                    st.session_state.preset_key = key
                    st.rerun()

        st.divider()

        # 曝气参数
        st.subheader("🌊 曝气参数")
        mode_value: str = st.selectbox(
            "曝气模式",
            [m.value for m in AerationMode],
            format_func=lambda x: AerationMode(x).label,
            index=[m.value for m in AerationMode].index(sim.mode.value),
        )
        sim.mode = AerationMode(mode_value)
        sim.intensity = st.slider("曝气强度 (Nm³/m²/h)", 50, 150, int(sim.intensity), step=5)
        if sim.mode == AerationMode.PULSE:
            sim.pulse_period = st.slider("脉冲周期 (s)", 2.0, 8.0, float(sim.pulse_period), step=0.5)
        sim.h_size = st.slider("曝气孔径 (mm)", 1.0, 15.0, float(sim.h_size), step=0.5)
        sim.p_pitch = st.slider("曝气管间距 (mm)", 30, 300, int(sim.p_pitch), step=10)
        sim.pipe_to_membrane_gap = st.slider("曝气-膜片距离 (mm)", 100, 500, int(sim.pipe_to_membrane_gap), step=10)

        st.divider()

        # 膜片参数
        st.subheader("🧬 膜片参数")
        sim.fiber_diameter = st.selectbox(
            "膜丝外径", [1.65, 2.8],
            format_func=lambda x: f"{x} mm → {'40' if x == 1.65 else '25'} m²/片",
        )
        sim.thickness = st.slider("膜片厚度 (mm)", 10, 100, int(sim.thickness), step=5)
        sim.s_pitch = st.slider("膜片间距 (mm)", 30, 120, int(sim.s_pitch), step=5)
        sim.f_len = st.slider("膜丝长度 (m)", 0.5, 3.0, float(sim.f_len), step=0.1)
        sim.slack = st.slider("松弛度 (%)", 0.2, 5.0,
                              value=float(round(sim.slack * 100, 1)),
                              step=0.2) / 100.0

        st.divider()

        # 污泥与操作
        st.subheader("🧫 污泥与操作")
        sim.mlss = st.slider("MLSS (mg/L)", 2000, 15000, int(sim.mlss), step=500)
        sim.srt = st.slider("污泥龄 (d)", 5, 40, int(sim.srt), step=1)
        sim.settling_rate = st.slider("沉降速率 (m/h)", 0.5, 6.0, float(sim.settling_rate), step=0.5)
        sim.return_ratio = st.slider("回流比 (%)", 50, 300, int(sim.return_ratio), step=10)

        # 操作按钮
        op_cols = st.columns(4)
        with op_cols[0]:
            if st.button("⬇️ 排泥", use_container_width=True, help="排出污泥，降低污泥层高度"):
                sim.discharge_sludge()
                st.rerun()
        with op_cols[1]:
            if st.button("🧼 反洗", use_container_width=True, help="反洗膜组件，清除滤饼层"):
                sim.perform_backwash()
                st.rerun()
        with op_cols[2]:
            if st.button("⏱ +1h", use_container_width=True, help="模拟运行 1 小时"):
                sim.step_simulation(1.0)
                st.rerun()
        with op_cols[3]:
            if st.button("🔄 重置", use_container_width=True, help="恢复到默认高冲刷模式"):
                st.session_state.sim = EnhancedMBRSimulator()
                st.rerun()

        # 污泥层进度
        sludge_percent: float = sim.sludge_level / PHYS.sludge_layer_max_height * 100
        st.progress(
            min(100, int(sludge_percent)),
            text=f"污泥层 {sim.sludge_level * 1000:.0f} mm ({sludge_percent:.0f}%)"
        )

        # 批量模拟
        st.divider()
        st.subheader("⏩ 批量推进")
        batch_cols = st.columns(3)
        for i, h in enumerate([2, 6, 12]):
            with batch_cols[i]:
                if st.button(f"+{h}h", key=f"batch_{h}h", use_container_width=True):
                    sim.step_simulation(float(h))
                    st.rerun()


def render_metrics(metrics: SimulationMetrics) -> None:
    """渲染指标卡片。"""
    rows_config = [
        [
            ("能耗 SEC", f"{metrics.sec} kWh/m³", None),
            ("平均剪切力", f"{metrics.shear_avg} Pa", None),
            ("最大剪切力", f"{metrics.shear_max} Pa", None),
            ("覆盖均匀度", f"{metrics.uniformity} %", None),
        ],
        [
            ("积垢风险", metrics.risk_text,
             "delta" if metrics.risk_text == "HIGH" else None),
            ("SVI", f"{metrics.svi} mL/g", None),
            ("TSS", f"{metrics.tss} mg/L", None),
            ("总膜面积", f"{metrics.total_area} m²", None),
        ],
        [
            ("TMP 跨膜压力", f"{metrics.tmp} kPa",
             ">35需反洗" if metrics.tmp > 35 else None),
            ("气含率", f"{metrics.gas_hold_up} %", None),
            ("膜丝数量", f"{metrics.fiber_count:,} 根", None),
        ],
    ]

    for row in rows_config:
        cols = st.columns(len(row))
        for col, (label, value, delta) in zip(cols, row):
            with col:
                kwargs = {"label": label, "value": value}
                if delta:
                    if isinstance(delta, str):
                        kwargs["delta"] = delta
                    else:
                        kwargs["delta_color"] = "inverse"
                st.metric(**kwargs)


def render_trend_chart(sim: EnhancedMBRSimulator) -> None:
    """渲染趋势预测图。"""
    with st.expander("📈 12小时趋势预测 (剪切力 & 污泥层 & TMP)", expanded=False):
        df: pd.DataFrame = sim.predict_trend(hours=12.0, steps=50)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("剪切力 (Pa)", "污泥层高度 (mm)", "跨膜压力 TMP (kPa)"),
        )
        fig.add_trace(
            go.Scatter(x=df["time_h"], y=df["shear_pa"], mode="lines+markers",
                       name="剪切力", line=dict(color="#00f2ff")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["time_h"], y=df["sludge_mm"], mode="lines",
                       name="污泥层", line=dict(color="#d4a84b")),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["time_h"], y=df["tmp_kpa"], mode="lines",
                       name="TMP", line=dict(color="#ff6644")),
            row=3, col=1,
        )

        # 添加阈值线
        fig.add_hline(y=35, line_dash="dash", line_color="#ff6644",
                      annotation_text="TMP 反洗阈值", row=3, col=1)

        fig.update_layout(
            height=550, template="plotly_dark",
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="模拟时间 (小时)", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)


def render_export_panel(sim: EnhancedMBRSimulator) -> None:
    """渲染导出面板。"""
    with st.expander("💾 导出与数据", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # 导出 JSON 配置
            config_json: str = json.dumps(
                sim.get_state_snapshot(), indent=2, ensure_ascii=False)
            st.download_button(
                "📄 导出配置 (JSON)",
                data=config_json,
                file_name="mbr_config.json",
                mime="application/json",
            )

        with col2:
            # 导出 CSV 趋势数据
            df: pd.DataFrame = sim.predict_trend(hours=12.0, steps=50)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📊 导出趋势预测 (CSV)",
                data=csv_buffer.getvalue(),
                file_name="mbr_trend_prediction.csv",
                mime="text/csv",
            )

        # 上传配置
        uploaded = st.file_uploader("📂 导入配置文件", type=["json"], key="config_upload")
        if uploaded is not None:
            try:
                snap = json.loads(uploaded.read())
                sim.load_state_snapshot(snap)
                st.success("配置导入成功！")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败: {e}")


def render_sensitivity_analysis(sim: EnhancedMBRSimulator) -> None:
    """渲染参数敏感性分析。"""
    with st.expander("🔬 参数敏感性分析", expanded=False):
        param = st.selectbox(
            "选择分析参数",
            ["intensity", "p_pitch", "slack", "mlss", "return_ratio"],
            format_func={
                "intensity": "曝气强度",
                "p_pitch": "曝气管间距",
                "slack": "松弛度",
                "mlss": "MLSS",
                "return_ratio": "回流比",
            }.get,
        )

        # 扫描范围
        base_val = getattr(sim, param)
        scan_range = {
            "intensity": np.linspace(50, 150, 20),
            "p_pitch": np.linspace(30, 300, 20),
            "slack": np.linspace(0.005, 0.05, 20),
            "mlss": np.linspace(2000, 15000, 20),
            "return_ratio": np.linspace(50, 300, 20),
        }[param]

        shear_list, sec_list, tmp_list = [], [], []
        state = sim.get_state_snapshot()
        temp = EnhancedMBRSimulator()

        for val in scan_range:
            temp.load_state_snapshot(state)
            setattr(temp, param, type(base_val)(val))
            avg_s, _ = temp.calculate_shear_stress()
            shear_list.append(avg_s)
            sec_list.append(temp.calculate_sec())
            tmp_list.append(temp.get_current_tmp())

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("剪切力 (Pa)", "能耗 SEC (kWh/m³)", "TMP (kPa)"))
        fig.add_trace(go.Scatter(x=scan_range, y=shear_list, mode="lines+markers",
                                 name="剪切力", line=dict(color="#00f2ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=scan_range, y=sec_list, mode="lines+markers",
                                 name="SEC", line=dict(color="#ffaa00")), row=2, col=1)
        fig.add_trace(go.Scatter(x=scan_range, y=tmp_list, mode="lines+markers",
                                 name="TMP", line=dict(color="#ff6644")), row=3, col=1)

        # 标注当前值
        for row_idx in range(1, 4):
            fig.add_vline(x=base_val, line_dash="dash", line_color="#ffffff",
                          opacity=0.5, row=row_idx, col=1,
                          annotation_text=f"当前={base_val:.2f}")

        fig.update_layout(height=550, template="plotly_dark", hovermode="x unified")
        fig.update_xaxes(title_text=param, row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)


# ==================== 主入口 ====================
def main() -> None:
    """MBR 仿真系统主入口。"""
    st.set_page_config(
        page_title="MBR 工程级仿真系统 v3.0",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()
    sim: EnhancedMBRSimulator = st.session_state.sim

    # 侧边栏
    render_sidebar(sim)

    # 主区域
    st.title("💧 MBR 工程级仿真系统 v3.0")
    st.caption(
        "增强物理模型：Vesilind 沉降 | 气泡剪切 | 膜污染(TMP) | 气含率 | "
        "曝气能耗 | 参数敏感性分析 | 配置导入导出"
    )

    # 指标卡片
    metrics: SimulationMetrics = sim.get_metrics()
    render_metrics(metrics)

    # 3D 可视化
    st.markdown("### 🖥️ 3D 可视化视图")
    html_code: str = generate_3d_html(sim)
    st.components.v1.html(html_code, height=600, scrolling=False)

    # 趋势预测
    render_trend_chart(sim)

    # 敏感性分析
    render_sensitivity_analysis(sim)

    # 导出面板
    render_export_panel(sim)


if __name__ == "__main__":
    main()