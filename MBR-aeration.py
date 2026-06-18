"""
MBR 工程级仿真系统 v4.0 — OpenFOAM 物理模型增强版
=====================================================
新增 OpenFOAM 物理模型:
  • 气泡群动力学 (Population Balance Model / PBM)
  • Drift-Flux 双流体模型 (Zuber-Findlay 相关性)
  • k-ε 湍流修正近壁剪切应力
  • DO 溶解氧 + MLR 混合液黏度模型
  • 膜污染: 总阻力模型 + 滤饼层压缩 + 孔堵
  • OpenFOAM 算例自动生成器 (blockMeshDict + transportProperties)
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from math import erfc, exp, log, pi, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ==================== 枚举定义 ====================
class AerationMode(Enum):
    CONTINUOUS = "cont"
    PULSE = "pulse"

    @property
    def label(self) -> str:
        return "连续曝气" if self == AerationMode.CONTINUOUS else "脉冲式曝气"


# ==================== OpenFOAM 物理常数 ====================
@dataclass(frozen=True)
class PhysicsConstants:
    """统一物理常数 (OpenFOAM 风格)"""

    # 膜架几何
    sheet_count: int = 5
    sheet_width: float = 1.25
    pipe_offset: float = 0.45

    # ---- 流体物性 (20°C 纯水) ----
    rho_l: float = 998.0       # 液相密度 kg/m³
    mu_l: float = 1.0e-3      # 动力黏度 Pa·s
    sigma: float = 0.072       # 表面张力 N/m

    # ---- 气泡群 (PBM) ----
    d_bubble_min: float = 1.0e-3   # 最小气泡径 mm
    d_bubble_max: float = 6.0e-3   # 最大气泡径 mm
    d_bubble_ref: float = 2.5e-3   # 参考气泡径 mm
    n_bins: int = 8               # PBM 粒径分组数
    breakup_C: float = 0.25        # 破碎常数 C_B
    coalescence_C: float = 0.10    # 聚并常数 C_C

    # ---- Drift-Flux 模型 ----
    C0: float = 1.0              # 分布系数 (Zuber-Findlay)
    V_drift: float = 0.25         # 漂移速度 m/s (2mm 气泡)

    # ---- 湍流 k-ε 模型 ----
    C_mu: float = 0.09
    C1_epsilon: float = 1.44
    C2_epsilon: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3

    # ---- 曝气剪切 ----
    uniform_penalty_factor: float = 0.4
    pulse_power_boost: float = 1.4
    off_phase_power: float = 0.05

    # ---- 膜污染 ----
    membrane_resistance: float = 2.0e11
    fouling_rate_const: float = 1.0e-5
    backwash_efficiency: float = 0.9
    tmp_max: float = 60.0
    critical_tmp: float = 35.0      # 反洗阈值 kPa

    # ---- 沉降 ----
    sludge_layer_max_height: float = 0.6
    sludge_discharge_rate: float = 0.02
    vesilind_v0: float = 7.0       # Vesilind 参数 m/h
    vesilind_k: float = 0.6         # Vesilind 参数
    compression_index: float = 0.2

    # ---- MLR 黏度 ----
    mu_max_factor: float = 8.0      # 最大黏度倍数 (高 MLSS 时)


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
    temperature: float           # 水温 °C


PRESETS: Dict[str, Preset] = {
    "eco": Preset(
        name="eco", label="节能模式", icon="🌿",
        intensity=60, pulse_period=6.0, p_pitch=250, s_pitch=100,
        h_size=6.0, f_len=2.0, slack=0.008, mode=AerationMode.CONTINUOUS,
        fiber_diameter=2.8, thickness=30, mlss=6000, srt=20,
        settling_rate=2.0, return_ratio=80,
        pipe_to_membrane_gap=400, temperature=20.0,
    ),
    "balanced": Preset(
        name="balanced", label="均衡模式", icon="⚖️",
        intensity=80, pulse_period=4.0, p_pitch=100, s_pitch=80,
        h_size=4.0, f_len=2.0, slack=0.015, mode=AerationMode.CONTINUOUS,
        fiber_diameter=1.65, thickness=30, mlss=8000, srt=15,
        settling_rate=2.5, return_ratio=100,
        pipe_to_membrane_gap=250, temperature=20.0,
    ),
    "flush": Preset(
        name="flush", label="高冲刷模式", icon="💨",
        intensity=110, pulse_period=3.0, p_pitch=50, s_pitch=50,
        h_size=2.5, f_len=2.0, slack=0.025, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=30, mlss=10000, srt=12,
        settling_rate=3.0, return_ratio=150,
        pipe_to_membrane_gap=150, temperature=20.0,
    ),
    "low_fouling": Preset(
        name="low_fouling", label="低污染模式", icon="🛡️",
        intensity=90, pulse_period=5.0, p_pitch=80, s_pitch=60,
        h_size=3.0, f_len=2.0, slack=0.02, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=40, mlss=6000, srt=25,
        settling_rate=2.0, return_ratio=120,
        pipe_to_membrane_gap=300, temperature=20.0,
    ),
    "high_flux": Preset(
        name="high_flux", label="高通量模式", icon="⚡",
        intensity=130, pulse_period=2.5, p_pitch=40, s_pitch=40,
        h_size=2.0, f_len=2.5, slack=0.03, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=25, mlss=12000, srt=10,
        settling_rate=3.5, return_ratio=200,
        pipe_to_membrane_gap=150, temperature=20.0,
    ),
}


# ==================== 数据类 ====================
@dataclass
class SimulationMetrics:
    """仿真指标汇总 (OpenFOAM 输出风格)"""
    # 能耗
    sec: float = 0.0          # kWh/m³
    power_w: float = 0.0      # W
    # 气泡群
    d_32: float = 0.0         # 索太尔平均直径 mm
    gas_velocity: float = 0.0 # 表观气速 m/s
    gas_holdup: float = 0.0   # 气含率
    # 剪切 (湍流 k-ε)
    shear_avg: float = 0.0    # Pa
    shear_max: float = 0.0    # Pa
    k_turb: float = 0.0       # 湍动能 m²/s²
    epsilon_turb: float = 0.0 # 湍流耗散率 m²/s³
    # 均匀度
    uniformity: float = 0.0   # %
    risk_text: str = "N/A"
    # 污泥
    mlvss: float = 0.0        # mg/L
    svi: float = 0.0          # mL/g
    tss: float = 0.0          # mg/L
    mlr: float = 1.0          # 混合液相对黏度
    # 膜
    total_area: float = 0.0    # m²
    fiber_count: int = 0
    tmp: float = 0.0           # kPa
    Rm: float = 0.0
    Rc: float = 0.0
    Rp: float = 0.0
    # 溶解氧
    do_level: float = 0.0     # mg/L
    # 污泥层
    sludge_level_mm: float = 0.0
    sludge_percent: float = 0.0

    def to_dict(self) -> Dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


# ==================== OpenFOAM 物理模型组件 ====================

class BubblePopulationBalance:
    """
    气泡群动力学模型 (Population Balance Model, PBM)
    离散分段法: 将气泡尺寸分布划分为 n_bins 个区间
    输运方程: d(N_i)/dt + ∇·(U_g N_i) = B_breakup - D_coalescence + S_i

    破碎核函数: g(d) = C_B * (ε/d)^(1/3)
    聚并核函数: h(d) = C_C * d^2 * ε^(1/3)
    """

    def __init__(self, n_bins: int = PHYS.n_bins,
                 d_min: float = PHYS.d_bubble_min,
                 d_max: float = PHYS.d_bubble_max) -> None:
        self.n_bins = n_bins
        self.d_min = d_min
        self.d_max = d_max
        # 对数等间距划分粒径区间
        self.bin_edges: np.ndarray = np.logspace(
            log(d_min), log(d_max), n_bins + 1)
        # 各区间代表直径 (几何平均)
        self.bin_centers: np.ndarray = np.sqrt(
            self.bin_edges[:-1] * self.bin_edges[1:])
        # 各区间气泡数密度 N(d_i) [1/m³/m]
        self.N: np.ndarray = np.ones(n_bins) * 1e7
        self.epsilon: float = 0.1   # 湍流耗散率 m²/s³

    def update(self, epsilon_turb: float, gas_vel: float, dt: float) -> None:
        """推进 PBM 一步。"""
        self.epsilon = epsilon_turb
        eps13 = epsilon_turb ** (1.0 / 3.0) if epsilon_turb > 1e-10 else 1e-4

        dNdt = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            di = self.bin_centers[i]
            # 破碎率 (单位: 1/s)
            breakup_rate = PHYS.breakup_C * (epsilon_turb / di) ** (1.0 / 3.0)
            # 聚并率
            coalescence_rate = PHYS.coalescence_C * di ** 2 * eps13
            # 来自更大气泡破碎产生的 i 区间气泡
            for j in range(i + 1, self.n_bins):
                dj = self.bin_centers[j]
                # 假设破碎为二元对称破碎
                parent_bin = np.searchsorted(self.bin_edges, dj * 2 ** (-1/3))
                if parent_bin < self.n_bins:
                    source = PHYS.breakup_C * (epsilon_turb / dj) ** (1/3)
                    dNdt[i] += source * self.N[j] * 0.5
            # 聚并汇: i + j -> k
            for j in range(self.n_bins):
                if i != j:
                    dk = (di ** 3 + self.bin_centers[j] ** 3) ** (1/3)
                    if dk <= self.d_max:
                        dNdt[i] -= coalescence_rate * self.N[j] * self.N[i] * 1e-9
            # 自身破碎损失
            dNdt[i] -= breakup_rate * self.N[i]
            # 气泡上升排出
            dNdt[i] -= gas_vel / 2.0 * self.N[i]

        self.N = np.clip(self.N + dNdt * dt, 1.0, 1e10)

    def get_sauter_diameter(self) -> float:
        """
        计算索太尔平均直径 d_32 (Sauter Mean Diameter)
        d_32 = Σ n_i d_i^3 / Σ n_i d_i^2
        """
        n = self.N
        d = self.bin_centers
        num = np.sum(n * d ** 3)
        den = np.sum(n * d ** 2)
        if den < 1e-20:
            return self.d_max
        return num / den

    def get_bubble_velocity(self, d: float) -> float:
        """
        单气泡终端速度 (Haberman-Morton / Goodman 图谱拟合)
        适用: 1-8 mm 气泡在水中的终端速度
        """
        d_m = d
        g = 9.81
        # 无量纲 Bond 数
        Bo = (998.0 ** 2) * g * d_m ** 3 / PHYS.sigma / 998.0
        if d_m < 1e-3:
            # Stokes 区 (d < 0.1mm)
            return (998.0 - 1.2) * g * d_m ** 2 / 18.0 / 1.8e-5
        elif d_m < 2e-3:
            # 过渡区
            v = 0.23 * sqrt(g * d_m)
            return min(v, 0.25)
        else:
            # 惯性主导区 (Haberman-Morton)
            v = sqrt(2.14 * 1.2 * g * d_m + 0.505 * (998.0 - 1.2) * g * d_m)
            return min(v, 0.40)

    def get_distribution(self) -> Dict:
        """返回各区间数量密度。"""
        return {f"{float(d)*1e3:.1f}mm": float(n)
                for d, n in zip(self.bin_centers, self.N)}


class DriftFluxModel:
    """
    Drift-Flux 双流体模型 (OpenFOAM twoPhaseEulerFoam 核心理论)

    气含率方程:
    ∂α_g/∂t + ∇·(α_g ⟨U⟩) = -∇·(C0 ⟨U⟩ α_g (1-α_g) + C0 V_drift α_g)

    Zuber-Findlay 相关性:
    ⟨U_g⟩ = C0 (⟨U_g⟩ + ⟨U_l⟩) + V_drift
    α_g = ⟨U_g⟩ / (C0 (⟨U_g⟩ + ⟨U_l⟩) + V_drift)

    其中:
    C0 = 1.0 + 0.35 (1-α_g)  (界面浓度分布修正)
    V_drift = 0.25 (1 - α_g)^0.5 m/s  (漂移速度)
    """

    def __init__(self) -> None:
        self.alpha_g: float = 0.05     # 气含率
        self.v_drift: float = 0.25     # 漂移速度 m/s
        self.C0: float = 1.0          # 分布系数

    def update(self, superficial_gas_vel: float,
               superficial_liquid_vel: float = 0.0,
               dt: float = 1.0) -> float:
        """
        根据表观气速更新气含率

        参数:
            superficial_gas_vel: 表观气速 U_g = Q_g / A_cross [m/s]
            superficial_liquid_vel: 表观液速 [m/s]

        返回:
            alpha_g: 气含率
        """
        Ug = max(superficial_gas_vel, 1e-6)
        Ul = superficial_liquid_vel

        # Zuber-Findlay: α_g = Ug / [C0*(Ug+Ul) + V_drift]
        # 迭代求解 (因为 C0 本身依赖 α_g)
        alpha = 0.05
        for _ in range(20):
            C0_iter = 1.0 + 0.35 * (1 - alpha)
            Vd_iter = 0.25 * (1 - alpha) ** 0.5
            denom = C0_iter * (Ug + Ul) + Vd_iter
            alpha_new = Ug / denom if denom > 1e-10 else 0.05
            alpha_new = float(np.clip(alpha_new, 0.001, 0.40))
            if abs(alpha_new - alpha) < 1e-6:
                break
            alpha = alpha_new

        # 时间平滑
        self.alpha_g = float(np.clip(
            0.9 * self.alpha_g + 0.1 * alpha, 0.001, 0.40))
        self.C0 = 1.0 + 0.35 * (1 - self.alpha_g)
        self.v_drift = 0.25 * (1 - self.alpha_g) ** 0.5
        return self.alpha_g

    def get_shear_dissipation(self, epsilon_turb: float) -> float:
        """湍流耗散率与气含率的关系: ε ∝ g·Ug·α_g"""
        g = 9.81
        return epsilon_turb + g * self.alpha_g * self.v_drift


class TurbulenceKEpsilon:
    """
    k-ε 湍流模型 (OpenFOAM twoPhaseEulerFoam/buoyantBoussinesqPimpleFoam)

    湍动能方程:
    ∂k/∂t + ∇·(U k) = ∇·[(ν+ν_t/σ_k)∇k] + G_k - ε

    耗散率方程:
    ∂ε/∂t + ∇·(U ε) = ∇·[(ν+ν_t/σ_ε)∇ε] + (C1 G_k - C2 ε) ε/k

    近壁处理:
    膜表面速度梯度产生湍动能: G_k = ν_t (∂U/∂y)²
    壁面剪切: τ_w = ρ ν_t (∂U/∂y)
    """

    def __init__(self) -> None:
        self.k: float = 0.001       # 湍动能 m²/s²
        self.epsilon: float = 1e-4   # 耗散率 m²/s³
        self.nu_t: float = 1e-4     # 湍流黏性 m²/s
        self.G_k: float = 0.0       # 湍流生成项

    def update(self, superficial_gas_vel: float,
               alpha_g: float,
               pipe_gap: float,
               mlss: float,
               dt: float = 1.0) -> Tuple[float, float]:
        """
        更新 k-ε 模型

        参数:
            superficial_gas_vel: 表观气速 m/s
            alpha_g: 气含率
            pipe_gap: 曝气管到膜片距离 m
            mlss: 混合液悬浮固体 mg/L

        返回:
            (k, epsilon)
        """
        rho = PHYS.rho_l
        mu = PHYS.mu_l

        # 含气泡混合液等效黏度
        mu_eff = mu * (1.0 + 2.5 * alpha_g + 5.0 * alpha_g ** 2)
        # MLSS 贡献的额外黏度 (Casson 模型近似)
        mlss_factor = 1.0 + (mlss / 10000.0) ** 2.5
        mu_eff *= mlss_factor

        # 气泡引起的液相湍流增强
        # 气泡涌动产生的额外湍动能: k_bubble ≈ 0.5 * (U_g - U_l)² * α_g
        k_bubble = 0.5 * (superficial_gas_vel ** 2) * alpha_g if superficial_gas_vel > 0 else 0.0

        # 液相特征速度梯度 (曝气流股扩展)
        U_char = superficial_gas_vel / max(alpha_g, 0.01)
        dUdy = U_char / max(pipe_gap, 0.05)

        # 湍动能生成: G_k = ν_t * (∂U/∂y)²
        # ν_t = Cμ * k² / ε
        nu_t_est = min(PHYS.C_mu * self.k ** 2 / max(self.epsilon, 1e-10), mu_eff / rho * 10)
        self.G_k = nu_t_est * (dUdy ** 2)

        # k 方程
        Pk_G = min(self.G_k, 10.0)
        dk_dt = Pk_G - self.epsilon + 0.05 * k_bubble / dt
        self.k = max(self.k + dk_dt * dt, 1e-6)
        self.k = min(self.k, 0.5)

        # ε 方程
        C_mu_k2_eps = PHYS.C_mu * self.k ** 2
        de_dt = (PHYS.C1_epsilon * Pk_G - PHYS.C2_epsilon * self.epsilon) \
            * self.epsilon / max(self.k, 1e-6) * 1.0
        self.epsilon = max(self.epsilon + de_dt * dt, 1e-10)
        self.epsilon = min(self.epsilon, 5.0)

        # 更新湍流黏度
        self.nu_t = C_mu_k2_eps / max(self.epsilon, 1e-10)
        self.nu_t = min(self.nu_t, mu_eff / rho * 50)

        return self.k, self.epsilon

    def wall_shear_stress(self, y_wall: float = 1e-4) -> float:
        """
        近壁剪切应力 (线性壁面法则)
        τ_w = ρ ν_t ∂U/∂y ≈ ρ ν_t U_char / y_wall
        """
        rho = PHYS.rho_l
        U_char = sqrt(max(self.k, 1e-6))
        # 膜表面特征距离
        y = max(y_wall, 1e-5)
        tau = rho * self.nu_t * U_char / y
        return float(np.clip(tau, 0.0, 20.0))


class MembraneFoulingOF:
    """
    膜污染阻力模型 (Darcy 定律, OpenFOAM 内嵌)

    总阻力: R_total = R_m + R_c + R_p + R_g
    TMP = μ · J · R_total

    其中:
    R_m: 干净膜阻力 [1/m]
    R_c: 滤饼层阻力 (可变, 可压缩)
    R_p: 孔堵阻力 (不可逆)
    R_g: 凝胶层阻力 (浓差极化)
    """

    def __init__(self, Rm: float = PHYS.membrane_resistance) -> None:
        self.Rm: float = Rm       # 膜固有阻力 1/m
        self.Rc: float = 0.0      # 滤饼层阻力 1/m
        self.Rp: float = 0.0      # 孔堵阻力 1/m
        self.Rg: float = 0.0      # 凝胶层阻力 1/m
        self.R_total: float = Rm  # 总阻力 1/m
        self.TMP: float = 0.0     # kPa
        self.flux: float = 15.0   # LMH
        self.cake_porosity: float = 0.85  # 滤饼孔隙率
        self.alpha_cake: float = 5e10     # 比阻 m/kg

    def update(self, mlss: float, wall_shear_pa: float,
               do_level: float, dt: float,
               backwash: bool = False) -> float:
        """
        更新污染状态

        参数:
            mlss: MLSS mg/L
            wall_shear_pa: 近壁剪切应力 Pa
            do_level: 溶解氧 mg/L
            dt: 时间步 s

        返回:
            TMP kPa
        """
        # 比阻 (Kozeny-Carman 模型): α ∝ (1-e)²/e³
        # 滤饼压缩性: α 随 TMP 增大 (指数增长)
        compressibility = exp(0.05 * self.TMP) if self.TMP > PHYS.critical_tmp else 1.0
        alpha_eff = self.alpha_cake * compressibility / (self.cake_porosity ** 3)

        # 滤饼层积累率 (质量平衡)
        # d(m_cake)/dt = J · MLSS · 10^-3 - shear_removal
        J_m_s = self.flux / 3600.0 / 1000.0          # m/s
        deposition = J_m_s * mlss * 1e-3 * dt        # kg/m²/s
        # 剪切去除 (剪切应力越大, 去除越多)
        shear_removal_rate = wall_shear_pa / 2.0     # kg/m²/s (经验)
        net_deposition = max(0.0, deposition - shear_removal_rate * dt)

        if backwash:
            net_deposition = -self.Rc * 0.5  # 反洗去除 50% 滤饼
            self.Rp *= 0.98

        self.Rc += alpha_eff * net_deposition
        self.Rc = float(np.clip(self.Rc, 0.0, 1e14))

        # 孔堵 (内部堵塞, 缓慢积累)
        dRp_dt = 1e-11 * mlss * exp(-do_level / 2.0) * dt
        self.Rp += dRp_dt
        self.Rp = min(self.Rp, 1e12)

        # 凝胶层 (浓差极化)
        Rg_rate = 1e9 * (1.0 - do_level / 8.0) if do_level < 6.0 else -1e7 * dt
        self.Rg = max(0.0, self.Rg + Rg_rate * dt)
        self.Rg = min(self.Rg, 1e13)

        # 总阻力
        self.R_total = self.Rm + self.Rc + self.Rp + self.Rg

        # Darcy 定律求 TMP
        mu = PHYS.mu_l
        self.TMP = mu * J_m_s * self.R_total / 1000.0  # kPa
        self.TMP = min(self.TMP, PHYS.tmp_max)
        return self.TMP

    def get_state(self) -> Dict:
        return {"Rm": self.Rm, "Rc": self.Rc, "Rp": self.Rp,
                "Rg": self.Rg, "TMP": self.TMP, "flux": self.flux,
                "R_total": self.R_total}

    def set_state(self, state: Dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)


class MixedLiquorModel:
    """
    混合液模型: MLVSS、溶解氧 DO、MLR 黏度

    DO 质量平衡:
    dDO/dt = KLa(DS - DO) - OUR
    KLa ∝ U_g^0.8 / H^0.4 (Oxygen mass transfer)

    MLR (Mixed Liquor Rheology):
    μ = μ_w · (1 + α·MLVSS)^β  (幂律模型)
    """

    def __init__(self) -> None:
        self.do_level: float = 6.0       # mg/L
        self.mlvss: float = 6000.0      # mg/L
        self.mlr: float = 1.0           # 相对黏度

    def update(self, superficial_gas_vel: float,
               mlss: float, temperature: float,
               aeration_intensity: float,
               dt: float = 1.0) -> Tuple[float, float]:
        """
        更新 DO 和黏度

        参数:
            superficial_gas_vel: 表观气速 m/s
            mlss: MLSS mg/L
            temperature: 水温 °C
            aeration_intensity: 曝气强度 Nm³/m²/h
            dt: 时间步 s

        返回:
            (do_level, mlr)
        """
        # 温度修正 (纯水 DO_sat, Henry 定律简化)
        T_k = temperature + 273.15
        DO_sat = 14.6 - 0.4 * temperature + 0.01 * temperature ** 2  # mg/L

        # KLa 氧传质系数 (OpenFOAM interFoam 经验关联)
        H_eff = 1.0  # 有效液位 m (简化)
        KLa = 5.0 * (superficial_gas_vel ** 0.8) / (H_eff ** 0.4) if superficial_gas_vel > 0 else 0.1

        # OUR 污泥需氧率 (与 MLSS 和 SRT 相关)
        # SRT 越大 → 污泥越老 → 内源呼吸率越高
        base_our = 1.5 + 2.0 * exp(-mlss / 5000.0)
        our = base_our * (1.0 + 0.1 * temperature)  # mg/L/h

        # DO 方程: dDO/dt = KLa(DS-DO) - OUR
        dDO_dt = KLa * (DO_sat - self.do_level) - our / 24.0
        self.do_level = float(np.clip(self.do_level + dDO_dt * dt, 0.0, DO_sat))

        # MLVSS = f(MLSS)  (VSS/TSS ≈ 0.8)
        self.mlvss = mlss * 0.82

        # MLR 黏度 (幂律模型)
        # β ≈ 2.5 (高 MLSS 时牛顿→非牛顿转变)
        alpha_r = 1.5e-4
        beta_r = 2.5
        self.mlr = (1.0 + alpha_r * (mlss ** beta_r))
        self.mlr = min(self.mlr, PHYS.mu_max_factor)

        return self.do_level, self.mlr


# ==================== 核心仿真器 ====================
class OpenFOAMMBRSimulator:
    """OpenFOAM 物理模型增强型 MBR 仿真器"""

    def __init__(self) -> None:
        # 曝气参数
        self.intensity: float = 110.0     # Nm³/m²/h
        self.pulse_period: float = 3.0   # s
        self.p_pitch: int = 50            # mm
        self.s_pitch: int = 50            # mm
        self.h_size: float = 2.5          # mm (孔径)
        self.f_len: float = 2.0           # m
        self.slack: float = 0.025         # 无量纲
        self.mode: AerationMode = AerationMode.PULSE
        self.pipe_to_membrane_gap: int = 200  # mm

        # 膜参数
        self.fiber_diameter: float = 1.65  # mm
        self.thickness: int = 30           # mm
        self.flux: float = 15.0            # LMH

        # 污泥参数
        self.mlss: int = 10000             # mg/L
        self.srt: int = 12                 # d
        self.settling_rate: float = 3.0     # m/h
        self.return_ratio: int = 150        # %
        self.temperature: float = 20.0      # °C

        self.sludge_level: float = 0.15    # m
        self.is_discharging: bool = False
        self.sim_time: float = 0.0

        # OpenFOAM 物理模型
        self.pbm = BubblePopulationBalance()
        self.drift_flux = DriftFluxModel()
        self.turbulence = TurbulenceKEpsilon()
        self.fouling = MembraneFoulingOF()
        self.mixed_liquor = MixedLiquorModel()

        self.TMP_history: List[float] = []
        self.k_history: List[float] = []
        self.eps_history: List[float] = []
        self.alpha_g_history: List[float] = []
        self.d32_history: List[float] = []
        self.backwash_flag: bool = False

    def apply_preset(self, name: str) -> None:
        preset = PRESETS[name]
        for key in ("intensity", "p_pitch", "s_pitch", "h_size", "f_len",
                     "slack", "mode", "fiber_diameter", "thickness",
                     "mlss", "srt", "settling_rate", "return_ratio",
                     "pipe_to_membrane_gap", "temperature", "pulse_period"):
            setattr(self, key, getattr(preset, key))

    def get_sheet_area(self) -> float:
        base_area = 40.0 if self.fiber_diameter <= 1.8 else 25.0
        return round(base_area * (self.thickness / 30.0) * (self.f_len / 2.0), 2)

    def get_total_area(self) -> float:
        return self.get_sheet_area() * PHYS.sheet_count

    def calculate_fiber_count(self) -> Tuple[int, int]:
        sheet_area = self.get_sheet_area()
        diameter_m = self.fiber_diameter / 1000.0
        area_per_fiber = pi * diameter_m * self.f_len
        real_count = max(1, int(sheet_area / area_per_fiber))
        return real_count, min(real_count, 300)

    def get_superficial_gas_velocity(self) -> float:
        """表观气速 (OpenFOAM 核心变量) Ug = Qg / A_cross"""
        Qg_m3_s = (self.intensity * self.get_total_area()) / 3600.0
        A_cross = PHYS.sheet_width * PHYS.sheet_count * 0.01  # m²
        return Qg_m3_s / A_cross

    def calculate_shear(self) -> Tuple[float, float]:
        """完整 k-ε 剪切计算"""
        Ug = self.get_superficial_gas_velocity()
        alpha_g = self.drift_flux.update(Ug, dt=1.0)
        self.alpha_g_history.append(alpha_g)

        pipe_gap_m = self.pipe_to_membrane_gap / 1000.0
        k, eps = self.turbulence.update(Ug, alpha_g, pipe_gap_m, self.mlss, dt=1.0)
        self.k_history.append(k)
        self.eps_history.append(eps)

        # 近壁剪切应力
        wall_shear = self.turbulence.wall_shear_stress(y_wall=1e-4)
        slack_factor = 1.0 + self.slack * 12.0

        if self.mode == AerationMode.PULSE:
            max_shear = wall_shear * slack_factor * PHYS.pulse_power_boost * 1.5
        else:
            max_shear = wall_shear * slack_factor * 1.2

        avg = float(np.clip(wall_shear, 0.05, 5.0))
        max_s = float(np.clip(max_shear, 0.1, 10.0))
        return round(avg, 3), round(max_s, 3)

    def calculate_sec(self) -> float:
        """曝气比能耗 SEC"""
        delta_p = 50e3   # Pa (风机压升)
        q_air = self.intensity * self.get_total_area() / 3600.0  # m³/s
        power = delta_p * q_air / 0.7   # W (效率 0.7)
        self.power_w = power
        sec = power / (self.flux * self.get_total_area() / 24.0) / 1000.0
        return round(float(np.clip(sec, 0.05, 2.0)), 3)

    def calculate_uniformity(self) -> float:
        diff = abs(self.p_pitch - self.s_pitch)
        return round(max(0.0, 100.0 - diff * PHYS.uniform_penalty_factor), 1)

    def calculate_risk_level(self, max_shear: float) -> str:
        if max_shear < 0.8:
            return "HIGH"
        elif max_shear < 1.8:
            return "MEDIUM"
        return "LOW"

    def calculate_svi(self) -> float:
        svi = 200.0 - self.srt * 3.0 + (self.mlss / 10000.0) * 50.0
        return round(float(np.clip(svi, 50.0, 280.0)), 1)

    def calculate_tss(self, avg_shear: float) -> float:
        base = 5.0 + (self.sludge_level / PHYS.sludge_layer_max_height) * 30.0
        shear_effect = max(0.0, avg_shear * 0.5)
        tss = base - shear_effect
        return round(float(np.clip(tss, 2.0, 40.0)), 1)

    def update_sludge_level(self, dt: float = 1.0) -> None:
        if self.is_discharging:
            self.sludge_level = max(0.0, self.sludge_level - PHYS.sludge_discharge_rate * dt)
            return
        g = 9.81
        v0 = PHYS.vesilind_v0
        k = PHYS.vesilind_k
        v_settle = v0 * exp(-k * self.mlss / 1000.0) / 3600.0
        compress = (self.sludge_level / PHYS.sludge_layer_max_height) ** PHYS.compression_index
        mlss_norm = (self.mlss - 2000.0) / 13000.0
        return_factor = float(np.clip(self.return_ratio / 100.0, 0.5, 3.0))
        net = v_settle * (1.0 - compress) * mlss_norm * return_factor
        self.sludge_level += net * dt
        self.sludge_level = min(self.sludge_level, PHYS.sludge_layer_max_height)

    def update_bubble_pbm(self, dt: float = 1.0) -> float:
        """更新气泡群 PBM, 返回索太尔直径"""
        Ug = self.get_superficial_gas_velocity()
        eps = self.turbulence.epsilon if self.turbulence.epsilon > 0 else 1e-4
        self.pbm.update(eps, Ug, dt)
        d32 = self.pbm.get_sauter_diameter()
        self.d32_history.append(d32 * 1e3)  # mm
        return d32

    def step_simulation(self, dt_hours: float = 1.0) -> None:
        dt_sec = dt_hours * 3600.0
        dt_min = dt_hours * 60.0

        Ug = self.get_superficial_gas_velocity()

        # 1. 更新气泡群 (PBM)
        self.update_bubble_pbm(dt=dt_sec)

        # 2. 更新气含率 (Drift-Flux)
        self.drift_flux.update(Ug, dt=dt_min)

        # 3. 更新湍流 (k-ε)
        alpha_g = self.drift_flux.alpha_g
        pipe_gap_m = self.pipe_to_membrane_gap / 1000.0
        self.turbulence.update(Ug, alpha_g, pipe_gap_m, self.mlss, dt=dt_sec)

        # 4. 近壁剪切
        wall_shear = self.turbulence.wall_shear_stress()

        # 5. 更新 DO + 黏度
        self.mixed_liquor.update(Ug, self.mlss, self.temperature,
                                  self.intensity, dt=dt_min)

        # 6. 更新膜污染
        do = self.mixed_liquor.do_level
        tmp = self.fouling.update(self.mlss, wall_shear, do, dt=dt_sec,
                                   backwash=self.backwash_flag)
        self.TMP_history.append(tmp)
        self.backwash_flag = False
        if len(self.TMP_history) > 7200:
            self.TMP_history.pop(0)

        # 7. 更新污泥层
        self.update_sludge_level(dt_sec)
        self.sim_time += dt_sec

    def perform_backwash(self) -> None:
        self.backwash_flag = True
        wall_shear, _ = self.calculate_shear()
        self.fouling.update(self.mlss, wall_shear,
                            self.mixed_liquor.do_level, 0.1,
                            backwash=True)

    def discharge_sludge(self) -> None:
        self.sludge_level = max(0.0, self.sludge_level - 0.08)
        self.is_discharging = False

    def reset(self) -> None:
        self.__init__()

    def get_current_tmp(self) -> float:
        J_m_s = self.fouling.flux / 3600.0 / 1000.0
        tmp = PHYS.mu_l * J_m_s * self.fouling.R_total / 1000.0
        return round(min(tmp, PHYS.tmp_max), 2)

    def get_metrics(self) -> SimulationMetrics:
        avg_shear, max_shear = self.calculate_shear()
        _, display_count = self.calculate_fiber_count()
        real_count, _ = self.calculate_fiber_count()
        d32 = self.pbm.get_sauter_diameter() * 1e3
        state = self.fouling.get_state()
        mlr = self.mixed_liquor.mlr

        return SimulationMetrics(
            sec=self.calculate_sec(),
            power_w=round(self.power_w, 1),
            d_32=round(d32, 2),
            gas_velocity=round(self.get_superficial_gas_velocity(), 4),
            gas_holdup=round(self.drift_flux.alpha_g * 100, 2),
            shear_avg=avg_shear,
            shear_max=max_shear,
            k_turb=round(self.turbulence.k, 5),
            epsilon_turb=round(self.turbulence.epsilon, 5),
            uniformity=self.calculate_uniformity(),
            risk_text=self.calculate_risk_level(max_shear),
            mlvss=round(self.mixed_liquor.mlvss, 0),
            svi=self.calculate_svi(),
            tss=self.calculate_tss(avg_shear),
            mlr=round(mlr, 2),
            total_area=self.get_total_area(),
            fiber_count=real_count * PHYS.sheet_count,
            tmp=self.get_current_tmp(),
            Rm=round(state["Rm"], 1),
            Rc=round(state["Rc"], 1),
            Rp=round(state["Rp"], 1),
            do_level=round(self.mixed_liquor.do_level, 1),
            sludge_level_mm=self.sludge_level * 1000.0,
            sludge_percent=(self.sludge_level / PHYS.sludge_layer_max_height) * 100.0,
        )

    def get_state_snapshot(self) -> Dict:
        return {
            "intensity": self.intensity, "mode": self.mode.value,
            "pulse_period": self.pulse_period, "p_pitch": self.p_pitch,
            "s_pitch": self.s_pitch, "h_size": self.h_size,
            "slack": self.slack, "mlss": self.mlss,
            "settling_rate": self.settling_rate, "return_ratio": self.return_ratio,
            "sludge_level": self.sludge_level, "sim_time": self.sim_time,
            "fouling": self.fouling.get_state(), "is_discharging": self.is_discharging,
            "pipe_to_membrane_gap": self.pipe_to_membrane_gap,
            "temperature": self.temperature,
            "PBM_N": list(self.pbm.N),
            "alpha_g": self.drift_flux.alpha_g,
            "k_turb": self.turbulence.k,
            "epsilon_turb": self.turbulence.epsilon,
            "do_level": self.mixed_liquor.do_level,
            "mlr": self.mixed_liquor.mlr,
            "mlvss": self.mixed_liquor.mlvss,
        }

    def load_state_snapshot(self, snap: Dict) -> None:
        for key in ("intensity", "p_pitch", "s_pitch", "h_size", "slack",
                     "mlss", "settling_rate", "return_ratio",
                     "sludge_level", "sim_time", "is_discharging",
                     "pipe_to_membrane_gap", "temperature", "pulse_period"):
            if key in snap:
                setattr(self, key, snap[key])
        if "mode" in snap:
            self.mode = AerationMode(snap["mode"])
        if "fouling" in snap:
            self.fouling.set_state(snap["fouling"])
        if "PBM_N" in snap:
            self.pbm.N = np.array(snap["PBM_N"])
        if "alpha_g" in snap:
            self.drift_flux.alpha_g = snap["alpha_g"]
        if "k_turb" in snap:
            self.turbulence.k = snap["k_turb"]
        if "epsilon_turb" in snap:
            self.turbulence.epsilon = snap["epsilon_turb"]
        if "do_level" in snap:
            self.mixed_liquor.do_level = snap["do_level"]
        if "mlr" in snap:
            self.mixed_liquor.mlr = snap["mlr"]
        if "mlvss" in snap:
            self.mixed_liquor.mlvss = snap["mlvss"]

    def predict_trend(self, hours: float = 12.0, steps: int = 50) -> pd.DataFrame:
        state = self.get_state_snapshot()
        temp = OpenFOAMMBRSimulator()
        temp.load_state_snapshot(state)
        temp.is_discharging = False
        times = np.linspace(0, hours, steps)
        shear_v, sludge_v, tmp_v, do_v, d32_v, k_v, eps_v = [], [], [], [], [], [], []
        step_h = hours / (steps - 1)
        for _ in times:
            avg, _ = temp.calculate_shear()
            shear_v.append(avg)
            sludge_v.append(temp.sludge_level * 1000.0)
            tmp_v.append(temp.get_current_tmp())
            do_v.append(temp.mixed_liquor.do_level)
            d32_v.append(temp.pbm.get_sauter_diameter() * 1e3)
            k_v.append(temp.turbulence.k)
            eps_v.append(temp.turbulence.epsilon)
            temp.step_simulation(step_h)
        return pd.DataFrame({
            "time_h": times, "shear_pa": shear_v, "sludge_mm": sludge_v,
            "tmp_kpa": tmp_v, "do_mgl": do_v, "d32_mm": d32_v,
            "k_m2s2": k_v, "eps_m2s3": eps_v,
        })

    def generate_openfoam_case(self) -> Dict[str, str]:
        """
        生成 OpenFOAM 算例文件 (blockMesh + transportProperties)
        用于导出后在本地 OpenFOAM 中运行 CFD 仿真
        """
        pipe_gap_m = self.pipe_to_membrane_gap / 1000.0
        gap_half = pipe_gap_m / 2.0
        channel_H = self.f_len + 0.4
        channel_W = PHYS.sheet_width * PHYS.sheet_count + 0.2
        channel_D = 0.4

        # blockMeshDict
        x1, x2 = 0.0, channel_W
        y1, y2 = -gap_half - 0.05, channel_H
        z1, z2 = -channel_D / 2, channel_D / 2

        blockmesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  10                                    |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    ({x1} {y1} {z1})
    ({x2} {y1} {z1})
    ({x2} {y2} {z1})
    ({x1} {y2} {z1})
    ({x1} {y1} {z2})
    ({x2} {y1} {z2})
    ({x2} {y2} {z2})
    ({x1} {y2} {z2})
);

blocks
(
    hex (0 1 2 3 7 6 5 4) (40 60 10) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    walls
    {{
        type wall;
        faces
        (
            (0 3 7 4)
            (1 2 6 5)
        );
    }}
    membrane
    {{
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }}
    aerator
    {{
        type patch;
        faces
        (
            (0 4 5 1)
        );
    }}
    atmosphere
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
);

// ************************************************************************ //
"""

        # transportProperties (两相流体)
        trans_prop = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  10                                    |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

phases (water air);

water
{{
    transportModel  Newtonian;
    nu              {PHYS.mu_l:.1e};
    rho             {PHYS.rho_l:.1f};
}}

air
{{
    transportModel  Newtonian;
    nu              1.48e-5;
    rho             1.225;
}}

sigma             {PHYS.sigma:.3f};

// ************************************************************************ //
"""

        # turbulenceProperties (k-epsilon)
        turb_props = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType  RAS;
RAS
{{
    RASModel      kEpsilon;
    turbulence    on;
    printCoeffs   on;
}}
"""

        # fvSolution / A 字典 (曝气边界)
        Ug = self.get_superficial_gas_velocity()
        aeration_bc = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      alpha.air;
}}
dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0.05;
boundaryField
{{
    aerator
    {{
        type            fixedValue;
        value           uniform {self.drift_flux.alpha_g:.4f};
    }}
    membrane
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            zeroGradient;
    }}
    atmosphere
    {{
        type            inletOutlet;
        inletValue      uniform 0;
    }}
}}
"""

        return {
            "system/blockMeshDict": blockmesh,
            "constant/transportProperties": trans_prop,
            "constant/turbulenceProperties": turb_props,
            "0/alpha.air": aeration_bc,
        }


# ==================== 3D 可视化 ====================
def generate_3d_html(sim: OpenFOAMMBRSimulator) -> str:
    """生成 MBR 帘式膜 3D 可视化 HTML"""
    sheet_count = PHYS.sheet_count
    sheet_width = PHYS.sheet_width
    sheet_spacing = sim.s_pitch / 1000.0
    fiber_len = sim.f_len
    fd_m = sim.fiber_diameter / 1000.0
    sludge_h = sim.sludge_level
    sludge_mm = sim.sludge_level * 1000.0
    pipe_gap = sim.pipe_to_membrane_gap / 1000.0
    slack_a = sim.slack * 1.5
    module_depth = 0.08
    header_h = 0.05
    header_w = 0.04
    total_depth = sheet_count * module_depth + (sheet_count - 1) * sheet_spacing
    module_height = fiber_len + 0.3
    bottomY = -(pipe_gap + 0.3)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<style>
  body {{ margin: 0; overflow: hidden; background: #0d1117; font-family: sans-serif; }}
  #info {{ position: absolute; top: 10px; left: 20px; color: #c9d1d9; font-size: 12px; line-height: 1.6; z-index: 10; }}
  .legend {{ position: absolute; bottom: 20px; right: 20px; color: #8b949e; font-size: 11px; background: rgba(13,17,23,0.75); padding: 8px 12px; border-radius: 4px; z-index: 10; }}
  .legend span {{ display: inline-block; width: 11px; height: 11px; margin-right: 4px; border-radius: 2px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <b>MBR 帘式膜 3D (OpenFOAM 物理模型)</b><br>
  膜丝 {sim.fiber_diameter}mm×{fiber_len:.1f}m | {sheet_count}帘 | 帘距{sim.s_pitch}mm<br>
  曝气-膜片 {sim.pipe_to_membrane_gap}mm | MLSS {sim.mlss}mg/L<br>
  气含率 {sim.drift_flux.alpha_g*100:.1f}% | d_32 {sim.pbm.get_sauter_diameter()*1e3:.2f}mm<br>
  k={sim.turbulence.k:.4f}m²/s² | ε={sim.turbulence.epsilon:.4f}m²/s³ | DO={sim.mixed_liquor.do_level:.1f}mg/L
</div>
<div class="legend">
  <span style="background:#dddddd"></span>膜壳
  <span style="background:#4499ff"></span>膜丝
  <span style="background:#66aadd"></span>集水管
  <span style="background:#ff8844"></span>曝气管
  <span style="background:#886633;opacity:0.4"></span>污泥
</div>
<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const SW={sheet_width:.3f}, FL={fiber_len:.3f}, SS={sheet_spacing:.3f};
const SC={sheet_count}, TD={total_depth:.3f}, MH={module_height:.3f};
const MD={module_depth:.3f}, HH={header_h:.3f}, HW={header_w:.3f};
const SH={sludge_h:.3f}, PG={pipe_gap:.3f}, SLACK={slack_a:.3f};
const FD={fd_m:.5f};
const VFC=6, VFCZ=3, SEGS=4;
const bottomY={bottomY:.3f};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 8, 30);
const camera = new THREE.PerspectiveCamera(45, 1.6, 0.1, 50);
camera.position.set(SW*0.3, MH*0.5, TD+3.5);
camera.lookAt(SW*0.5, FL*0.3, TD*0.5);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = false;
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(SW*0.5, FL*0.3, TD*0.5);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.update();

scene.add(new THREE.AmbientLight(0x8090b0, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(TD+4, MH+3, TD+3);
scene.add(sun);

const grid = new THREE.GridHelper(Math.max(SW,TD)+2, 24, 0x334455, 0x1e2a38);
grid.position.set(SW*0.5, bottomY-0.01, TD*0.5);
scene.add(grid);

const caseMat = new THREE.MeshStandardMaterial({{color:0xdddddd, metalness:0.05, roughness:0.6}});
const fiberMat = new THREE.MeshStandardMaterial({{color:0x4499ff, metalness:0.02, roughness:0.6, transparent:true, opacity:0.80}});
const headerMat = new THREE.MeshStandardMaterial({{color:0x66aadd, metalness:0.4, roughness:0.3}});
const pipeMat = new THREE.MeshStandardMaterial({{color:0xff8844, metalness:0.4, roughness:0.4}});
const sludgeMat = new THREE.MeshStandardMaterial({{color:0x775522, metalness:0, roughness:1.0, transparent:true, opacity:0.45}});
const segGeo = new THREE.CylinderGeometry(FD*0.5, FD*0.5, FL/SEGS, 4);
const dummy = new THREE.Object3D();

for (let si=0; si<SC; si++) {{
  const cz = si*(MD+SS);
  const yTop = FL+0.15;
  // 膜壳
  const fg = new THREE.BoxGeometry(SW, FL, 0.006);
  for (let side=-1; side<=1; side+=2) {{
    const p = new THREE.Mesh(fg, caseMat);
    p.position.set(SW*0.5, FL*0.5, cz+side*MD*0.5);
    scene.add(p);
  }}
  const tg = new THREE.BoxGeometry(SW, 0.012, MD);
  const top = new THREE.Mesh(tg, caseMat);
  top.position.set(SW*0.5, yTop, cz);
  scene.add(top);
  // 集水管
  const hg = new THREE.CylinderGeometry(HH*0.5,HH*0.5, SW+HW*2, 14);
  const th = new THREE.Mesh(hg, headerMat);
  th.rotation.z=Math.PI/2; th.position.set(SW*0.5, yTop, cz);
  scene.add(th);
  const bg = new THREE.CylinderGeometry(HH*0.4,HH*0.4, SW+HW*0.5, 14);
  const bh = new THREE.Mesh(bg, headerMat);
  bh.rotation.z=Math.PI/2; bh.position.set(SW*0.5, 0.04, cz);
  scene.add(bh);
  // 侧连接管
  const sg = new THREE.CylinderGeometry(HH*0.35,HH*0.35, MD*0.5, 10);
  for (let side=-1; side<=1; side+=2) {{
    const s = new THREE.Mesh(sg, headerMat);
    s.rotation.x=Math.PI/2;
    s.position.set(side>0?SW*0.98:SW*0.02, yTop-HH*0.3, cz+side*MD*0.25);
    scene.add(s);
  }}
  // 膜丝 InstancedMesh
  const fcount = VFC*VFCZ*SEGS;
  const fm = new THREE.InstancedMesh(segGeo, fiberMat, fcount);
  let idx=0;
  for (let fi=0; fi<VFC; fi++) {{
    const fx=(fi+0.5)/VFC*(SW-HW*1.5)+HW*0.75;
    const bx=(Math.random()-0.5)*SLACK, bz=(Math.random()-0.5)*SLACK*0.4;
    for (let fzi=0; fzi<VFCZ; fzi++) {{
      const fz=cz-MD*0.45+(fzi+0.5)/VFCZ*MD*0.9;
      for (let s=0; s<SEGS; s++) {{
        const t=(s+0.5)/SEGS, curve=Math.sin(t*Math.PI);
        dummy.position.set(fx+bx*curve, s*(FL/SEGS)+FL/SEGS*0.5, fz+bz*curve);
        dummy.updateMatrix();
        fm.setMatrixAt(idx++, dummy.matrix);
      }}
    }}
  }}
  fm.instanceMatrix.needsUpdate=true;
  scene.add(fm);
}}

// 曝气管
const pipeR=0.018, pipeY=-PG;
const pGeo = new THREE.CylinderGeometry(pipeR,pipeR, SW*0.85, 10);
for (let si=0; si<SC; si++) {{
  const cz=si*(MD+SS);
  const mp=new THREE.Mesh(pGeo, pipeMat);
  mp.rotation.z=Math.PI/2; mp.position.set(SW*0.5, pipeY, cz);
  scene.add(mp);
  const brGeo=new THREE.CylinderGeometry(pipeR*0.5,pipeR*0.5, PG*0.7, 6);
  for (let pi=0; pi<4; pi++) {{
    const px=SW*0.18+pi*SW*0.18;
    const br=new THREE.Mesh(brGeo, pipeMat);
    br.position.set(px, pipeY+PG*0.35, cz);
    scene.add(br);
  }}
}}

// 污泥层
const sTopY=bottomY+SH;
const sgGeo=new THREE.BoxGeometry(SW*1.1, SH, TD*1.1);
const sludge=new THREE.Mesh(sgGeo, sludgeMat);
sludge.position.set(SW*0.5, (bottomY+sTopY)*0.5, TD*0.5);
scene.add(sludge);
const wGeo=new THREE.BoxGeometry(SW*1.1, 0.005, TD*1.1);
const wMat=new THREE.MeshBasicMaterial({{color:0x1a4a6e, transparent:true, opacity:0.3}});
const wl=new THREE.Mesh(wGeo, wMat);
wl.position.set(SW*0.5, sTopY, TD*0.5);
scene.add(wl);

// 气泡
const bGeo=new THREE.SphereGeometry(0.008,5,5);
const bMat=new THREE.MeshBasicMaterial({{color:0xaaddff, transparent:true, opacity:0.5}});
const bubbles=[];
for (let i=0; i<50; i++) {{
  const b=new THREE.Mesh(bGeo, bMat);
  const si=Math.floor(Math.random()*SC);
  b.position.set(SW*0.1+Math.random()*SW*0.8, pipeY+Math.random()*(FL*0.8), si*(MD+SS)+Math.random()*MD);
  b.userData={{speed:0.004+Math.random()*0.008, ox:b.position.x, oz:b.position.z, phase:Math.random()*Math.PI*2}};
  scene.add(b); bubbles.push(b);
}}

function animate(time) {{
  requestAnimationFrame(animate);
  const t=time*0.001;
  bubbles.forEach(b => {{
    b.position.y += b.userData.speed;
    b.position.x = b.userData.ox + Math.sin(t*2+b.userData.phase)*0.003;
    if (b.position.y > FL+0.15) {{
      b.position.y = pipeY;
      const si=Math.floor(Math.random()*SC);
      b.userData.ox=SW*0.1+Math.random()*SW*0.8;
      b.userData.oz=si*(MD+SS)+Math.random()*MD;
      b.position.x=b.userData.ox; b.position.z=b.userData.oz;
    }}
  }});
  controls.update();
  renderer.render(scene, camera);
}}
requestAnimationFrame(animate);
window.addEventListener('resize', () => {{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


# ==================== Streamlit UI ====================
def init_session() -> None:
    if "sim" not in st.session_state:
        st.session_state.sim = OpenFOAMMBRSimulator()
    if "preset_key" not in st.session_state:
        st.session_state.preset_key = "flush"


def render_sidebar(sim: OpenFOAMMBRSimulator) -> None:
    with st.sidebar:
        st.header("🔬 MBR OpenFOAM 仿真系统")

        st.subheader("📋 预设方案")
        cols = st.columns(len(PRESETS))
        for i, (key, preset) in enumerate(PRESETS.items()):
            with cols[i]:
                active = st.session_state.get("preset_key") == key
                if st.button(f"{preset.icon}", key=f"preset_{key}",
                             help=preset.label,
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    sim.apply_preset(key)
                    st.session_state.preset_key = key
                    st.rerun()

        st.divider()

        st.subheader("🌊 曝气参数 (OpenFOAM)")
        mode_opts = [m.value for m in AerationMode]
        idx = mode_opts.index(sim.mode.value)
        sim.mode = AerationMode(st.selectbox("曝气模式", mode_opts,
                              format_func=lambda x: AerationMode(x).label,
                              index=idx))
        sim.intensity = st.slider("曝气强度 (Nm³/m²/h)", 50, 150, int(sim.intensity), step=5)
        if sim.mode == AerationMode.PULSE:
            sim.pulse_period = st.slider("脉冲周期 (s)", 2.0, 8.0, float(sim.pulse_period), step=0.5)
        sim.h_size = st.slider("曝气孔径 (mm)", 1.0, 15.0, float(sim.h_size), step=0.5)
        sim.p_pitch = st.slider("曝气管间距 (mm)", 30, 300, int(sim.p_pitch), step=10)
        sim.pipe_to_membrane_gap = st.slider("曝气-膜片距离 (mm)", 100, 500, int(sim.pipe_to_membrane_gap), step=10)
        sim.temperature = st.slider("水温 (°C)", 5.0, 35.0, float(sim.temperature), step=1.0)

        st.divider()

        st.subheader("🧬 膜参数")
        sim.fiber_diameter = st.selectbox("膜丝外径", [1.65, 2.8],
                              format_func=lambda x: f"{x}mm → {'40' if x==1.65 else '25'}m²/片")
        sim.thickness = st.slider("膜厚 (mm)", 10, 100, int(sim.thickness), step=5)
        sim.s_pitch = st.slider("帘间距 (mm)", 30, 120, int(sim.s_pitch), step=5)
        sim.f_len = st.slider("膜丝长 (m)", 0.5, 3.0, float(sim.f_len), step=0.1)
        sim.slack = st.slider("松弛度 (%)", 0.2, 5.0,
                              value=float(round(sim.slack*100, 1)), step=0.2) / 100.0

        st.divider()

        st.subheader("🧫 污泥与操作")
        sim.mlss = st.slider("MLSS (mg/L)", 2000, 15000, int(sim.mlss), step=500)
        sim.srt = st.slider("SRT (d)", 5, 40, int(sim.srt), step=1)
        sim.return_ratio = st.slider("回流比 (%)", 50, 300, int(sim.return_ratio), step=10)

        op_cols = st.columns(4)
        for ci, (label, icon, action) in enumerate([
            ("排泥", "⬇️", lambda: sim.discharge_sludge()),
            ("反洗", "🧼", lambda: sim.perform_backwash()),
            ("+1h", "⏱", lambda: sim.step_simulation(1.0)),
            ("重置", "🔄", lambda: st.rerun(set(st.session_state, sim=OpenFOAMMBRSimulator()))),
        ]):
            with op_cols[ci]:
                if st.button(f"{icon}", key=f"op_{ci}", help=label, use_container_width=True):
                    action()
                    st.rerun()

        sludge_pct = sim.sludge_level / PHYS.sludge_layer_max_height * 100
        st.progress(min(100, int(sludge_pct)),
                    text=f"污泥层 {sim.sludge_level*1000:.0f}mm ({sludge_pct:.0f}%)")

        st.divider()
        st.subheader("⏩ 批量推进")
        bcols = st.columns(3)
        for ci, h in enumerate([2, 6, 12]):
            with bcols[ci]:
                if st.button(f"+{h}h", key=f"batch_{h}", use_container_width=True):
                    sim.step_simulation(float(h))
                    st.rerun()


def render_metrics(m: SimulationMetrics) -> None:
    rows = [
        [("SEC", f"{m.sec} kWh/m³", None), ("功率", f"{m.power_w:.0f} W", None),
         ("DO", f"{m.do_level:.1f} mg/L", None), ("MLR黏度", f"{m.mlr:.2f}x", None)],
        [("气含率", f"{m.gas_holdup:.2f} %", None), ("表观气速", f"{m.gas_velocity:.4f} m/s", None),
         ("d_32", f"{m.d_32:.2f} mm", None), ("均匀度", f"{m.uniformity:.1f} %", None)],
        [("平均剪切", f"{m.shear_avg:.3f} Pa", None), ("最大剪切", f"{m.shear_max:.3f} Pa",
         "delta" if m.risk_text=="HIGH" else None),
         ("k湍动能", f"{m.k_turb:.4f}", None), ("ε耗散率", f"{m.epsilon_turb:.4f}", None)],
        [("TMP", f"{m.tmp:.1f} kPa", ">35需反洗" if m.tmp>PHYS.critical_tmp else None),
         ("R_cake", f"{m.Rc:.1e}", None), ("R_pore", f"{m.Rp:.1e}", None),
         ("膜面积", f"{m.total_area:.0f} m²", None)],
        [("SVI", f"{m.svi:.0f} mL/g", None), ("TSS", f"{m.tss:.1f} mg/L", None),
         ("MLVSS", f"{m.mlvss:.0f} mg/L", None), ("积垢风险", m.risk_text,
         "inverse" if m.risk_text=="HIGH" else "normal")],
    ]
    for row in rows:
        cols = st.columns(len(row))
        for col, (label, value, delta) in zip(cols, row):
            with col:
                kw = {"label": label, "value": value}
                if delta:
                    if isinstance(delta, str):
                        kw["delta"] = delta
                    else:
                        kw["delta_color"] = delta
                st.metric(**kw)


def render_trend_charts(sim: OpenFOAMMBRSimulator) -> None:
    """OpenFOAM 多变量趋势图"""
    with st.expander("📈 12小时趋势 (OpenFOAM: 剪切·TMP·DO·d_32·k-ε)", expanded=False):
        df = sim.predict_trend(hours=12.0, steps=60)
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("剪切应力 τ_w (Pa)", "跨膜压力 TMP (kPa)",
                            "溶解氧 DO (mg/L) & 索太尔直径 d_32 (mm)",
                            "湍动能 k (m²/s²) & 耗散率 ε (m²/s³)"))
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["shear_pa"],
                                 name="τ_w", line=dict(color="#00f2ff"), mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["tmp_kpa"],
                                 name="TMP", line=dict(color="#ff6644"), mode="lines"), row=2, col=1)
        fig.add_hline(y=PHYS.critical_tmp, line_dash="dash", line_color="#ff6644",
                      annotation_text="反洗阈值", row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["do_mgl"],
                                 name="DO", line=dict(color="#44ff88"), mode="lines"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["d32_mm"],
                                 name="d_32", line=dict(color="#ffaa44"), yaxis="y2", mode="lines"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["k_m2s2"],
                                 name="k", line=dict(color="#aa44ff"), mode="lines"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["eps_m2s3"],
                                 name="ε", line=dict(color="#ff44aa"), yaxis="y2", mode="lines"), row=4, col=1)
        fig.update_layout(height=700, template="plotly_dark", hovermode="x unified")
        fig.update_xaxes(title_text="时间 (h)", row=4, col=1)
        fig.update_yaxes(title_text="d_32 (mm)", row=3, col=1, overlaying="y", side="right")
        fig.update_yaxes(title_text="ε (m²/s³)", row=4, col=1, overlaying="y", side="right")
        st.plotly_chart(fig, use_container_width=True)


def render_openfoam_export(sim: OpenFOAMMBRSimulator) -> None:
    """OpenFOAM 算例导出"""
    with st.expander("⚙️ OpenFOAM CFD 算例导出", expanded=False):
        st.info("生成 blockMeshDict + transportProperties + turbulenceProperties，可在本地 OpenFOAM 运行两相流 CFD 仿真")
        files = sim.generate_openfoam_case()
        st.json({k.split("/")[-1]: v[:100]+"..." for k, v in files.items()})
        if st.button("📦 下载 OpenFOAM 算例 (ZIP)"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for path, content in files.items():
                    zf.writestr(path, content)
            buf.seek(0)
            st.download_button("下载算例包",
                               data=buf.getvalue(),
                               file_name="MBR_OpenFOAM_case.zip",
                               mime="application/zip")


def render_sensitivity(sim: OpenFOAMMBRSimulator) -> None:
    with st.expander("🔬 参数敏感性分析", expanded=False):
        params = ["intensity", "pipe_to_membrane_gap", "mlss", "temperature", "s_pitch"]
        param_labels = {"intensity": "曝气强度", "pipe_to_membrane_gap": "曝气-膜片距离",
                         "mlss": "MLSS", "temperature": "水温", "s_pitch": "帘间距"}
        param = st.selectbox("分析参数", params, format_func=param_labels.get)
        base_val = getattr(sim, param)
        ranges = {
            "intensity": np.linspace(50, 150, 20),
            "pipe_to_membrane_gap": np.linspace(100, 500, 20),
            "mlss": np.linspace(2000, 15000, 20),
            "temperature": np.linspace(5, 35, 20),
            "s_pitch": np.linspace(30, 120, 20),
        }
        sv, tmp_v, do_v = [], [], []
        state = sim.get_state_snapshot()
        temp = OpenFOAMMBRSimulator()
        for v in ranges[param]:
            temp.load_state_snapshot(state)
            setattr(temp, param, type(base_val)(v))
            avg, _ = temp.calculate_shear()
            sv.append(avg)
            tmp_v.append(temp.get_current_tmp())
            do_v.append(temp.mixed_liquor.do_level)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("剪切应力 τ (Pa)", "TMP (kPa)", "DO (mg/L)"))
        fig.add_trace(go.Scatter(x=ranges[param], y=sv, mode="lines+markers",
                                 name="τ", line=dict(color="#00f2ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ranges[param], y=tmp_v, mode="lines+markers",
                                 name="TMP", line=dict(color="#ff6644")), row=2, col=1)
        fig.add_trace(go.Scatter(x=ranges[param], y=do_v, mode="lines+markers",
                                 name="DO", line=dict(color="#44ff88")), row=3, col=1)
        for ri in range(1, 4):
            fig.add_vline(x=base_val, line_dash="dash", line_color="#ffffff",
                          opacity=0.4, row=ri, col=1,
                          annotation_text=f"当前={base_val:.1f}")
        fig.update_layout(height=550, template="plotly_dark", hovermode="x unified")
        fig.update_xaxes(title_text=param_labels.get(param, param), row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)


def render_export(sim: OpenFOAMMBRSimulator) -> None:
    with st.expander("💾 导出与数据", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            snap = json.dumps(sim.get_state_snapshot(), indent=2, ensure_ascii=False)
            st.download_button("📄 导出配置 (JSON)", data=snap,
                              file_name="MBR_openfoam_config.json",
                              mime="application/json")
        with c2:
            df = sim.predict_trend(hours=12.0, steps=50)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("📊 导出趋势 (CSV)", data=buf.getvalue(),
                              file_name="MBR_openfoam_trend.csv",
                              mime="text/csv")
        up = st.file_uploader("📂 导入配置", type=["json"])
        if up:
            try:
                sim.load_state_snapshot(json.loads(up.read()))
                st.success("导入成功")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败: {e}")


# ==================== 主入口 ====================
def main() -> None:
    st.set_page_config(
        page_title="MBR OpenFOAM 仿真系统 v4.0",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session()
    sim: OpenFOAMMBRSimulator = st.session_state.sim

    render_sidebar(sim)

    st.title("💧 MBR OpenFOAM 工程级仿真系统 v4.0")
    st.caption(
        "OpenFOAM 物理模型: PBM气泡群 · Drift-Flux气含率 · k-ε湍流 · "
        "膜污染阻力 · DO氧传质 · MLR黏度 · OpenFOAM算例导出"
    )

    m = sim.get_metrics()
    render_metrics(m)

    st.markdown("### 🖥️ 3D 可视化视图")
    st.components.v1.html(generate_3d_html(sim), height=600, scrolling=False)

    render_trend_charts(sim)
    render_sensitivity(sim)
    render_openfoam_export(sim)
    render_export(sim)


if __name__ == "__main__":
    main()
