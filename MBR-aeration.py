"""
MBR 工程级仿真系统 v5.0 — 行业标准专业模型
=============================================
参考行业标准与学术文献:
  1. ASM1 (Henze et al, 1987 / IWA) — 生物动力学: COD去除、硝化、反硝化
  2. Critical Flux J_crit (Field et al, 1995) — 临界通量理论
  3. Sustainable Critical Flux (Xu et al, 2023) — 可持续临界通量 ≈ 0.67 J_crit
  4. Resistance-in-Series (Darcy, 标准膜阻力模型) — R_total = R_m + R_c + R_p + R_irr
  5. SADm / SADp (行业标准曝气指标) — 单位膜面积/产水曝气量
  6. K_Lim 阈值 (Jun & Daigger, 2024) — 沉积-剪切平衡, 最小 SADp 预测
  7. EPS/SMP 模型 (Laspidou & Rittmann, 2002) — 胞外聚合物/溶解性微生物产物
  8. DFCm (Delft Filtration Characterisation, 2014) — 污泥过滤性表征
  9. Slug Bubble 两阶段曝气 (Wang et al, 2018 AIChE J) — 聚并气泡强化剪切
  10. BioWin ASDM (EnviroSim, 50状态变量含P、pH、气体传质)
  11. TMP < 30 kPa 行业运行推荐 (amembrane, 2025)
  12. MLSS 8-12 g/L, HRT 4-8h, SRT 15-100d (MBR行业典型参数)
  13. 能耗 0.5-1.2 kWh/m³ 行业基准 (MBR vs CAS)
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from math import exp, log, pi, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ==================== 枚举定义 ====================
class AerationMode(Enum):
    CONTINUOUS = "cont"       # 连续曝气
    PULSE = "pulse"           # 脉冲曝气
    SLUG = "slug"             # 聚并气泡曝气 (Wang et al, 2018)

    @property
    def label(self) -> str:
        return {"cont": "连续曝气", "pulse": "脉冲式曝气",
                "slug": "聚并气泡 (slug)"}[self.value]


# ==================== 行业标准物理常数 ====================
@dataclass(frozen=True)
class PhysicsConstants:
    """统一物理常数 (行业标准参数)"""

    # ---- 膜架几何 ----
    sheet_count: int = 5
    sheet_width: float = 1.25
    pipe_offset: float = 0.45
    # Slug bubble 临界高度 (Wang et al, 2018: 250mm)
    slug_critical_height: float = 0.25

    # ---- 流体物性 ----
    rho_l: float = 998.0
    mu_l: float = 1.0e-3
    sigma: float = 0.072

    # ---- 气泡群 ----
    d_bubble_min: float = 1.0e-3
    d_bubble_max: float = 6.0e-3
    d_bubble_ref: float = 2.5e-3
    n_bins: int = 8
    breakup_C: float = 0.25
    coalescence_C: float = 0.10

    # ---- Drift-Flux ----
    C0: float = 1.0
    V_drift: float = 0.25

    # ---- 湍流 k-ε ----
    C_mu: float = 0.09
    C1_epsilon: float = 1.44
    C2_epsilon: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3

    # ---- 曝气剪切 ----
    uniform_penalty_factor: float = 0.4
    pulse_power_boost: float = 1.4
    slug_shear_boost: float = 6.0  # 6x 强化 (Wang et al, 2018)

    # ---- 膜污染 (Resistance-in-Series) ----
    membrane_resistance: float = 2.0e11        # R_m [1/m]
    fouling_rate_const: float = 1.0e-5
    backwash_efficiency: float = 0.9
    tmp_max: float = 60.0
    tmp_industry_recommended: float = 30.0      # 行业推荐运行 TMP < 30 kPa

    # ---- 沉降 ----
    sludge_layer_max_height: float = 0.6
    sludge_discharge_rate: float = 0.02
    vesilind_v0: float = 7.0
    vesilind_k: float = 0.6
    compression_index: float = 0.2

    # ---- ASM1 化学计量 ----
    Y_H: float = 0.67           # 异养菌产率系数 gCOD/gCOD
    Y_A: float = 0.24           # 自养菌产率系数 gCOD/gN
    i_XB: float = 0.086          # 生物量氮含量 gN/gCOD
    i_XP: float = 0.06           # 产物氮含量 gN/gCOD
    f_P: float = 0.08            # 内源呼吸惰性颗粒比例

    # ---- ASM1 动力学 (20°C) ----
    mu_H: float = 6.0            # 异养菌最大比增长速率 1/d
    K_S: float = 20.0             # 半饱和系数 gCOD/m³
    K_O_H: float = 0.2            # 氧半饱和系数 gO2/m³
    K_NO: float = 0.5             # 硝酸盐半饱和系数 gNO3-N/m³
    b_H: float = 0.62             # 异养菌衰减系数 1/d
    mu_A: float = 0.8             # 自养菌最大比增长速率 1/d
    K_NH: float = 1.0             # 氨氮半饱和系数 gNH4-N/m³
    K_O_A: float = 0.4            # 自养菌氧半饱和系数 gO2/m³
    b_A: float = 0.15             # 自养菌衰减系数 1/d
    k_a: float = 0.08             # 氨化速率 m³/(gCOD·d)
    k_h: float = 3.0              # 最大水解速率 1/d
    K_X: float = 0.1              # 水解半饱和系数 gCOD/gCOD
    eta_g: float = 0.8            # 缺氧校正因子 (水解)
    eta_h: float = 0.8            # 缺氧校正因子 (异养)

    # ---- Critical Flux (Field et al, 1995) ----
    J_crit_default: float = 30.0  # LMH 临界通量
    J_sustainable_ratio: float = 0.67  # (Xu et al, 2023)

    # ---- 行业基准 ----
    energy_benchmark_low: float = 0.5   # kWh/m³
    energy_benchmark_high: float = 1.2   # kWh/m³
    mlss_typical_low: float = 8000.0     # mg/L
    mlss_typical_high: float = 12000.0   # mg/L
    hrt_typical_low: float = 4.0         # h
    hrt_typical_high: float = 8.0        # h
    srt_typical_low: float = 15.0        # d
    srt_typical_high: float = 100.0      # d


PHYS = PhysicsConstants()


# ==================== 预设 ====================
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
    temperature: float
    hrt: float              # 水力停留时间 h
    flux: float             # 设计通量 LMH


PRESETS: Dict[str, Preset] = {
    "eco": Preset(
        name="eco", label="节能模式", icon="🌿",
        intensity=60, pulse_period=6.0, p_pitch=250, s_pitch=100,
        h_size=6.0, f_len=2.0, slack=0.008, mode=AerationMode.CONTINUOUS,
        fiber_diameter=2.8, thickness=30, mlss=6000, srt=20,
        settling_rate=2.0, return_ratio=80,
        pipe_to_membrane_gap=400, temperature=20.0, hrt=8.0, flux=12.0,
    ),
    "balanced": Preset(
        name="balanced", label="均衡模式", icon="⚖️",
        intensity=80, pulse_period=4.0, p_pitch=100, s_pitch=80,
        h_size=4.0, f_len=2.0, slack=0.015, mode=AerationMode.CONTINUOUS,
        fiber_diameter=1.65, thickness=30, mlss=8000, srt=15,
        settling_rate=2.5, return_ratio=100,
        pipe_to_membrane_gap=250, temperature=20.0, hrt=6.0, flux=15.0,
    ),
    "flush": Preset(
        name="flush", label="高冲刷模式", icon="💨",
        intensity=110, pulse_period=3.0, p_pitch=50, s_pitch=50,
        h_size=2.5, f_len=2.0, slack=0.025, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=30, mlss=10000, srt=12,
        settling_rate=3.0, return_ratio=150,
        pipe_to_membrane_gap=150, temperature=20.0, hrt=4.0, flux=18.0,
    ),
    "slug_opt": Preset(
        name="slug_opt", label="聚并气泡", icon="🫧",
        intensity=80, pulse_period=5.0, p_pitch=60, s_pitch=60,
        h_size=3.0, f_len=2.0, slack=0.02, mode=AerationMode.SLUG,
        fiber_diameter=1.65, thickness=30, mlss=8000, srt=20,
        settling_rate=2.5, return_ratio=100,
        pipe_to_membrane_gap=250, temperature=20.0, hrt=6.0, flux=16.0,
    ),
    "high_flux": Preset(
        name="high_flux", label="高通量", icon="⚡",
        intensity=130, pulse_period=2.5, p_pitch=40, s_pitch=40,
        h_size=2.0, f_len=2.5, slack=0.03, mode=AerationMode.PULSE,
        fiber_diameter=1.65, thickness=25, mlss=12000, srt=10,
        settling_rate=3.5, return_ratio=200,
        pipe_to_membrane_gap=150, temperature=20.0, hrt=4.0, flux=22.0,
    ),
}


# ==================== 数据类 ====================
@dataclass
class SimulationMetrics:
    """仿真指标汇总 (行业标准格式)"""
    # 能耗 & SAD
    sec: float = 0.0
    power_w: float = 0.0
    sadm: float = 0.0          # SADm [Nm³/m²/h]
    sadp: float = 0.0          # SADp [Nm³-air/m³-permeate]
    sadp_crit: float = 0.0     # 最小 SADp (K_Lim)
    # 气泡群
    d_32: float = 0.0
    gas_velocity: float = 0.0
    gas_holdup: float = 0.0
    # 剪切
    shear_avg: float = 0.0
    shear_max: float = 0.0
    k_turb: float = 0.0
    epsilon_turb: float = 0.0
    uniformity: float = 0.0
    risk_text: str = "N/A"
    # 污泥
    mlvss: float = 0.0
    svi: float = 0.0
    tss: float = 0.0
    mlr: float = 1.0
    # 膜
    total_area: float = 0.0
    fiber_count: int = 0
    tmp: float = 0.0
    Rm: float = 0.0
    Rc: float = 0.0
    Rp: float = 0.0
    R_irr: float = 0.0         # 不可逆阻力
    R_total: float = 0.0
    # 临界通量
    J_crit: float = 0.0         # 临界通量 LMH
    J_sustainable: float = 0.0  # 可持续临界通量 LMH
    J_actual: float = 0.0       # 实际通量 LMH
    # EPS/SMP
    eps: float = 0.0            # mg/gVSS
    smp: float = 0.0            # mg/L
    # ASM1 出水
    cod_eff: float = 0.0        # mg/L
    nh4_eff: float = 0.0        # mg/L
    no3_eff: float = 0.0        # mg/L
    tn_eff: float = 0.0         # mg/L
    # DO
    do_level: float = 0.0
    # DFCm (过滤性指数)
    dfcm_index: float = 0.0    # ΔR20 过滤性指数
    # 污泥层
    sludge_level_mm: float = 0.0
    sludge_percent: float = 0.0
    # K_Lim
    K_val: float = 0.0          # K 值 (Jun & Daigger, 2024)
    K_lim: float = 0.0          # 极限 K 值


# ==================== ASM1 生物动力学模型 ====================
class ASM1Model:
    """
    Activated Sludge Model No.1 (Henze et al, 1987 / IWA)
    行业标准生物动力学模型

    状态变量 (8):
      S_S  - 溶解性可生物降解 COD
      X_S  - 慢速降解颗粒 COD
      X_BH - 异养菌生物量
      X_BA - 自养菌生物量
      S_O  - 溶解氧
      S_NO - 硝酸盐氮
      S_NH - 氨氮
      S_ND - 溶解性可生物降解有机氮

    主要过程 (8):
      1. 异养菌好氧生长
      2. 异养菌缺氧生长 (反硝化)
      3. 自养菌好氧生长 (硝化)
      4. 异养菌衰减
      5. 自养菌衰减
      6. 溶解性有机氮氨化
      7. 慢速有机物水解
      8. 慢速有机氮水解
    """

    def __init__(self) -> None:
        # 状态变量 (初始值, 典型城市污水)
        self.S_S: float = 50.0      # gCOD/m³
        self.X_S: float = 200.0     # gCOD/m³
        self.X_BH: float = 2000.0   # gCOD/m³
        self.X_BA: float = 50.0     # gCOD/m³
        self.S_O: float = 6.0       # gO2/m³
        self.S_NO: float = 0.5      # gNO3-N/m³
        self.S_NH: float = 30.0     # gNH4-N/m³
        self.S_ND: float = 5.0      # gN/m³
        # 惰性
        self.X_P: float = 0.0       # 惰性颗粒产物 gCOD/m³
        self.X_I: float = 0.0       # 惰性颗粒有机物 gCOD/m³
        # 温度
        self.temperature: float = 20.0

    def _temp_correction(self, base: float, T: float, T_ref: float = 20.0,
                         theta: float = 1.07) -> float:
        """温度修正: k(T) = k(20) × θ^(T-20)"""
        return base * theta ** (T - T_ref)

    def _monod(self, S: float, K: float) -> float:
        """Monod 动力学项: S/(K+S)"""
        return S / (K + S) if (K + S) > 0 else 0.0

    def _switching(self, S_O: float, K_O: float) -> float:
        """溶解氧开关函数: S_O/(K_O+S_O)"""
        return S_O / (K_O + S_O) if (K_O + S_O) > 0 else 0.0

    def step(self, dt_hours: float, mlss: float,
             influent_COD: float = 400.0,
             influent_NH4: float = 40.0,
             influent_TKN: float = 50.0) -> Dict[str, float]:
        """
        推进 ASM1 一步 (简化 Euler 积分)

        返回:
            dict: 出水 COD, NH4, NO3, TN, 耗氧速率 OUR
        """
        T = self.temperature
        dt = dt_hours / 24.0  # 转换为天

        # 温度修正动力学参数
        mu_H = self._temp_correction(PHYS.mu_H, T, theta=1.07)
        b_H = self._temp_correction(PHYS.b_H, T, theta=1.07)
        mu_A = self._temp_correction(PHYS.mu_A, T, theta=1.10)
        b_A = self._temp_correction(PHYS.b_A, T, theta=1.08)
        k_h = self._temp_correction(PHYS.k_h, T, theta=1.08)
        k_a = self._temp_correction(PHYS.k_a, T, theta=1.08)

        # 进水稀释 (HRT 效应)
        HRT_d = dt_hours / 24.0
        dilution = HRT_d / (PHYS.hrt_typical_low / 24.0) * 0.05

        # ===== 水解 (慢速→溶解性) =====
        X_BH_ratio = self.X_BH / max(mlss, 1.0)
        K_X_eff = PHYS.K_X * max(1.0, self.X_BH / 1000.0)
        hydrolysis = k_h * (self.X_S / self.X_BH) / (K_X_eff + self.X_S / self.X_BH) * self.X_BH

        # 缺氧/好氧校正
        O2_switch = self._switching(self.S_O, PHYS.K_O_H)
        NO_switch = self._monod(self.S_NO, PHYS.K_NO)
        anoxic_switch = (1.0 - O2_switch) * NO_switch
        hydro_rate = hydrolysis * (O2_switch + PHYS.eta_g * anoxic_switch)

        self.X_S -= hydro_rate * dt
        self.S_S += hydro_rate * dt
        self.S_ND += hydro_rate * PHYS.i_XP * dt
        self.S_NH -= hydro_rate * PHYS.i_XP * dt * 0.5

        # ===== 异养菌好氧生长 (COD 去除) =====
        mu_H_rate = mu_H * self._monod(self.S_S, PHYS.K_S) \
            * self._switching(self.S_O, PHYS.K_O_H) * self.X_BH
        self.S_S -= mu_H_rate * dt / PHYS.Y_H
        self.S_O -= mu_H_rate * dt * (1.0 - PHYS.Y_H) / PHYS.Y_H
        self.X_BH += mu_H_rate * dt
        self.S_NH -= mu_H_rate * dt * PHYS.i_XB

        # ===== 异养菌缺氧生长 (反硝化) =====
        anoxic_rate = mu_H * PHYS.eta_h \
            * self._monod(self.S_S, PHYS.K_S) \
            * self._monod(self.S_NO, PHYS.K_NO) \
            * (1.0 - self._switching(self.S_O, PHYS.K_O_H)) * self.X_BH
        self.S_S -= anoxic_rate * dt / PHYS.Y_H
        self.S_NO -= anoxic_rate * dt * (1.0 - PHYS.Y_H) / (2.86 * PHYS.Y_H)
        self.X_BH += anoxic_rate * dt * 0.8
        self.S_NH -= anoxic_rate * dt * PHYS.i_XB

        # ===== 自养菌好氧生长 (硝化) =====
        mu_A_rate = mu_A * self._monod(self.S_NH, PHYS.K_NH) \
            * self._switching(self.S_O, PHYS.K_O_A) * self.X_BA
        self.S_NH -= mu_A_rate * dt * (1.0 / PHYS.Y_A + PHYS.i_XB)
        self.S_O -= mu_A_rate * dt * (4.57 - PHYS.Y_A) / PHYS.Y_A
        self.S_NO += mu_A_rate * dt / PHYS.Y_A
        self.X_BA += mu_A_rate * dt

        # ===== 衰减 =====
        decay_H = b_H * self.X_BH
        decay_A = b_A * self.X_BA
        self.X_BH -= decay_H * dt
        self.X_BA -= decay_A * dt
        self.X_S += (1.0 - PHYS.f_P) * (decay_H + decay_A) * dt
        self.X_P += PHYS.f_P * (decay_H + decay_A) * dt
        self.S_ND += (decay_H + decay_A) * PHYS.i_XB * dt * 0.5

        # ===== 氨化 =====
        ammonia_rate = k_a * self.S_ND * self.X_BH
        self.S_ND -= ammonia_rate * dt
        self.S_NH += ammonia_rate * dt

        # ===== 进水稀释 (简化为连续流) =====
        self.S_S += dilution * (influent_COD * 0.5 - self.S_S)
        self.S_NH += dilution * (influent_NH4 - self.S_NH)
        self.S_ND += dilution * (influent_TKN * 0.1 - self.S_ND)

        # 确保非负
        for attr in ("S_S", "X_S", "X_BH", "X_BA", "S_O", "S_NO", "S_NH", "S_ND"):
            setattr(self, attr, max(0.0, getattr(self, attr)))

        # 出水浓度 (假设完全混合, 膜出水中不含 SS)
        cod_eff = self.S_S + 0.05 * self.X_S  # 溶解性 COD + 少量颗粒
        nh4_eff = self.S_NH
        no3_eff = self.S_NO
        tn_eff = nh4_eff + no3_eff + self.S_ND * 0.2

        # 耗氧速率 OUR (gO2/m³/d)
        OUR = (self.S_O * 0.1 + mu_H_rate * (1.0 - PHYS.Y_H) / PHYS.Y_H
               + mu_A_rate * (4.57 - PHYS.Y_A) / PHYS.Y_A)

        return {
            "COD_eff": cod_eff, "NH4_eff": nh4_eff,
            "NO3_eff": no3_eff, "TN_eff": tn_eff,
            "OUR": OUR, "MLVSS": self.X_BH + self.X_BA + 0.1 * self.X_S,
        }


# ==================== EPS/SMP 模型 ====================
class EPS_SMP_Model:
    """
    EPS/SMP 模型 (Laspidou & Rittmann, 2002)
    统一理论: EPS=solid-phase, SMP=soluble-phase

    EPS 产生:
      - 生长相关 (UAP): 与底物利用速率成正比
      - 内源呼吸相关 (BAP): 与生物量衰减成正比

    EPS → SMP 水解:
      k_hyd * EPS   (酶促水解)

    膜污染贡献:
      EPS 增加滤饼层比阻 (α_cake ∝ EPS)
      SMP 增加孔堵阻力 (R_p ∝ SMP)
    """

    def __init__(self) -> None:
        self.eps: float = 50.0      # mg/gVSS (bound EPS)
        self.smp: float = 20.0      # mg/L (soluble SMP)
        self.uap: float = 0.0       # utilization-associated products
        self.bap: float = 0.0       # biomass-associated products

    def update(self, mlss: float, SRT: float, temperature: float,
               OUR: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        更新 EPS/SMP

        参数:
            mlss: 混合液悬浮固体 mg/L
            SRT: 污泥龄 d
            temperature: 水温 °C
            OUR: 耗氧速率 gO2/m³/d
            dt: 时间步 h

        返回:
            (EPS, SMP)
        """
        # 生长相关 UAP (与底物利用速率成正比)
        k_uap = 0.05 * (1.0 + 0.05 * (temperature - 20.0))
        uap_prod = k_uap * OUR * dt / 24.0

        # 内源 BAP (与衰减成正比)
        k_bap = 0.02 * (1.0 + 0.04 * (temperature - 20.0))
        bap_prod = k_bap * mlss / 1000.0 * dt / 24.0

        self.uap += uap_prod
        self.bap += bap_prod

        # EPS 积累 = UAP→EPS + BAP→EPS - EPS水解
        k_eps_hyd = 0.01 * (1.0 + 0.06 * (temperature - 20.0))
        eps_prod = (self.uap + self.bap) * 0.3
        eps_loss = k_eps_hyd * self.eps * dt / 24.0

        self.eps += (eps_prod - eps_loss) * dt / 24.0
        self.eps = float(np.clip(self.eps, 10.0, 200.0))

        # SMP = EPS水解产物
        smp_prod = eps_loss * 0.5
        # SMP 生物降解
        smp_degrad = 0.02 * self.smp * dt / 24.0
        self.smp += (smp_prod - smp_degrad) * dt / 24.0
        self.smp = float(np.clip(self.smp, 5.0, 100.0))

        return self.eps, self.smp


# ==================== Critical Flux & K_Lim 模型 ====================
class CriticalFluxModel:
    """
    Critical Flux (Field et al, 1995) + K_Lim (Jun & Daigger, 2024)

    J_crit: 临界通量, 低于此通量无污染
    J_sustainable: 可持续临界通量 ≈ 0.67 J_crit (Xu et al, 2023)

    K 值 (Jun & Daigger, 2024):
      K = (J × MLSS × μ_ML) / (τ_w × packing_density)
      当 K > K_Lim 时, 沉积 > 剪切, 进入不可逆污染区
    """

    def __init__(self) -> None:
        self.J_crit: float = PHYS.J_crit_default  # LMH
        self.J_sustainable: float = PHYS.J_crit_default * PHYS.J_sustainable_ratio
        self.K_lim: float = 0.05  # 极限 K 值 (经验)
        self.ml_viscosity: float = 1.0

    def update(self, mlss: float, mlr: float, eps: float,
               temperature: float, wall_shear: float,
               packing_density: float) -> Dict[str, float]:
        """
        更新临界通量和 K 值

        参数:
            mlss: MLSS mg/L
            mlr: 混合液相对黏度
            eps: EPS mg/gVSS
            temperature: 水温 °C
            wall_shear: 近壁剪切 Pa
            packing_density: 膜装填密度 m²/m³

        返回:
            dict: J_crit, J_sustainable, K_val
        """
        # 临界通量修正 (EPS 和温度影响)
        eps_factor = 1.0 + (eps - 50.0) / 100.0  # EPS 高 → J_crit 低
        temp_factor = 1.0 + 0.02 * (temperature - 20.0)  # 温度高 → J_crit 高
        self.J_crit = PHYS.J_crit_default * eps_factor * temp_factor * (1.0 / mlr)
        self.J_crit = float(np.clip(self.J_crit, 10.0, 60.0))

        self.J_sustainable = self.J_crit * PHYS.J_sustainable_ratio

        # K 值 (Jun & Daigger, 2024)
        # K = (J × MLSS × μ_ML) / (τ_w × packing_density)
        J_mh = self.J_crit / 1000.0  # m/h
        self.ml_viscosity = mlr * PHYS.mu_l * 1000.0  # 转换为 mPa·s
        K_val = (J_mh * mlss / 1000.0 * self.ml_viscosity) / \
                (max(wall_shear, 0.01) * max(packing_density, 10.0))
        K_val = float(np.clip(K_val, 0.001, 1.0))

        return {"J_crit": self.J_crit, "J_sustainable": self.J_sustainable, "K_val": K_val}


# ==================== DFCm 过滤性表征 ====================
class DFCmModel:
    """
    Delft Filtration Characterisation method (DFCm, 2014)
    标准污泥过滤性表征方法

    过滤性指数 ΔR20:
      在恒压过滤实验中, 20分钟内阻力增加量
      ΔR20 < 0.5e12 → 良好过滤性
      ΔR20 > 2.0e12 → 差过滤性, 需调整操作

    影响因素:
      EPS 浓度 (正相关)
      MLSS 浓度 (正相关)
      污泥龄 SRT (负相关, 长 SRT → 好过滤性)
      剪切 (负相关, 高剪切 → 好过滤性)
    """

    def __init__(self) -> None:
        self.delta_R20: float = 0.5e12  # 过滤性指数

    def update(self, eps: float, smp: float, mlss: float,
               srt: float, wall_shear: float) -> float:
        """
        更新 DFCm 过滤性指数

        返回:
            delta_R20 [1/m]
        """
        # 基础过滤性
        base = 0.3e12

        # EPS 贡献 (正相关, EPS 越高过滤性越差)
        eps_contrib = 0.02e12 * (eps / 50.0) ** 1.5

        # SMP 贡献 (溶解性物质直接堵塞孔隙)
        smp_contrib = 0.01e12 * (smp / 20.0) ** 2.0

        # MLSS 贡献
        mlss_contrib = 0.01e12 * (mlss / 8000.0)

        # SRT 修正 (长 SRT → 好过滤性)
        srt_factor = (15.0 / max(srt, 5.0)) ** 0.5

        # 剪切修正 (高剪切 → 好过滤性, 剥离 EPS)
        shear_factor = 1.0 / (1.0 + wall_shear * 0.5)

        self.delta_R20 = (base + eps_contrib + smp_contrib + mlss_contrib) \
            * srt_factor * shear_factor
        self.delta_R20 = float(np.clip(self.delta_R20, 0.1e12, 5.0e12))
        return self.delta_R20


# ==================== 膜污染阻力模型 (Resistance-in-Series) ====================
class MembraneFoulingRS:
    """
    Resistance-in-Series 模型 (行业标准)

    R_total = R_m + R_c + R_p + R_irr + R_g

    R_m   : 膜固有阻力 (constant)
    R_c   : 滤饼层阻力 (可逆, 物理清洗可去除)
    R_p   : 孔堵阻力 (部分可逆, 化学清洗可去除)
    R_irr : 不可逆污染阻力 (仅化学清洗可部分去除)
    R_g   : 凝胶层阻力 (浓差极化)
    """

    def __init__(self, Rm: float = PHYS.membrane_resistance) -> None:
        self.Rm: float = Rm
        self.Rc: float = 0.0
        self.Rp: float = 0.0
        self.R_irr: float = 0.0
        self.Rg: float = 0.0
        self.R_total: float = Rm
        self.TMP: float = 0.0
        self.flux: float = 15.0
        self.cake_porosity: float = 0.85
        self.alpha_cake: float = 5e10

    def update(self, mlss: float, wall_shear_pa: float,
               eps: float, smp: float, do_level: float,
               dt: float, backwash: bool = False,
               chem_clean: bool = False) -> float:
        """
        更新阻力模型

        返回:
            TMP kPa
        """
        J_m_s = self.flux / 3600.0 / 1000.0

        # EPS 修正比阻 (EPS 高 → 滤饼更致密)
        eps_factor = (eps / 50.0) ** 0.5
        alpha_eff = self.alpha_cake * eps_factor

        # 压缩性 (TMP > 30 kPa 时加速)
        compress = exp(0.03 * max(0.0, self.TMP - PHYS.tmp_industry_recommended))
        alpha_eff *= compress

        # 滤饼层积累 (质量平衡)
        deposition = J_m_s * mlss * 1e-3 * dt
        shear_removal = wall_shear_pa * 0.5 * dt
        net_deposition = max(0.0, deposition - shear_removal)

        if backwash:
            net_deposition = -self.Rc * 0.5
            self.Rp *= 0.95
        if chem_clean:
            self.R_irr *= 0.3
            self.Rp = 0.0

        self.Rc += alpha_eff * net_deposition
        self.Rc = float(np.clip(self.Rc, 0.0, 1e14))

        # 孔堵 (SMP 主导)
        dRp = 1e-11 * smp * exp(-do_level / 3.0) * dt
        self.Rp += dRp
        self.Rp = min(self.Rp, 1e12)

        # 不可逆污染 (缓慢积累)
        dR_irr = 1e-12 * mlss * dt / 3600.0
        self.R_irr += dR_irr
        self.R_irr = min(self.R_irr, 5e12)

        # 凝胶层
        Rg_rate = 1e9 * (1.0 - do_level / 8.0) * dt if do_level < 6.0 else -1e7 * dt
        self.Rg = max(0.0, self.Rg + Rg_rate)
        self.Rg = min(self.Rg, 1e13)

        self.R_total = self.Rm + self.Rc + self.Rp + self.R_irr + self.Rg
        self.TMP = PHYS.mu_l * J_m_s * self.R_total / 1000.0
        self.TMP = min(self.TMP, PHYS.tmp_max)
        return self.TMP

    def get_state(self) -> Dict:
        return {"Rm": self.Rm, "Rc": self.Rc, "Rp": self.Rp,
                "R_irr": self.R_irr, "Rg": self.Rg, "TMP": self.TMP,
                "flux": self.flux, "R_total": self.R_total}

    def set_state(self, state: Dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)


# ==================== 气泡群 PBM + Drift-Flux + k-ε ====================
class BubblePopulationBalance:
    """气泡群动力学 (PBM) — 离散分段法"""

    def __init__(self) -> None:
        self.d_min = PHYS.d_bubble_min
        self.d_max = PHYS.d_bubble_max
        self.n_bins = PHYS.n_bins
        self.bin_edges = np.logspace(log(self.d_min), log(self.d_max), self.n_bins + 1)
        self.bin_centers = np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])
        self.N = np.ones(self.n_bins) * 1e7
        self.epsilon = 0.1

    def update(self, epsilon_turb: float, gas_vel: float, dt: float) -> None:
        self.epsilon = epsilon_turb
        eps13 = epsilon_turb ** (1.0 / 3.0) if epsilon_turb > 1e-10 else 1e-4
        dNdt = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            di = self.bin_centers[i]
            breakup_rate = PHYS.breakup_C * (epsilon_turb / di) ** (1.0 / 3.0)
            coalescence_rate = PHYS.coalescence_C * di ** 2 * eps13
            for j in range(i + 1, self.n_bins):
                dj = self.bin_centers[j]
                parent = np.searchsorted(self.bin_edges, dj * 2 ** (-1/3))
                if parent < self.n_bins:
                    dNdt[i] += PHYS.breakup_C * (epsilon_turb / dj) ** (1/3) * self.N[j] * 0.5
            for j in range(self.n_bins):
                if i != j:
                    dk = (di ** 3 + self.bin_centers[j] ** 3) ** (1/3)
                    if dk <= self.d_max:
                        dNdt[i] -= coalescence_rate * self.N[j] * self.N[i] * 1e-9
            dNdt[i] -= breakup_rate * self.N[i]
            dNdt[i] -= gas_vel / 2.0 * self.N[i]
        self.N = np.clip(self.N + dNdt * dt, 1.0, 1e10)

    def get_sauter_diameter(self) -> float:
        n, d = self.N, self.bin_centers
        num, den = np.sum(n * d ** 3), np.sum(n * d ** 2)
        return num / den if den > 1e-20 else self.d_max

    def get_bubble_velocity(self, d: float) -> float:
        if d < 1e-3:
            return (998.0 - 1.2) * 9.81 * d ** 2 / 18.0 / 1.8e-5
        elif d < 2e-3:
            return min(0.23 * sqrt(9.81 * d), 0.25)
        return min(sqrt(2.14 * 1.2 * 9.81 * d + 0.505 * (998.0 - 1.2) * 9.81 * d), 0.40)


class DriftFluxModel:
    """Drift-Flux 双流体模型 (Zuber-Findlay)"""

    def __init__(self) -> None:
        self.alpha_g = 0.05
        self.v_drift = 0.25
        self.C0 = 1.0

    def update(self, Ug: float, Ul: float = 0.0, dt: float = 1.0) -> float:
        alpha = 0.05
        Ug = max(Ug, 1e-6)
        for _ in range(20):
            C0_iter = 1.0 + 0.35 * (1 - alpha)
            Vd_iter = 0.25 * (1 - alpha) ** 0.5
            denom = C0_iter * (Ug + Ul) + Vd_iter
            alpha_new = Ug / denom if denom > 1e-10 else 0.05
            alpha_new = float(np.clip(alpha_new, 0.001, 0.40))
            if abs(alpha_new - alpha) < 1e-6:
                break
            alpha = alpha_new
        self.alpha_g = float(np.clip(0.9 * self.alpha_g + 0.1 * alpha, 0.001, 0.40))
        self.C0 = 1.0 + 0.35 * (1 - self.alpha_g)
        self.v_drift = 0.25 * (1 - self.alpha_g) ** 0.5
        return self.alpha_g


class TurbulenceKEpsilon:
    """k-ε 湍流模型"""

    def __init__(self) -> None:
        self.k = 0.001
        self.epsilon = 1e-4
        self.nu_t = 1e-4
        self.G_k = 0.0

    def update(self, Ug: float, alpha_g: float, pipe_gap: float,
               mlss: float, mlr: float, dt: float = 1.0) -> Tuple[float, float]:
        rho, mu = PHYS.rho_l, PHYS.mu_l * mlr
        mu_eff = mu * (1.0 + 2.5 * alpha_g + 5.0 * alpha_g ** 2)
        k_bubble = 0.5 * (Ug ** 2) * alpha_g if Ug > 0 else 0.0
        U_char = Ug / max(alpha_g, 0.01)
        dUdy = U_char / max(pipe_gap, 0.05)
        nu_t_est = min(PHYS.C_mu * self.k ** 2 / max(self.epsilon, 1e-10), mu_eff / rho * 10)
        self.G_k = nu_t_est * (dUdy ** 2)
        Pk_G = min(self.G_k, 10.0)
        self.k = min(max(self.k + (Pk_G - self.epsilon + 0.05 * k_bubble / dt) * dt, 1e-6), 0.5)
        C_mu_k2 = PHYS.C_mu * self.k ** 2
        de_dt = (PHYS.C1_epsilon * Pk_G - PHYS.C2_epsilon * self.epsilon) * self.epsilon / max(self.k, 1e-6)
        self.epsilon = min(max(self.epsilon + de_dt * dt, 1e-10), 5.0)
        self.nu_t = min(C_mu_k2 / max(self.epsilon, 1e-10), mu_eff / rho * 50)
        return self.k, self.epsilon

    def wall_shear_stress(self, y_wall: float = 1e-4) -> float:
        U_char = sqrt(max(self.k, 1e-6))
        y = max(y_wall, 1e-5)
        tau = PHYS.rho_l * self.nu_t * U_char / y
        return float(np.clip(tau, 0.0, 20.0))


class MixedLiquorModel:
    """混合液: DO + MLVSS + MLR 黏度"""

    def __init__(self) -> None:
        self.do_level = 6.0
        self.mlvss = 6000.0
        self.mlr = 1.0

    def update(self, Ug: float, mlss: float, temperature: float,
               OUR: float, dt: float = 1.0) -> Tuple[float, float, float]:
        DO_sat = 14.6 - 0.4 * temperature + 0.01 * temperature ** 2
        KLa = 5.0 * (Ug ** 0.8) / (1.0 ** 0.4) if Ug > 0 else 0.1
        dDO = KLa * (DO_sat - self.do_level) - OUR / 24.0
        self.do_level = float(np.clip(self.do_level + dDO * dt, 0.0, DO_sat))
        self.mlvss = mlss * 0.82
        alpha_r, beta_r = 1.5e-4, 2.5
        self.mlr = min((1.0 + alpha_r * (mlss ** beta_r)), 8.0)
        return self.do_level, self.mlvss, self.mlr


# ==================== Slug Bubble 曝气模型 ====================
class SlugBubbleModel:
    """
    Slug Bubble 两阶段曝气 (Wang et al, 2018 AIChE J)

    阶段1: 单气泡聚并 (coalescence) → 形成大尺寸 slug
    阶段2: 大 slug 破裂 (split) → 进入膜片间通道

    关键参数:
      - 临界高度 h_crit = 250 mm (聚并必需)
      - 剪切增强: 6× 常规气泡 (slug vs single bubble)
      - 能耗降低: ~50% (vs 行业平均 SADm)

    设计条件:
      pipe_gap > h_crit 时 slug 气泡可充分发展
    """

    def __init__(self) -> None:
        self.slug_formed: bool = False
        self.slug_length: float = 0.0      # 聚并气泡长度 m
        self.shear_boost: float = 1.0       # 剪切增强倍数

    def update(self, pipe_gap: float, intensity: float,
               h_size: float, sheet_spacing: float) -> Dict[str, float]:
        """
        评估 slug 气泡形成条件

        返回:
            dict: slug_formed, slug_length, shear_boost, energy_reduction
        """
        h_crit = PHYS.slug_critical_height

        # 条件1: 曝气-膜片距离 > 临界高度 (250 mm)
        gap_condition = pipe_gap >= h_crit

        # 条件2: 膜片间距足够 (通道宽度 > 5 mm)
        spacing_condition = sheet_spacing >= 5.0

        # 条件3: 足够的气量 (> 50 Nm³/m²/h)
        flow_condition = intensity >= 50.0

        self.slug_formed = gap_condition and spacing_condition and flow_condition

        if self.slug_formed:
            # slug 长度 ≈ 0.5 × 通道宽度 (经验)
            self.slug_length = min(0.5 * (sheet_spacing / 1000.0), 0.3)
            self.shear_boost = PHYS.slug_shear_boost  # 6×
            energy_reduction = 0.50  # 50% 能耗降低
        else:
            self.slug_length = 0.0
            self.shear_boost = 1.0
            energy_reduction = 0.0

        return {
            "slug_formed": self.slug_formed,
            "slug_length": self.slug_length,
            "shear_boost": self.shear_boost,
            "energy_reduction": energy_reduction,
        }


# ==================== 核心仿真器 ====================
class MBRSimulator:
    """行业标准专业模型 MBR 仿真器"""

    def __init__(self) -> None:
        # 曝气
        self.intensity: float = 110.0
        self.pulse_period: float = 3.0
        self.p_pitch: int = 50
        self.s_pitch: int = 50
        self.h_size: float = 2.5
        self.f_len: float = 2.0
        self.slack: float = 0.025
        self.mode: AerationMode = AerationMode.PULSE
        self.pipe_to_membrane_gap: int = 200
        self.temperature: float = 20.0

        # 膜
        self.fiber_diameter: float = 1.65
        self.thickness: int = 30
        self.flux: float = 15.0

        # 污泥
        self.mlss: int = 10000
        self.srt: int = 12
        self.settling_rate: float = 3.0
        self.return_ratio: int = 150
        self.hrt: float = 6.0
        self.sludge_level: float = 0.15
        self.is_discharging: bool = False
        self.sim_time: float = 0.0

        # 物理模型
        self.pbm = BubblePopulationBalance()
        self.drift_flux = DriftFluxModel()
        self.turbulence = TurbulenceKEpsilon()
        self.fouling = MembraneFoulingRS()
        self.mixed_liquor = MixedLiquorModel()
        self.asm1 = ASM1Model()
        self.eps_smp = EPS_SMP_Model()
        self.critical_flux = CriticalFluxModel()
        self.dfcm = DFCmModel()
        self.slug = SlugBubbleModel()

        self.TMP_history: List[float] = []
        self.backwash_flag: bool = False

    def apply_preset(self, name: str) -> None:
        p = PRESETS[name]
        for key in ("intensity", "p_pitch", "s_pitch", "h_size", "f_len",
                     "slack", "mode", "fiber_diameter", "thickness",
                     "mlss", "srt", "settling_rate", "return_ratio",
                     "pipe_to_membrane_gap", "temperature", "pulse_period",
                     "hrt", "flux"):
            setattr(self, key, getattr(p, key))

    def get_sheet_area(self) -> float:
        base = 40.0 if self.fiber_diameter <= 1.8 else 25.0
        return round(base * (self.thickness / 30.0) * (self.f_len / 2.0), 2)

    def get_total_area(self) -> float:
        return self.get_sheet_area() * PHYS.sheet_count

    def calculate_fiber_count(self) -> Tuple[int, int]:
        sa = self.get_sheet_area()
        dm = self.fiber_diameter / 1000.0
        af = pi * dm * self.f_len
        rc = max(1, int(sa / af))
        return rc, min(rc, 300)

    def get_superficial_gas_velocity(self) -> float:
        Qg = (self.intensity * self.get_total_area()) / 3600.0
        A_cross = PHYS.sheet_width * PHYS.sheet_count * 0.01
        return Qg / A_cross

    def calculate_shear(self) -> Tuple[float, float, Dict]:
        """完整剪切计算 (含 slug 增强)"""
        Ug = self.get_superficial_gas_velocity()
        alpha_g = self.drift_flux.update(Ug, dt=1.0)
        pipe_gap_m = self.pipe_to_membrane_gap / 1000.0
        k, eps = self.turbulence.update(Ug, alpha_g, pipe_gap_m,
                                         self.mlss, self.mixed_liquor.mlr, dt=1.0)
        wall_shear = self.turbulence.wall_shear_stress()

        # Slug 增强
        slug_info = self.slug.update(pipe_gap_m, self.intensity,
                                      self.h_size, self.s_pitch)
        if slug_info["slug_formed"] and self.mode == AerationMode.SLUG:
            wall_shear *= slug_info["shear_boost"]

        sf = 1.0 + self.slack * 12.0
        if self.mode == AerationMode.PULSE:
            max_s = wall_shear * sf * PHYS.pulse_power_boost * 1.5
        elif self.mode == AerationMode.SLUG:
            max_s = wall_shear * sf * slug_info["shear_boost"] * 1.3
        else:
            max_s = wall_shear * sf * 1.2

        avg = float(np.clip(wall_shear, 0.05, 5.0))
        max_s = float(np.clip(max_s, 0.1, 12.0))
        return round(avg, 3), round(max_s, 3), slug_info

    def calculate_sad(self) -> Tuple[float, float]:
        """SADm 和 SADp (行业标准)"""
        sadm = self.intensity  # Nm³/m²/h
        sadp = self.intensity * self.get_total_area() / \
               (self.flux * self.get_total_area() / 24.0 / 1000.0)
        sadp = round(sadp, 1)
        return sadm, sadp

    def calculate_sec(self) -> float:
        dp = 50e3
        q_air = self.intensity * self.get_total_area() / 3600.0
        power = dp * q_air / 0.7
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
        return round(float(np.clip(base - avg_shear * 0.5, 2.0, 40.0)), 1)

    def update_sludge_level(self, dt: float = 1.0) -> None:
        if self.is_discharging:
            self.sludge_level = max(0.0, self.sludge_level - PHYS.sludge_discharge_rate * dt)
            return
        v_settle = PHYS.vesilind_v0 * exp(-PHYS.vesilind_k * self.mlss / 1000.0) / 3600.0
        compress = (self.sludge_level / PHYS.sludge_layer_max_height) ** PHYS.compression_index
        mlss_n = float(np.clip((self.mlss - 2000.0) / 13000.0, 0.0, 1.0))
        rf = float(np.clip(self.return_ratio / 100.0, 0.5, 3.0))
        self.sludge_level += v_settle * (1.0 - compress) * mlss_n * rf * dt
        self.sludge_level = min(self.sludge_level, PHYS.sludge_layer_max_height)

    def step_simulation(self, dt_hours: float = 1.0) -> None:
        dt_sec = dt_hours * 3600.0
        dt_min = dt_hours * 60.0

        Ug = self.get_superficial_gas_velocity()

        # 1. ASM1 生物动力学
        asm_out = self.asm1.step(dt_hours, self.mlss)

        # 2. EPS/SMP
        eps, smp = self.eps_smp.update(
            self.mlss, self.srt, self.temperature, asm_out["OUR"], dt_hours)

        # 3. DO + MLR
        do, mlvss, mlr = self.mixed_liquor.update(
            Ug, self.mlss, self.temperature, asm_out["OUR"], dt_min)

        # 4. 气泡群 PBM
        self.pbm.update(self.turbulence.epsilon, Ug, dt_sec)

        # 5. 气含率
        self.drift_flux.update(Ug, dt=dt_min)

        # 6. 湍流 + 剪切
        pipe_gap_m = self.pipe_to_membrane_gap / 1000.0
        self.turbulence.update(Ug, self.drift_flux.alpha_g,
                               pipe_gap_m, self.mlss, mlr, dt_sec)
        wall_shear = self.turbulence.wall_shear_stress()

        # Slug 增强
        slug_info = self.slug.update(pipe_gap_m, self.intensity,
                                      self.h_size, self.s_pitch)
        if slug_info["slug_formed"] and self.mode == AerationMode.SLUG:
            wall_shear *= slug_info["shear_boost"]

        # 7. 膜污染
        tmp = self.fouling.update(self.mlss, wall_shear, eps, smp, do,
                                   dt_sec, backwash=self.backwash_flag)
        self.TMP_history.append(tmp)
        self.backwash_flag = False
        if len(self.TMP_history) > 7200:
            self.TMP_history.pop(0)

        # 8. Critical Flux / K_Lim
        pd = self.get_total_area() / (PHYS.sheet_width * PHYS.sheet_count * 0.01 * pipe_gap_m)
        self.critical_flux.update(self.mlss, mlr, eps, self.temperature,
                                  wall_shear, pd)

        # 9. DFCm
        self.dfcm.update(eps, smp, self.mlss, self.srt, wall_shear)

        # 10. 污泥层
        self.update_sludge_level(dt_sec)
        self.sim_time += dt_sec

    def perform_backwash(self) -> None:
        self.backwash_flag = True
        _, _, _ = self.calculate_shear()
        self.fouling.update(self.mlss, 0.5, 0.0, 0.0,
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
        avg_s, max_s, slug_info = self.calculate_shear()
        sadm, sadp = self.calculate_sad()
        rc, _ = self.calculate_fiber_count()
        d32 = self.pbm.get_sauter_diameter() * 1e3
        fs = self.fouling.get_state()
        cf = self.critical_flux
        asm_out = self.asm1.step(0.01, self.mlss)

        # K_Lim 最小 SADp
        sadp_crit = sadp * (cf.K_lim / max(cf.K_lim, 0.001)) if cf.K_lim > 0 else 0.0

        return SimulationMetrics(
            sec=self.calculate_sec(),
            power_w=round(self.power_w, 1),
            sadm=sadm, sadp=sadp, sadp_crit=round(sadp_crit, 1),
            d_32=round(d32, 2),
            gas_velocity=round(self.get_superficial_gas_velocity(), 4),
            gas_holdup=round(self.drift_flux.alpha_g * 100, 2),
            shear_avg=avg_s, shear_max=max_s,
            k_turb=round(self.turbulence.k, 5),
            epsilon_turb=round(self.turbulence.epsilon, 5),
            uniformity=self.calculate_uniformity(),
            risk_text=self.calculate_risk_level(max_s),
            mlvss=round(self.mixed_liquor.mlvss, 0),
            svi=self.calculate_svi(),
            tss=self.calculate_tss(avg_s),
            mlr=round(self.mixed_liquor.mlr, 2),
            total_area=self.get_total_area(),
            fiber_count=rc * PHYS.sheet_count,
            tmp=self.get_current_tmp(),
            Rm=round(fs["Rm"], 1), Rc=round(fs["Rc"], 1),
            Rp=round(fs["Rp"], 1), R_irr=round(fs["R_irr"], 1),
            R_total=round(fs["R_total"], 1),
            J_crit=round(cf.J_crit, 1),
            J_sustainable=round(cf.J_sustainable, 1),
            J_actual=self.flux,
            eps=round(self.eps_smp.eps, 1),
            smp=round(self.eps_smp.smp, 1),
            cod_eff=round(asm_out["COD_eff"], 1),
            nh4_eff=round(asm_out["NH4_eff"], 2),
            no3_eff=round(asm_out["NO3_eff"], 2),
            tn_eff=round(asm_out["TN_eff"], 2),
            do_level=round(self.mixed_liquor.do_level, 1),
            dfcm_index=round(self.dfcm.delta_R20 / 1e12, 2),
            sludge_level_mm=self.sludge_level * 1000.0,
            sludge_percent=(self.sludge_level / PHYS.sludge_layer_max_height) * 100.0,
            K_val=round(getattr(cf, 'K_val', 0.0), 4),
            K_lim=round(cf.K_lim, 4),
        )

    def get_state_snapshot(self) -> Dict:
        return {
            "intensity": self.intensity, "mode": self.mode.value,
            "p_pitch": self.p_pitch, "s_pitch": self.s_pitch,
            "h_size": self.h_size, "slack": self.slack,
            "mlss": self.mlss, "srt": self.srt,
            "settling_rate": self.settling_rate,
            "return_ratio": self.return_ratio,
            "sludge_level": self.sludge_level,
            "sim_time": self.sim_time,
            "fouling": self.fouling.get_state(),
            "pipe_to_membrane_gap": self.pipe_to_membrane_gap,
            "temperature": self.temperature, "hrt": self.hrt, "flux": self.flux,
            "PBM_N": list(self.pbm.N),
            "alpha_g": self.drift_flux.alpha_g,
            "k_turb": self.turbulence.k,
            "epsilon_turb": self.turbulence.epsilon,
            "do_level": self.mixed_liquor.do_level,
            "mlr": self.mixed_liquor.mlr,
            "mlvss": self.mixed_liquor.mlvss,
            "eps": self.eps_smp.eps, "smp": self.eps_smp.smp,
            "asm_S_S": self.asm1.S_S, "asm_S_NH": self.asm1.S_NH,
            "asm_S_NO": self.asm1.S_NO, "asm_X_BH": self.asm1.X_BH,
        }

    def load_state_snapshot(self, snap: Dict) -> None:
        for key in ("intensity", "p_pitch", "s_pitch", "h_size", "slack",
                     "mlss", "srt", "settling_rate", "return_ratio",
                     "sludge_level", "sim_time", "pipe_to_membrane_gap",
                     "temperature", "hrt", "flux", "pulse_period"):
            if key in snap:
                setattr(self, key, snap[key])
        if "mode" in snap:
            self.mode = AerationMode(snap["mode"])
        if "fouling" in snap:
            self.fouling.set_state(snap["fouling"])
        for attr, key in [("pbm", "PBM_N"), ("drift_flux", "alpha_g"),
                          ("turbulence", "k_turb"), ("turbulence", "epsilon_turb"),
                          ("mixed_liquor", "do_level"), ("mixed_liquor", "mlr"),
                          ("mixed_liquor", "mlvss"), ("eps_smp", "eps"),
                          ("eps_smp", "smp"), ("asm1", "asm_S_S"),
                          ("asm1", "asm_S_NH"), ("asm1", "asm_S_NO"),
                          ("asm1", "asm_X_BH")]:
            if key in snap and hasattr(self, attr):
                obj = getattr(self, attr)
                if key == "PBM_N":
                    obj.N = np.array(snap[key])
                elif hasattr(obj, key.split("_")[-1].lower()):
                    setattr(obj, key.split("_")[-1].lower(), snap[key])
                elif key in ("alpha_g", "k_turb", "epsilon_turb", "do_level", "mlr", "mlvss"):
                    setattr(obj, key.split("_")[-1] if "_" in key else key, snap[key])
                elif key in ("eps", "smp"):
                    setattr(obj, key, snap[key])
                elif key.startswith("asm_"):
                    setattr(obj, key[4:], snap[key])

    def predict_trend(self, hours: float = 12.0, steps: int = 50) -> pd.DataFrame:
        state = self.get_state_snapshot()
        temp = MBRSimulator()
        temp.load_state_snapshot(state)
        temp.is_discharging = False
        times = np.linspace(0, hours, steps)
        records = []
        step_h = hours / (steps - 1)
        for t in times:
            avg, _, _ = temp.calculate_shear()
            m = temp.get_metrics()
            records.append({
                "time_h": t, "shear_pa": avg,
                "sludge_mm": temp.sludge_level * 1000.0,
                "tmp_kpa": m.tmp, "do_mgl": m.do_level,
                "d32_mm": m.d_32, "k_m2s2": m.k_turb,
                "eps_m2s3": m.epsilon_turb,
                "eps_mgg": m.eps, "smp_mgl": m.smp,
                "cod_mgl": m.cod_eff, "nh4_mgl": m.nh4_eff,
                "no3_mgl": m.no3_eff, "tn_mgl": m.tn_eff,
                "J_crit": m.J_crit, "J_sus": m.J_sustainable,
                "K_val": m.K_val, "dfcm": m.dfcm_index,
            })
            temp.step_simulation(step_h)
        return pd.DataFrame(records)


# ==================== 3D 可视化 ====================
def generate_3d_html(sim: MBRSimulator) -> str:
    sw = PHYS.sheet_width
    ss = sim.s_pitch / 1000.0
    fl = sim.f_len
    fd = sim.fiber_diameter / 1000.0
    sh = sim.sludge_level
    pg = sim.pipe_to_membrane_gap / 1000.0
    sl = sim.slack * 1.5
    md = 0.08; hh = 0.05; hw = 0.04
    td = PHYS.sheet_count * md + (PHYS.sheet_count - 1) * ss
    mh = fl + 0.3; by = -(pg + 0.3)
    d32 = sim.pbm.get_sauter_diameter() * 1e3
    slug_ok = sim.slug.slug_formed

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<style>
  body {{ margin: 0; overflow: hidden; background: #0d1117; font-family: sans-serif; }}
  #info {{ position: absolute; top: 10px; left: 20px; color: #c9d1d9; font-size: 11px; line-height: 1.5; z-index: 10; }}
  .legend {{ position: absolute; bottom: 20px; right: 20px; color: #8b949e; font-size: 10px; background: rgba(13,17,23,0.75); padding: 6px 10px; border-radius: 4px; z-index: 10; }}
  .legend span {{ display: inline-block; width: 10px; height: 10px; margin-right: 3px; border-radius: 2px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <b>MBR 帘式膜 3D (行业标准模型 v5.0)</b><br>
  ASM1 | Critical Flux | Resistance-in-Series | EPS/SMP | DFCm | Slug Bubble | K_Lim<br>
  膜丝{sim.fiber_diameter}mm×{fl:.1f}m | {PHYS.sheet_count}帘 | SADm={sim.intensity:.0f} | d_32={d32:.2f}mm<br>
  TMP={sim.get_current_tmp():.1f}kPa | J_crit={sim.critical_flux.J_crit:.0f}LMH | DO={sim.mixed_liquor.do_level:.1f}mg/L<br>
  EPS={sim.eps_smp.eps:.0f}mg/gVSS | SMP={sim.eps_smp.smp:.1f}mg/L | ΔR20={sim.dfcm.delta_R20/1e12:.2f}e12<br>
  &#9888; TMP>{PHYS.tmp_industry_recommended:.0f}kPa需反洗 (行业推荐) | {'Slug气泡已形成' if slug_ok else 'Slug未形成'}
</div>
<div class="legend">
  <span style="background:#ddd"></span>膜壳 <span style="background:#4499ff"></span>膜丝
  <span style="background:#66aadd"></span>集水管 <span style="background:#ff8844"></span>曝气管
  <span style="background:#886633;opacity:0.4"></span>污泥
</div>
<script type="importmap">
{{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const SW={sw:.3f},FL={fl:.3f},SS={ss:.3f},SC={PHYS.sheet_count},TD={td:.3f};
const MH={mh:.3f},MD={md:.3f},HH={hh:.3f},HW={hw:.3f},SH={sh:.3f},PG={pg:.3f};
const SLACK={sl:.3f},FD={fd:.5f},VFC=6,VFCZ=3,SEGS=4,by={by:.3f};

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
grid.position.set(SW*0.5, by-0.01, TD*0.5);
scene.add(grid);

const caseMat = new THREE.MeshStandardMaterial({{color:0xdddddd, metalness:0.05, roughness:0.6}});
const fiberMat = new THREE.MeshStandardMaterial({{color:0x4499ff, metalness:0.02, roughness:0.6, transparent:true, opacity:0.80}});
const headerMat = new THREE.MeshStandardMaterial({{color:0x66aadd, metalness:0.4, roughness:0.3}});
const pipeMat = new THREE.MeshStandardMaterial({{color:0xff8844, metalness:0.4, roughness:0.4}});
const sludgeMat = new THREE.MeshStandardMaterial({{color:0x775522, metalness:0, roughness:1.0, transparent:true, opacity:0.45}});
const segGeo = new THREE.CylinderGeometry(FD*0.5, FD*0.5, FL/SEGS, 4);
const dummy = new THREE.Object3D();

for (let si=0; si<SC; si++) {{
  const cz=si*(MD+SS), yTop=FL+0.15;
  const fg=new THREE.BoxGeometry(SW,FL,0.006);
  for (let side=-1; side<=1; side+=2) {{
    const p=new THREE.Mesh(fg,caseMat);
    p.position.set(SW*0.5,FL*0.5,cz+side*MD*0.5);
    scene.add(p);
  }}
  const tg=new THREE.BoxGeometry(SW,0.012,MD);
  const t=new THREE.Mesh(tg,caseMat); t.position.set(SW*0.5,yTop,cz); scene.add(t);
  const hg=new THREE.CylinderGeometry(HH*0.5,HH*0.5,SW+HW*2,14);
  const th=new THREE.Mesh(hg,headerMat); th.rotation.z=Math.PI/2; th.position.set(SW*0.5,yTop,cz); scene.add(th);
  const bg=new THREE.CylinderGeometry(HH*0.4,HH*0.4,SW+HW*0.5,14);
  const bh=new THREE.Mesh(bg,headerMat); bh.rotation.z=Math.PI/2; bh.position.set(SW*0.5,0.04,cz); scene.add(bh);
  const sg=new THREE.CylinderGeometry(HH*0.35,HH*0.35,MD*0.5,10);
  for (let side=-1; side<=1; side+=2) {{
    const s=new THREE.Mesh(sg,headerMat); s.rotation.x=Math.PI/2;
    s.position.set(side>0?SW*0.98:SW*0.02,yTop-HH*0.3,cz+side*MD*0.25);
    scene.add(s);
  }}
  const fc=VFC*VFCZ*SEGS;
  const fm=new THREE.InstancedMesh(segGeo,fiberMat,fc);
  let idx=0;
  for (let fi=0; fi<VFC; fi++) {{
    const fx=(fi+0.5)/VFC*(SW-HW*1.5)+HW*0.75;
    const bx=(Math.random()-0.5)*SLACK,bz=(Math.random()-0.5)*SLACK*0.4;
    for (let fzi=0; fzi<VFCZ; fzi++) {{
      const fz=cz-MD*0.45+(fzi+0.5)/VFCZ*MD*0.9;
      for (let s=0; s<SEGS; s++) {{
        const t=(s+0.5)/SEGS,curve=Math.sin(t*Math.PI);
        dummy.position.set(fx+bx*curve,s*(FL/SEGS)+FL/SEGS*0.5,fz+bz*curve);
        dummy.updateMatrix(); fm.setMatrixAt(idx++,dummy.matrix);
      }}
    }}
  }}
  fm.instanceMatrix.needsUpdate=true; scene.add(fm);
}}

const pipeR=0.018,pipeY=-PG;
const pGeo=new THREE.CylinderGeometry(pipeR,pipeR,SW*0.85,10);
for (let si=0; si<SC; si++) {{
  const cz=si*(MD+SS);
  const mp=new THREE.Mesh(pGeo,pipeMat); mp.rotation.z=Math.PI/2; mp.position.set(SW*0.5,pipeY,cz); scene.add(mp);
  const brGeo=new THREE.CylinderGeometry(pipeR*0.5,pipeR*0.5,PG*0.7,6);
  for (let pi=0; pi<4; pi++) {{
    const px=SW*0.18+pi*SW*0.18;
    const br=new THREE.Mesh(brGeo,pipeMat); br.position.set(px,pipeY+PG*0.35,cz); scene.add(br);
  }}
}}

const sTopY=by+SH;
const sgGeo=new THREE.BoxGeometry(SW*1.1,SH,TD*1.1);
const sludge=new THREE.Mesh(sgGeo,sludgeMat); sludge.position.set(SW*0.5,(by+sTopY)*0.5,TD*0.5); scene.add(sludge);
const wGeo=new THREE.BoxGeometry(SW*1.1,0.005,TD*1.1);
const wMat=new THREE.MeshBasicMaterial({{color:0x1a4a6e,transparent:true,opacity:0.3}});
const wl=new THREE.Mesh(wGeo,wMat); wl.position.set(SW*0.5,sTopY,TD*0.5); scene.add(wl);

const bGeo=new THREE.SphereGeometry(0.008,5,5);
const bMat=new THREE.MeshBasicMaterial({{color:0xaaddff,transparent:true,opacity:0.5}});
const bubbles=[];
for (let i=0; i<50; i++) {{
  const b=new THREE.Mesh(bGeo,bMat);
  const si=Math.floor(Math.random()*SC);
  b.position.set(SW*0.1+Math.random()*SW*0.8,pipeY+Math.random()*(FL*0.8),si*(MD+SS)+Math.random()*MD);
  b.userData={{speed:0.004+Math.random()*0.008,ox:b.position.x,oz:b.position.z,phase:Math.random()*Math.PI*2}};
  scene.add(b); bubbles.push(b);
}}

function animate(time) {{
  requestAnimationFrame(animate);
  const t=time*0.001;
  bubbles.forEach(b=>{{
    b.position.y+=b.userData.speed;
    b.position.x=b.userData.ox+Math.sin(t*2+b.userData.phase)*0.003;
    if (b.position.y>FL+0.15) {{ b.position.y=pipeY; const si=Math.floor(Math.random()*SC); b.userData.oz=si*(MD+SS)+Math.random()*MD; b.userData.ox=SW*0.1+Math.random()*SW*0.8; b.position.x=b.userData.ox; b.position.z=b.userData.oz; }}
  }});
  controls.update(); renderer.render(scene,camera);
}}
requestAnimationFrame(animate);
window.addEventListener('resize',()=>{{ camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth,window.innerHeight); }});
</script>
</body>
</html>"""


# ==================== Streamlit UI ====================
def init_session() -> None:
    if "sim" not in st.session_state:
        st.session_state.sim = MBRSimulator()
    if "preset_key" not in st.session_state:
        st.session_state.preset_key = "flush"


def render_sidebar(sim: MBRSimulator) -> None:
    with st.sidebar:
        st.header("🔬 MBR 行业标准模型 v5.0")

        st.subheader("📋 预设方案")
        cols = st.columns(len(PRESETS))
        for i, (key, p) in enumerate(PRESETS.items()):
            with cols[i]:
                active = st.session_state.get("preset_key") == key
                if st.button(f"{p.icon}", key=f"preset_{key}", help=p.label,
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    sim.apply_preset(key)
                    st.session_state.preset_key = key
                    st.rerun()

        st.divider()
        st.subheader("🌊 曝气参数")
        m_opts = [m.value for m in AerationMode]
        idx = m_opts.index(sim.mode.value)
        sim.mode = AerationMode(st.selectbox("曝气模式", m_opts,
                              format_func=lambda x: AerationMode(x).label, index=idx))
        sim.intensity = st.slider("曝气强度 SADm (Nm³/m²/h)", 50, 150, int(sim.intensity), step=5)
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
        sim.flux = st.slider("设计通量 (LMH)", 5.0, 30.0, float(sim.flux), step=1.0)

        st.divider()
        st.subheader("🧫 污泥与工艺")
        sim.mlss = st.slider("MLSS (mg/L)", 2000, 15000, int(sim.mlss), step=500)
        sim.srt = st.slider("SRT (d)", 5, 100, int(sim.srt), step=1)
        sim.hrt = st.slider("HRT (h)", 2.0, 12.0, float(sim.hrt), step=0.5)
        sim.return_ratio = st.slider("回流比 (%)", 50, 300, int(sim.return_ratio), step=10)

        op_cols = st.columns(4)
        actions = [
            ("排泥", "⬇️", lambda: sim.discharge_sludge()),
            ("反洗", "🧼", lambda: sim.perform_backwash()),
            ("+1h", "⏱", lambda: sim.step_simulation(1.0)),
            ("重置", "🔄", lambda: sim.reset()),
        ]
        for ci, (label, icon, fn) in enumerate(actions):
            with op_cols[ci]:
                if st.button(icon, key=f"op_{ci}", help=label, use_container_width=True):
                    fn()
                    st.rerun()

        sp = sim.sludge_level / PHYS.sludge_layer_max_height * 100
        st.progress(min(100, int(sp)), text=f"污泥层 {sim.sludge_level*1000:.0f}mm ({sp:.0f}%)")

        st.divider()
        st.subheader("⏩ 批量推进")
        bcols = st.columns(3)
        for ci, h in enumerate([2, 6, 12]):
            with bcols[ci]:
                if st.button(f"+{h}h", key=f"batch_{h}", use_container_width=True):
                    sim.step_simulation(float(h))
                    st.rerun()


def render_metrics(m: SimulationMetrics) -> None:
    tmp_warn = f">{PHYS.tmp_industry_recommended:.0f}kPa需反洗" if m.tmp > PHYS.tmp_industry_recommended else None
    flux_warn = f">J_crit {m.J_sustainable:.0f}" if m.J_actual > m.J_sustainable else None

    rows = [
        [("SEC", f"{m.sec}kWh/m³", ">1.2行业基准" if m.sec>1.2 else None),
         ("SADm", f"{m.sadm}Nm³/m²/h", None), ("SADp", f"{m.sadp}Nm³/m³", None),
         ("SADp_crit", f"{m.sadp_crit}Nm³/m³", None)],
        [("DO", f"{m.do_level}mg/L", None), ("MLR", f"{m.mlr:.1f}x", None),
         ("气含率", f"{m.gas_holdup}%", None), ("d_32", f"{m.d_32}mm", None)],
        [("TMP", f"{m.tmp}kPa", tmp_warn),
         ("J_crit", f"{m.J_crit}LMH", None), ("J_sus", f"{m.J_sustainable}LMH", None),
         ("J_actual", f"{m.J_actual}LMH", flux_warn)],
        [("τ_avg", f"{m.shear_avg}Pa", None), ("τ_max", f"{m.shear_max}Pa", None),
         ("k", f"{m.k_turb:.4f}", None), ("ε", f"{m.epsilon_turb:.4f}", None)],
        [("COD", f"{m.cod_eff}mg/L", None), ("NH4", f"{m.nh4_eff}mg/L", None),
         ("NO3", f"{m.no3_eff}mg/L", None), ("TN", f"{m.tn_eff}mg/L", None)],
        [("EPS", f"{m.eps}mg/gVSS", None), ("SMP", f"{m.smp}mg/L", None),
         ("ΔR20", f"{m.dfcm_index}e12", ">2需调整" if m.dfcm_index>2.0 else None),
         ("积垢风险", m.risk_text, "inverse" if m.risk_text=="HIGH" else "normal")],
        [("R_m", f"{m.Rm:.1e}", None), ("R_cake", f"{m.Rc:.1e}", None),
         ("R_pore", f"{m.Rp:.1e}", None), ("R_irr", f"{m.R_irr:.1e}", None)],
        [("SVI", f"{m.svi}mL/g", None), ("TSS", f"{m.tss}mg/L", None),
         ("MLVSS", f"{m.mlvss}mg/L", None), ("膜面积", f"{m.total_area}㎡", None)],
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


def render_trend(sim: MBRSimulator) -> None:
    with st.expander("📈 12小时趋势 (ASM1·TMP·DO·EPS·d_32·k-ε)", expanded=False):
        df = sim.predict_trend(hours=12.0, steps=60)
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("COD/NH4/NO3/TN 出水 (mg/L)", "TMP & J_crit (kPa/LMH)",
                            "DO & EPS & SMP (mg/L, mg/gVSS)",
                            "k & ε (m²/s², m²/s³)"))
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["cod_mgl"], name="COD", line=dict(color="#00f2ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["nh4_mgl"], name="NH4", line=dict(color="#44ff88")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["no3_mgl"], name="NO3", line=dict(color="#ffaa44")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["tn_mgl"], name="TN", line=dict(color="#ff44aa")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["tmp_kpa"], name="TMP", line=dict(color="#ff6644")), row=2, col=1)
        fig.add_hline(y=PHYS.tmp_industry_recommended, line_dash="dash", line_color="#ff6644",
                      annotation_text="行业推荐TMP≤30kPa", row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["J_crit"], name="J_crit", line=dict(color="#aaaaaa", dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["J_sus"], name="J_sus", line=dict(color="#888888", dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["do_mgl"], name="DO", line=dict(color="#44ff88")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["eps_mgg"], name="EPS", line=dict(color="#ffaa44"), yaxis="y2"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["smp_mgl"], name="SMP", line=dict(color="#ff44aa"), yaxis="y2"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["k_m2s2"], name="k", line=dict(color="#aa44ff")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time_h"], y=df["eps_m2s3"], name="ε", line=dict(color="#ff44aa"), yaxis="y2"), row=4, col=1)
        fig.update_layout(height=700, template="plotly_dark", hovermode="x unified")
        fig.update_xaxes(title_text="时间 (h)", row=4, col=1)
        fig.update_yaxes(title_text="EPS/SMP", row=3, col=1, overlaying="y", side="right")
        fig.update_yaxes(title_text="ε", row=4, col=1, overlaying="y", side="right")
        st.plotly_chart(fig, use_container_width=True)


def render_openfoam_export(sim: MBRSimulator) -> None:
    with st.expander("⚙️ OpenFOAM CFD 算例导出", expanded=False):
        st.info("生成 blockMeshDict + transportProperties + turbulenceProperties + 0/alpha.air")
        if st.button("📦 下载 OpenFOAM 算例 (ZIP)"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                Ug = sim.get_superficial_gas_velocity()
                pg = sim.pipe_to_membrane_gap / 1000.0
                sw = PHYS.sheet_width
                ch = sim.f_len + 0.4
                cw = sw * PHYS.sheet_count + 0.2
                zf.writestr("system/blockMeshDict", f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices ((0 0 -0.2) ({cw} 0 -0.2) ({cw} {ch} -0.2) (0 {ch} -0.2) (0 0 0.2) ({cw} 0 0.2) ({cw} {ch} 0.2) (0 {ch} 0.2));
blocks (hex (0 1 2 3 4 5 6 7) (40 60 10) simpleGrading (1 1 1));
boundary (walls {{ type wall; faces ((0 3 7 4) (1 2 6 5)); }} membrane {{ type wall; faces ((3 7 6 2)); }} aerator {{ type patch; faces ((0 4 5 1)); }} atmosphere {{ type patch; faces ((4 5 6 7)); }});
""")
                zf.writestr("constant/transportProperties", f"""FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}
phases (water air);
water {{ transportModel Newtonian; nu {PHYS.mu_l:.1e}; rho {PHYS.rho_l:.1f}; }}
air {{ transportModel Newtonian; nu 1.48e-5; rho 1.225; }}
sigma {PHYS.sigma:.3f};
""")
                zf.writestr("constant/turbulenceProperties", """FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType RAS;
RAS { RASModel kEpsilon; turbulence on; printCoeffs on; }
""")
                zf.writestr("0/alpha.air", f"""FoamFile {{ version 2.0; format ascii; class volScalarField; location "0"; object alpha.air; }}
dimensions [0 0 0 0 0 0 0];
internalField uniform 0.05;
boundaryField {{ aerator {{ type fixedValue; value uniform {sim.drift_flux.alpha_g:.4f}; }} }}
""")
            buf.seek(0)
            st.download_button("下载算例包", data=buf.getvalue(),
                              file_name="MBR_OpenFOAM_case.zip", mime="application/zip")


def render_export_panel(sim: MBRSimulator) -> None:
    with st.expander("💾 导出与数据", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            snap = json.dumps(sim.get_state_snapshot(), indent=2, ensure_ascii=False)
            st.download_button("📄 导出配置 (JSON)", data=snap,
                              file_name="MBR_v5_config.json", mime="application/json")
        with c2:
            df = sim.predict_trend(hours=12.0, steps=50)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("📊 导出趋势 (CSV)", data=buf.getvalue(),
                              file_name="MBR_v5_trend.csv", mime="text/csv")
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
        page_title="MBR 行业标准模型 v5.0",
        page_icon="💧", layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session()
    sim: MBRSimulator = st.session_state.sim

    render_sidebar(sim)

    st.title("💧 MBR 行业标准专业模型仿真系统 v5.0")
    st.caption(
        "ASM1 (IWA) · Critical Flux (Field 1995) · Resistance-in-Series · "
        "EPS/SMP (Laspidou & Rittmann) · DFCm (Delft) · Slug Bubble (Wang 2018) · "
        "K_Lim (Jun & Daigger 2024) · SADm/SADp · TMP≤30kPa 行业推荐"
    )

    m = sim.get_metrics()
    render_metrics(m)

    st.markdown("### 🖥️ 3D 可视化视图")
    st.components.v1.html(generate_3d_html(sim), height=600, scrolling=False)

    render_trend(sim)
    render_openfoam_export(sim)
    render_export_panel(sim)


if __name__ == "__main__":
    main()