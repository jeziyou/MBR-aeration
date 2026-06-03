import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import math

# ==================== 常数与物理定义 ====================
CONSTANTS = {
    "SHEET_COUNT": 5,
    "SHEET_WIDTH": 1.25,
    "SHEET_END_MARGIN": 0.05,
    "PIPE_OFFSET": 0.45,
    "EFFECT_DECAY_RATE": 14,
    "SLUDGE_LAYER_MAX_HEIGHT": 0.6,
    "SLUDGE_SETTLE_RATE": 0.0002,
    "SLUDGE_DISCHARGE_RATE": 0.02,
    "UNIF_PENALTY_FACTOR": 0.4,
    "PULSE_POWER_BOOST": 1.4,
    "OFF_PHASE_POWER": 0.05,
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

class MBRSimulator:
    """工艺仿真核心逻辑（与之前相同，增加了生成3D参数的方法）"""
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

    def apply_preset(self, name: str):
        preset = PRESETS[name]
        for k, v in preset.items():
            setattr(self, k, v)

    def get_sheet_area(self) -> float:
        base_area = 40.0 if self.fiber_diameter <= 1.8 else 25.0
        return round(base_area * (self.thickness / 30.0) * (self.f_len / 2.0), 2)

    def get_total_area(self) -> float:
        return self.get_sheet_area() * CONSTANTS["SHEET_COUNT"]

    def get_effective_intensity_factor(self, burst_active: bool = True) -> float:
        norm_intensity = (self.intensity - 50.0) / 100.0
        if self.mode == "pulse":
            duty = min(1.0 / self.pulse_period, 0.5)
            if burst_active:
                pwr = norm_intensity * CONSTANTS["PULSE_POWER_BOOST"]
            else:
                pwr = CONSTANTS["OFF_PHASE_POWER"]
            return np.clip(duty * pwr + (1-duty)*CONSTANTS["OFF_PHASE_POWER"], 0.05, 1.2)
        return np.clip(norm_intensity, 0.1, 1.0)

    def calculate_shear_stress(self) -> tuple:
        factor = self.get_effective_intensity_factor(True)
        slack_factor = 1.0 + self.slack * 12.0
        orifice_factor = (4.0 / max(1.5, self.h_size)) ** 0.4
        pipe_factor = np.exp(-self.p_pitch / 180.0)
        sheet_factor = np.exp(-self.s_pitch / 100.0)
        base = 0.3 * factor * slack_factor * orifice_factor * (pipe_factor+0.3) * (sheet_factor+0.5)
        avg = np.clip(base, 0.1, 3.5)
        max_mult = 1.8 if self.mode == "pulse" else 1.2
        max_shear = np.clip(avg * max_mult * (1+self.slack*3), 0.2, 6.0)
        return round(avg, 3), round(max_shear, 3)

    def calculate_sec(self) -> float:
        norm = (self.intensity - 50)/100
        base = 0.12 + norm * 0.35
        if self.mode == "pulse":
            duty = min(1.0/self.pulse_period, 0.5)
            base *= (duty*1.2 + (1-duty)*0.3)
        return round(base, 3)

    def calculate_uniformity(self) -> float:
        diff = abs(self.p_pitch - self.s_pitch)
        return round(max(0, 100 - diff * CONSTANTS["UNIF_PENALTY_FACTOR"]), 1)

    def calculate_risk_level(self, max_shear: float):
        if max_shear < 0.8:
            return "HIGH", "risk-high"
        if max_shear < 1.8:
            return "MEDIUM", "risk-medium"
        return "LOW", "risk-low"

    def calculate_svi(self) -> float:
        svi = 200 - self.srt*3 + (self.mlss/10000)*50
        return round(np.clip(svi, 50, 280), 1)

    def calculate_tss(self, avg_shear: float) -> float:
        base = 5 + (self.sludge_level/CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])*25
        return round(np.clip(base + avg_shear*1.5*2, 4, 35), 1)

    def update_sludge_level(self, dt: float = 1.0):
        if self.is_discharging:
            self.sludge_level = max(0, self.sludge_level - CONSTANTS["SLUDGE_DISCHARGE_RATE"]*dt)
        else:
            mlss_norm = (self.mlss-2000)/13000
            return_factor = np.clip(self.return_ratio/100, 0.5, 3.0)
            settling_effect = self.settling_rate/2.5
            acc = CONSTANTS["SLUDGE_SETTLE_RATE"] * mlss_norm * return_factor * settling_effect
            self.sludge_level += acc * dt
            self.sludge_level = min(self.sludge_level, CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])

    def discharge_sludge(self):
        self.sludge_level = max(0, self.sludge_level - 0.08)
        self.is_discharging = False

    def get_metrics(self):
        avg, max_sh = self.calculate_shear_stress()
        risk_text, risk_class = self.calculate_risk_level(max_sh)
        return {
            "sec": self.calculate_sec(),
            "shear_avg": avg,
            "shear_max": max_sh,
            "uniformity": self.calculate_uniformity(),
            "risk_text": risk_text,
            "risk_class": risk_class,
            "svi": self.calculate_svi(),
            "tss": self.calculate_tss(avg),
            "total_area": self.get_total_area(),
            "fiber_count": int(self.get_total_area()/(np.pi*(self.fiber_diameter/1000)*self.f_len))*CONSTANTS["SHEET_COUNT"],
            "sludge_level_mm": self.sludge_level*1000,
            "sludge_percent": (self.sludge_level/CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])*100
        }

# ==================== 生成嵌入的HTML（Three.js 3D场景）====================
def generate_3d_html(sim: MBRSimulator) -> str:
    """根据当前模拟参数生成完整的Three.js场景HTML"""
    # 将参数转换为JavaScript可用的格式
    js_config = {
        "sheetCount": CONSTANTS["SHEET_COUNT"],
        "sheetWidth": CONSTANTS["SHEET_WIDTH"],
        "sheetEndMargin": CONSTANTS["SHEET_END_MARGIN"],
        "pipeOffset": CONSTANTS["PIPE_OFFSET"],
        "effectDecayRate": CONSTANTS["EFFECT_DECAY_RATE"],
        "sludgeLevel": sim.sludge_level,
        "sludgeMaxHeight": CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"],
        "fLen": sim.f_len,
        "sPitch": sim.s_pitch,
        "pPitch": sim.p_pitch,
        "hSize": sim.h_size,
        "slack": sim.slack,
        "intensity": sim.intensity,
        "mode": sim.mode,
        "pulsePeriod": sim.pulse_period,
        "fiberDiameter": sim.fiber_diameter,
        "thickness": sim.thickness,
        "mlss": sim.mlss,
        "settlingRate": sim.settling_rate,
        "returnRatio": sim.return_ratio
    }
    # 计算纤维视觉数量（简化）
    sheet_area = sim.get_sheet_area()
    diameter_m = sim.fiber_diameter/1000
    area_per_fiber = np.pi * diameter_m * sim.f_len
    real_fibers = max(1, int(sheet_area / area_per_fiber))
    visual_fibers = min(real_fibers, 300)
    js_config["visualFibersPerSheet"] = visual_fibers

    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
            #info {{
                position: absolute;
                top: 20px;
                left: 20px;
                color: white;
                background: rgba(0,0,0,0.6);
                padding: 8px 15px;
                border-radius: 8px;
                pointer-events: none;
                z-index: 10;
                font-size: 12px;
                backdrop-filter: blur(5px);
            }}
        </style>
    </head>
    <body>
        <div id="info">
            MBR 3D 实时仿真 | 膜丝数量: {visual_fibers*CONSTANTS['SHEET_COUNT']} | 污泥层: {sim.sludge_level*1000:.0f} mm
        </div>
        <script type="importmap">
            {{
                "imports": {{
                    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
                }}
            }}
        </script>
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            // 接收后端参数
            const CONFIG = {json.dumps(js_config)};

            // 常数定义
            const SHEET_COUNT = CONFIG.sheetCount;
            const SHEET_WIDTH = CONFIG.sheetWidth;
            const SHEET_END_MARGIN = CONFIG.sheetEndMargin;
            const PIPE_OFFSET = CONFIG.pipeOffset;
            const EFFECT_DECAY_RATE = CONFIG.effectDecayRate;
            const SLUDGE_LEVEL = CONFIG.sludgeLevel;
            const SLUDGE_MAX_H = CONFIG.sludgeMaxHeight;
            const F_LEN = CONFIG.fLen;
            const S_PITCH_M = CONFIG.sPitch / 1000;
            const P_PITCH_M = CONFIG.pPitch / 1000;
            const H_SIZE = CONFIG.hSize;
            const SLACK = CONFIG.slack;
            const INTENSITY = CONFIG.intensity;
            const MODE = CONFIG.mode;
            const PULSE_PERIOD = CONFIG.pulsePeriod;
            const FIBER_DIA = CONFIG.fiberDiameter;
            const VISUAL_FIBERS = CONFIG.visualFibersPerSheet;

            // 物理辅助参数
            const WATER_SURFACE_ABOVE = 0.3;
            const TANK_MARGIN = 0.4;
            const TANK_EXTRA_HEIGHT = 1.0;
            const HEADER_HEIGHT = 0.12;
            const HEADER_DEPTH = 0.03;
            const RAIL_SIZE = 0.04;
            const PIPE_RADIUS = 0.025;
            const ORIFICE_RADIUS = 0.015;
            const BUBBLE_COUNT = 3000;
            const SLUDGE_PARTICLE_COUNT = 2000;
            const FIBER_SEGMENTS = 12;
            const BUBBLE_MIN_VEL = 0.015;
            const BUBBLE_MAX_VEL = 0.025;
            const BUBBLE_BASE_SCALE = 0.007;
            const BUBBLE_EXTRA_VEL_FACTOR = 1.5;
            const BUBBLE_FIBER_COUPLING = 0.025;
            const WAVE_FREQ_BASE = 1.8;
            const WAVE_FREQ_INTENSITY_FACTOR = 6.0;
            const MAX_BASE_AMPLITUDE = 0.045;
            const MAX_BUBBLE_PUSH = 0.012;
            const MAX_TOTAL_SHIFT = 0.06;
            const MAX_VERTICAL_PUSH = 0.015;
            const PULSE_POWER_BOOST = 1.4;
            const OFF_PHASE_POWER = 0.05;
            const SLUDGE_MIN_VEL = -0.008;
            const SLUDGE_MAX_VEL = -0.002;
            const SLUDGE_BASE_SCALE = 0.004;
            const SLUDGE_TURBULENCE = 0.003;
            
            // 全局变量
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x01050a);
            scene.fog = new THREE.FogExp2(0x01050a, 0.02);
            
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(6, 4.5, 8);
            camera.lookAt(0, 0, 0);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.target.set(0, 0, 0);
            
            // 光照
            scene.add(new THREE.AmbientLight(0x8899bb, 0.5));
            const dirLight = new THREE.DirectionalLight(0xffffff, 2);
            dirLight.position.set(5, 12, 5);
            scene.add(dirLight);
            const rim = new THREE.PointLight(0x00f2ff, 0.8, 25);
            rim.position.set(0, 6, -4);
            scene.add(rim);
            
            // 材质
            const headerMat = new THREE.MeshStandardMaterial({{ color: 0xF5DEB3, metalness: 0.05, roughness: 0.7 }});
            const railMat = new THREE.MeshStandardMaterial({{ color: 0xC8C8C8, metalness: 0.9, roughness: 0.2 }});
            const pipeMat = new THREE.MeshStandardMaterial({{ color: 0x445566, metalness: 0.85, roughness: 0.3 }});
            const orificeMat = new THREE.MeshBasicMaterial({{ color: 0x00f2ff, transparent: true, opacity: 0.65 }});
            const fiberMat = new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.85 }});
            const bubbleMat = new THREE.MeshBasicMaterial({{ color: 0xaaddff, transparent: true, opacity: 0.6, depthWrite: false }});
            const sludgeMat = new THREE.MeshBasicMaterial({{ color: 0x8b5a2b, transparent: true, opacity: 0.7, depthWrite: false }});
            const sludgeLayerMat = new THREE.MeshStandardMaterial({{ color: 0x5a3a1a, transparent: true, opacity: 0.6, roughness: 0.9, side: THREE.DoubleSide }});
            const tankEdgeMat = new THREE.LineBasicMaterial({{ color: 0x1e4a85, transparent: true, opacity: 0.5 }});
            
            // 辅助函数
            function getPipePositions() {{
                const depth = S_PITCH_M * SHEET_COUNT + TANK_MARGIN;
                const count = Math.max(3, Math.round(depth / P_PITCH_M));
                const start = -depth/2 + P_PITCH_M/2;
                const positions = [];
                for (let i = 0; i < count; i++) {{
                    positions.push({{ x: 0, z: start + i * P_PITCH_M }});
                }}
                return positions;
            }}
            
            // 创建膜组件
            const sheetGroup = new THREE.Group();
            const halfLen = F_LEN/2;
            const halfWidth = SHEET_WIDTH/2;
            
            for (let i = 0; i < SHEET_COUNT; i++) {{
                const z = (i - 2) * S_PITCH_M;
                const sheet = new THREE.Group();
                // 上下集水槽
                const headerGeo = new THREE.BoxGeometry(SHEET_WIDTH, HEADER_HEIGHT, HEADER_DEPTH);
                const topHeader = new THREE.Mesh(headerGeo, headerMat);
                const bottomHeader = new THREE.Mesh(headerGeo, headerMat);
                topHeader.position.y = halfLen;
                bottomHeader.position.y = -halfLen;
                sheet.add(topHeader, bottomHeader);
                // 两侧导轨
                const railGeo = new THREE.BoxGeometry(RAIL_SIZE, F_LEN, RAIL_SIZE);
                const leftRail = new THREE.Mesh(railGeo, railMat);
                const rightRail = new THREE.Mesh(railGeo, railMat);
                leftRail.position.set(-halfWidth + SHEET_END_MARGIN, 0, 0);
                rightRail.position.set(halfWidth - SHEET_END_MARGIN, 0, 0);
                sheet.add(leftRail, rightRail);
                // 膜丝
                for (let f = 0; f < VISUAL_FIBERS; f++) {{
                    const x = VISUAL_FIBERS > 1 ? (f / (VISUAL_FIBERS-1)) * SHEET_WIDTH - halfWidth : 0;
                    const positions = new Float32Array(FIBER_SEGMENTS * 3);
                    for (let j = 0; j < FIBER_SEGMENTS; j++) {{
                        const y = (j / (FIBER_SEGMENTS-1)) * F_LEN - halfLen;
                        positions[j*3] = x;
                        positions[j*3+1] = y;
                        positions[j*3+2] = 0;
                    }}
                    const geom = new THREE.BufferGeometry();
                    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    const line = new THREE.Line(geom, fiberMat);
                    line.userData = {{ baseX: x, baseZ: z, offset: Math.random() * Math.PI * 2, segments: FIBER_SEGMENTS }};
                    sheet.add(line);
                }}
                sheet.position.z = z;
                sheetGroup.add(sheet);
            }}
            scene.add(sheetGroup);
            
            // 曝气管道
            const pipeGroup = new THREE.Group();
            const mainY = -halfLen - PIPE_OFFSET - 0.15;
            const mainPipeGeo = new THREE.CylinderGeometry(PIPE_RADIUS*1.2, PIPE_RADIUS*1.2, S_PITCH_M*SHEET_COUNT + TANK_MARGIN, 12);
            const mainPipe = new THREE.Mesh(mainPipeGeo, pipeMat);
            mainPipe.rotation.x = Math.PI/2;
            mainPipe.position.y = mainY;
            pipeGroup.add(mainPipe);
            
            const pipePositions = getPipePositions();
            const pipeLen = SHEET_WIDTH + 0.3;
            const pipeGeo = new THREE.CylinderGeometry(PIPE_RADIUS, PIPE_RADIUS, pipeLen, 12);
            const orificeRadius = ORIFICE_RADIUS * 1.5 * (H_SIZE/4);
            const orificeGeo = new THREE.SphereGeometry(orificeRadius, 6, 6);
            for (const pos of pipePositions) {{
                const pipe = new THREE.Mesh(pipeGeo, pipeMat);
                pipe.rotation.z = Math.PI/2;
                pipe.position.set(pos.x, -halfLen - PIPE_OFFSET, pos.z);
                for (let k = 0; k < 10; k++) {{
                    const orifice = new THREE.Mesh(orificeGeo, orificeMat);
                    orifice.position.set(PIPE_RADIUS*1.3, k/9*pipeLen - pipeLen/2, 0);
                    pipe.add(orifice);
                }}
                pipeGroup.add(pipe);
                // 连接管
                const connectorHeight = Math.abs(pos.y - mainY);
                const connector = new THREE.Mesh(
                    new THREE.CylinderGeometry(PIPE_RADIUS*0.7, PIPE_RADIUS*0.7, connectorHeight, 8),
                    pipeMat
                );
                connector.position.set(0, (pos.y + mainY)/2, pos.z);
                pipeGroup.add(connector);
            }}
            scene.add(pipeGroup);
            
            // 污泥层
            const sludgeLayerWidth = SHEET_WIDTH + TANK_MARGIN - 0.05;
            const depth = S_PITCH_M * SHEET_COUNT + TANK_MARGIN;
            const sludgeLayerDepth = depth - 0.05;
            const sludgeLayer = new THREE.Mesh(new THREE.BoxGeometry(sludgeLayerWidth, 1, sludgeLayerDepth), sludgeLayerMat);
            sludgeLayer.scale.y = Math.max(SLUDGE_LEVEL, 0.005);
            const bottomY = -halfLen - TANK_EXTRA_HEIGHT/2;
            sludgeLayer.position.y = bottomY + sludgeLayer.scale.y/2;
            scene.add(sludgeLayer);
            
            // 气泡实例网格
            const bubbles = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 6), bubbleMat, BUBBLE_COUNT);
            bubbles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            bubbles.frustumCulled = false;
            scene.add(bubbles);
            
            // 污泥颗粒实例
            const sludgeParticles = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 6, 4), sludgeMat, SLUDGE_PARTICLE_COUNT);
            sludgeParticles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            scene.add(sludgeParticles);
            
            // 数据存储
            const bubbleData = {{
                posX: new Float32Array(BUBBLE_COUNT),
                posY: new Float32Array(BUBBLE_COUNT),
                posZ: new Float32Array(BUBBLE_COUNT),
                vel: new Float32Array(BUBBLE_COUNT),
                phase: new Float32Array(BUBBLE_COUNT),
                size: new Float32Array(BUBBLE_COUNT),
                baseScale: new Float32Array(BUBBLE_COUNT),
                active: new Uint8Array(BUBBLE_COUNT)
            }};
            const sludgeData = {{
                posX: new Float32Array(SLUDGE_PARTICLE_COUNT),
                posY: new Float32Array(SLUDGE_PARTICLE_COUNT),
                posZ: new Float32Array(SLUDGE_PARTICLE_COUNT),
                velY: new Float32Array(SLUDGE_PARTICLE_COUNT),
                phase: new Float32Array(SLUDGE_PARTICLE_COUNT),
                size: new Float32Array(SLUDGE_PARTICLE_COUNT),
                active: new Uint8Array(SLUDGE_PARTICLE_COUNT),
                settled: new Uint8Array(SLUDGE_PARTICLE_COUNT),
                turbulence: new Float32Array(SLUDGE_PARTICLE_COUNT)
            }};
            
            function resetBubble(idx) {{
                const pipes = pipePositions;
                const pipe = pipes[Math.floor(Math.random() * pipes.length)];
                const hw = halfWidth - SHEET_END_MARGIN;
                bubbleData.posX[idx] = (Math.random() - 0.5) * hw * 2;
                bubbleData.posY[idx] = -halfLen - PIPE_OFFSET + PIPE_RADIUS*1.5 + 0.03;
                bubbleData.posZ[idx] = pipe.z + (Math.random()-0.5)*0.08;
                bubbleData.vel[idx] = BUBBLE_MIN_VEL + Math.random() * (BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.phase[idx] = Math.random() * Math.PI * 2;
                bubbleData.size[idx] = 0.7 + Math.random()*0.6;
                bubbleData.baseScale[idx] = 0.8 + Math.random()*0.4;
                bubbleData.active[idx] = 1;
            }}
            
            for (let i = 0; i < BUBBLE_COUNT; i++) resetBubble(i);
            
            // 污泥颗粒初始化
            const mlssNorm = Math.max(0, (CONFIG.mlss - 2000) / 13000);
            const activeRatio = 0.4 + mlssNorm*0.5;
            for (let i = 0; i < SLUDGE_PARTICLE_COUNT; i++) {{
                sludgeData.posX[i] = (Math.random()-0.5) * (SHEET_WIDTH - SHEET_END_MARGIN*2);
                sludgeData.posZ[i] = (Math.random()-0.5) * depth;
                const vertical = Math.random();
                sludgeData.posY[i] = bottomY + vertical*vertical*(F_LEN + TANK_EXTRA_HEIGHT);
                sludgeData.velY[i] = SLUDGE_MIN_VEL + Math.random()*(SLUDGE_MAX_VEL - SLUDGE_MIN_VEL);
                sludgeData.phase[i] = Math.random()*Math.PI*2;
                sludgeData.size[i] = 0.5 + Math.random();
                sludgeData.turbulence[i] = 0.5 + Math.random()*1.5;
                sludgeData.active[i] = Math.random() < activeRatio ? 1 : 0;
                sludgeData.settled[i] = 0;
            }}
            
            // 动画循环
            let lastTime = performance.now();
            let burstActive = true;
            let lastBurstState = true;
            let spawnAccum = 0;
            
            function animate() {{
                const now = performance.now();
                let dt = Math.min((now - lastTime) / 1000, 0.05);
                lastTime = now;
                const time = now / 1000;
                
                // 脉冲模式控制
                if (MODE === 'pulse') {{
                    const cycle = (time / PULSE_PERIOD) % 1;
                    burstActive = cycle < (1 / PULSE_PERIOD);
                }} else {{
                    burstActive = true;
                }}
                
                const intensityNorm = (INTENSITY - 50) / 100;
                const pwr = burstActive ? (MODE==='pulse' ? intensityNorm*PULSE_POWER_BOOST : intensityNorm) : OFF_PHASE_POWER;
                
                // 更新气泡
                const frameScale = dt * 60;
                const baseScale = H_SIZE * BUBBLE_BASE_SCALE;
                const targetActive = Math.floor(BUBBLE_COUNT * intensityNorm * 0.7);
                const spawnRate = burstActive ? targetActive * 1.2 * dt : 0;
                spawnAccum += spawnRate;
                let activeCount = 0;
                let spawns = 0;
                const dummy = new THREE.Object3D();
                for (let i = 0; i < BUBBLE_COUNT; i++) {{
                    if (!bubbleData.active[i] && activeCount < targetActive && spawnAccum >= 1 && spawns < 5) {{
                        resetBubble(i);
                        spawnAccum -= 1;
                        spawns++;
                        activeCount++;
                        continue;
                    }}
                    if (!bubbleData.active[i]) continue;
                    activeCount++;
                    bubbleData.posY[i] += bubbleData.vel[i] * (1 + intensityNorm*BUBBLE_EXTRA_VEL_FACTOR) * frameScale;
                    if (bubbleData.posY[i] >= halfLen + WATER_SURFACE_ABOVE) {{
                        bubbleData.active[i] = 0;
                        dummy.scale.setScalar(0);
                        dummy.updateMatrix();
                        bubbles.setMatrixAt(i, dummy.matrix);
                        continue;
                    }}
                    dummy.position.set(bubbleData.posX[i], bubbleData.posY[i], bubbleData.posZ[i]);
                    const scaleVar = baseScale * bubbleData.size[i];
                    dummy.scale.setScalar(scaleVar);
                    dummy.updateMatrix();
                    bubbles.setMatrixAt(i, dummy.matrix);
                }}
                bubbles.instanceMatrix.needsUpdate = true;
                
                // 更新膜丝摆动（动态修改几何体）
                const allFibers = [];
                sheetGroup.children.forEach(sheet => {{
                    sheet.children.forEach(child => {{
                        if (child.isLine) allFibers.push(child);
                    }});
                }});
                const pipePosArray = pipePositions;
                for (const fiber of allFibers) {{
                    const geom = fiber.geometry;
                    const posAttr = geom.attributes.position;
                    const baseX = fiber.userData.baseX;
                    const baseZ = fiber.userData.baseZ;
                    const offset = fiber.userData.offset;
                    let minDist = Infinity;
                    for (const pp of pipePosArray) minDist = Math.min(minDist, Math.abs(baseZ - pp.z));
                    const effect = Math.exp(-minDist * EFFECT_DECAY_RATE);
                    for (let seg = 0; seg < FIBER_SEGMENTS; seg++) {{
                        const ratio = seg / (FIBER_SEGMENTS-1);
                        const segY = (ratio - 0.5) * F_LEN;
                        const envelope = Math.sin(Math.PI * ratio);
                        const phase = time * (WAVE_FREQ_BASE + pwr * WAVE_FREQ_INTENSITY_FACTOR) + offset + seg * 0.2;
                        let amplitude = F_LEN * SLACK * pwr * envelope * (0.15 + effect*1.2);
                        amplitude = Math.min(amplitude, MAX_BASE_AMPLITUDE);
                        let shift = Math.sin(phase) * amplitude;
                        // 简化气泡推力（省略邻近搜索，性能优化）
                        shift = Math.min(MAX_TOTAL_SHIFT, Math.max(-MAX_TOTAL_SHIFT, shift));
                        posAttr.setX(seg, baseX + shift);
                        // 垂直摆动保持原Y
                    }}
                    posAttr.needsUpdate = true;
                }}
                
                // 更新污泥层视觉
                sludgeLayer.scale.y = Math.max(SLUDGE_LEVEL, 0.005);
                sludgeLayer.position.y = bottomY + sludgeLayer.scale.y/2;
                
                controls.update();
                renderer.render(scene, camera);
                requestAnimationFrame(animate);
            }}
            
            animate();
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_template

# ==================== Streamlit UI ====================
st.set_page_config(page_title="MBR 工业仿真 v14.4 - 3D版", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = MBRSimulator()

sim = st.session_state.sim

# 侧边栏控制（与之前完全相同）
with st.sidebar:
    st.markdown("## 🧪 MBR 系统控制")
    if st.button("🌿 节能模式"): sim.apply_preset("eco"); st.rerun()
    if st.button("⚖️ 均衡模式"): sim.apply_preset("balanced"); st.rerun()
    if st.button("💨 高冲刷模式"): sim.apply_preset("flush"); st.rerun()
    st.markdown("---")
    sim.mode = st.selectbox("曝气模式", ["cont","pulse"], format_func=lambda x:"连续曝气" if x=="cont" else "脉冲式曝气")
    sim.intensity = st.slider("曝气强度", 50,150, sim.intensity,5)
    if sim.mode == "pulse":
        sim.pulse_period = st.slider("脉冲周期(s)", 3.0,6.0, sim.pulse_period,0.5)
    sim.h_size = st.slider("曝气孔径(mm)", 1.0,15.0, sim.h_size,0.5)
    sim.p_pitch = st.slider("曝气管间距(mm)", 50,300, sim.p_pitch,10)
    st.markdown("---")
    sim.fiber_diameter = st.selectbox("膜丝外径", [1.65,2.8], format_func=lambda x:"1.65mm (40m²)" if x==1.65 else "2.8mm (25m²)")
    sim.thickness = st.slider("膜片厚度(mm)", 10,100, sim.thickness,5)
    sim.s_pitch = st.slider("膜片间距(mm)", 50,100, sim.s_pitch,5)
    sim.f_len = st.slider("膜丝长度(m)", 0.1,3.0, sim.f_len,0.1)
    sim.slack = st.slider("松弛度(%)", 0.2,5.0, sim.slack*100,0.2)/100
    st.markdown("---")
    sim.mlss = st.slider("MLSS(mg/L)", 2000,15000, sim.mlss,500)
    sim.srt = st.slider("污泥龄(d)", 5,40, sim.srt,1)
    sim.settling_rate = st.slider("沉降速率(m/h)", 0.5,6.0, sim.settling_rate,0.5)
    sim.return_ratio = st.slider("回流比(%)", 50,300, sim.return_ratio,10)
    if st.button("⬇️ 排泥"): sim.discharge_sludge(); st.rerun()
    if st.button("⏱️ 模拟1h"): sim.update_sludge_level(3600); st.rerun()
    sludge_percent = (sim.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])*100
    st.progress(min(100,int(sludge_percent)), text=f"污泥层 {sim.sludge_level*1000:.0f} mm")

# 主区域：上方指标卡片，下方嵌入3D视图
st.title("💧 MBR 工业仿真系统 v14.4 - 3D 实时渲染")
st.caption("参数调整后，3D 场景自动刷新（约1-2秒）")

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

# 嵌入3D HTML
st.markdown("### 🖥️ 3D 可视化视图")
html_code = generate_3d_html(sim)
st.components.v1.html(html_code, height=600, scrolling=False)

# 趋势图（可选）
with st.expander("📈 12小时趋势预测 (剪切力 & 污泥层)"):
    # 快速模拟趋势
    times = np.linspace(0,12,50)
    shear_vals = []
    sludge_vals = []
    temp = MBRSimulator()
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
    for t in times:
        avg, _ = temp.calculate_shear_stress()
        shear_vals.append(avg)
        sludge_vals.append(temp.sludge_level*1000)
        temp.update_sludge_level(3600*0.24)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=times, y=shear_vals, name="剪切力(Pa)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=sludge_vals, name="污泥层(mm)"), row=2, col=1)
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
