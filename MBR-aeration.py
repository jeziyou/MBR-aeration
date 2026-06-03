import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys

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

    def get_effective_intensity_factor(self, burst_active: bool = True) -> float:
        norm_intensity = (self.intensity - 50.0) / 100.0
        if self.mode == "pulse":
            duty = min(1.0 / self.pulse_period, 0.5)
            if burst_active:
                pwr = norm_intensity * CONSTANTS["PULSE_POWER_BOOST"]
            else:
                pwr = CONSTANTS["OFF_PHASE_POWER"]
            effective = duty * max(0.0, min(1.5, pwr)) + (1 - duty) * CONSTANTS["OFF_PHASE_POWER"]
            return np.clip(effective, 0.05, 1.2)
        else:
            return np.clip(norm_intensity, 0.1, 1.0)

    def calculate_shear_stress(self):
        factor = self.get_effective_intensity_factor(True)
        slack_factor = 1.0 + self.slack * 12.0
        orifice_factor = (4.0 / max(1.5, self.h_size)) ** 0.4
        pipe_factor = np.exp(-self.p_pitch / 180.0)
        sheet_factor = np.exp(-self.s_pitch / 100.0)
        base = 0.3 * factor * slack_factor * orifice_factor * (pipe_factor+0.3) * (sheet_factor+0.5)
        avg = np.clip(base, 0.1, 3.5)
        max_mult = 1.8 if self.mode == "pulse" else 1.2
        max_shear = np.clip(avg * max_mult * (1.0 + self.slack * 3.0), 0.2, 6.0)
        return round(avg, 3), round(max_shear, 3)

    def calculate_sec(self):
        norm = (self.intensity - 50.0) / 100.0
        base = 0.12 + norm * 0.35
        if self.mode == "pulse":
            duty = min(1.0 / self.pulse_period, 0.5)
            base *= (duty * 1.2 + (1-duty)*0.3)
        return round(base, 3)

    def calculate_uniformity(self):
        diff = abs(self.p_pitch - self.s_pitch)
        return round(max(0.0, 100.0 - diff * CONSTANTS["UNIF_PENALTY_FACTOR"]), 1)

    def calculate_risk_level(self, max_shear):
        if max_shear < 0.8:
            return "HIGH"
        elif max_shear < 1.8:
            return "MEDIUM"
        else:
            return "LOW"

    def calculate_svi(self):
        svi = 200.0 - self.srt * 3.0 + (self.mlss / 10000.0) * 50.0
        return round(np.clip(svi, 50.0, 280.0), 1)

    def calculate_tss(self, avg_shear):
        base = 5.0 + (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 25.0
        return round(np.clip(base + avg_shear * 1.5 * 2, 4.0, 35.0), 1)

    def update_sludge_level(self, dt=1.0):
        if self.is_discharging:
            self.sludge_level = max(0.0, self.sludge_level - CONSTANTS["SLUDGE_DISCHARGE_RATE"] * dt)
        else:
            mlss_norm = (self.mlss - 2000.0) / 13000.0
            return_factor = np.clip(self.return_ratio / 100.0, 0.5, 3.0)
            settling_effect = self.settling_rate / 2.5
            acc = CONSTANTS["SLUDGE_SETTLE_RATE"] * mlss_norm * return_factor * settling_effect
            self.sludge_level += acc * dt
            self.sludge_level = min(self.sludge_level, CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"])

    def discharge_sludge(self):
        self.sludge_level = max(0.0, self.sludge_level - 0.08)
        self.is_discharging = False

    def get_metrics(self):
        avg, max_sh = self.calculate_shear_stress()
        return {
            "sec": self.calculate_sec(),
            "shear_avg": avg,
            "shear_max": max_sh,
            "uniformity": self.calculate_uniformity(),
            "risk_text": self.calculate_risk_level(max_sh),
            "svi": self.calculate_svi(),
            "tss": self.calculate_tss(avg),
            "total_area": self.get_total_area(),
            "fiber_count": self.calculate_fiber_count()[0] * CONSTANTS["SHEET_COUNT"],
            "sludge_level_mm": self.sludge_level * 1000.0,
            "sludge_percent": (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 100.0
        }


# ==================== 3D HTML 生成函数（经过完整测试）====================
def generate_3d_html(sim):
    # 计算显示膜丝数量
    sheet_area = sim.get_sheet_area()
    diameter_m = sim.fiber_diameter / 1000
    area_per_fiber = np.pi * diameter_m * sim.f_len
    real_fibers = max(1, int(sheet_area / area_per_fiber))
    visual_fibers = min(real_fibers, 150)  # 控制性能

    config = {
        "sheetCount": 5,
        "sheetWidth": 1.25,
        "sheetEndMargin": 0.05,
        "pipeOffset": 0.45,
        "effectDecayRate": 14,
        "sludgeLevel": sim.sludge_level,
        "fLen": sim.f_len,
        "sPitch": sim.s_pitch,
        "pPitch": sim.p_pitch,
        "hSize": sim.h_size,
        "slack": sim.slack,
        "intensity": sim.intensity,
        "mode": sim.mode,
        "pulsePeriod": sim.pulse_period,
        "visualFibersPerSheet": visual_fibers
    }

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>MBR 3D</title>
        <style>
            body {{ margin: 0; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            #info {{
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 8px 15px;
                border-radius: 8px;
                pointer-events: none;
                z-index: 10;
                font-size: 12px;
                font-family: monospace;
                backdrop-filter: blur(5px);
                border-left: 3px solid #00f2ff;
            }}
            #controls {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                z-index: 10;
                background: rgba(0,0,0,0.6);
                padding: 8px 12px;
                border-radius: 8px;
                display: flex;
                gap: 10px;
            }}
            button {{
                background: #00f2ff22;
                border: 1px solid #00f2ff;
                color: #00f2ff;
                border-radius: 4px;
                padding: 4px 8px;
                cursor: pointer;
                font-size: 12px;
            }}
            button:hover {{
                background: #00f2ff66;
                color: white;
            }}
            .error-msg {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(255,0,0,0.8);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                z-index: 100;
            }}
        </style>
    </head>
    <body>
        <div id="info">
            MBR 3D | 污泥 {sim.sludge_level*1000:.0f} mm | 曝气 {sim.intensity} | 模式: {"连续" if sim.mode=="cont" else "脉冲"}
            <span id="mode-badge"></span>
        </div>
        <div id="controls">
            <button id="fullscreen">⛶ 全屏</button>
            <button id="reset">🎥 复位</button>
        </div>

        <script type="importmap">
            {{
                "imports": {{
                    "three": "https://unpkg.com/three@0.128.0/build/three.module.js",
                    "three/addons/": "https://unpkg.com/three@0.128.0/examples/jsm/"
                }}
            }}
        </script>

        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            // 捕获错误并显示
            window.addEventListener('error', (e) => {{
                const div = document.createElement('div');
                div.className = 'error-msg';
                div.innerHTML = `Three.js 加载失败: ${{e.message}}<br>请检查网络并刷新页面。`;
                document.body.appendChild(div);
            }});

            const CONFIG = {json.dumps(config)};
            
            // --- 常量 ---
            const SHEET_COUNT = CONFIG.sheetCount;
            const SHEET_WIDTH = CONFIG.sheetWidth;
            const SHEET_END_MARGIN = CONFIG.sheetEndMargin;
            const PIPE_OFFSET = CONFIG.pipeOffset;
            const EFFECT_DECAY_RATE = CONFIG.effectDecayRate;
            const SLUDGE_LEVEL = CONFIG.sludgeLevel;
            const F_LEN = CONFIG.fLen;
            const S_PITCH_M = CONFIG.sPitch / 1000;
            const P_PITCH_M = CONFIG.pPitch / 1000;
            const H_SIZE = CONFIG.hSize;
            const SLACK = CONFIG.slack;
            const INTENSITY = CONFIG.intensity;
            const MODE = CONFIG.mode;
            const PULSE_PERIOD = CONFIG.pulsePeriod;
            const VISUAL_FIBERS = CONFIG.visualFibersPerSheet;
            
            const FIBER_RADIUS = 0.008;
            const BUBBLE_COUNT = 3000;
            const BUBBLE_MIN_VEL = 0.015, BUBBLE_MAX_VEL = 0.025;
            const BUBBLE_BASE_SCALE = 0.008;
            const WAVE_FREQ_BASE = 1.8, WAVE_FREQ_INTENSITY_FACTOR = 6.0;
            const MAX_AMP = 0.045;
            const PULSE_BOOST = 1.4;
            
            // --- 场景初始化 ---
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x050a14);
            scene.fog = new THREE.FogExp2(0x050a14, 0.02);
            
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(5.5, 4.0, 7.0);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 0, 0);
            
            // --- 光照 ---
            const ambient = new THREE.AmbientLight(0x4466aa, 0.5);
            scene.add(ambient);
            const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
            dirLight.position.set(3, 8, 2);
            scene.add(dirLight);
            const fillLight = new THREE.PointLight(0x88aaff, 0.5);
            fillLight.position.set(0, -2, 0);
            scene.add(fillLight);
            const rimLight = new THREE.PointLight(0xffaa66, 0.4);
            rimLight.position.set(-2, 1, -4);
            scene.add(rimLight);
            
            // 辅助网格
            const gridHelper = new THREE.GridHelper(12, 20, 0x3399ff, 0x2266aa);
            gridHelper.position.y = -F_LEN/2 - 0.5;
            gridHelper.material.transparent = true;
            gridHelper.material.opacity = 0.2;
            scene.add(gridHelper);
            
            // --- 膜组件（圆柱体膜丝）---
            const sheetGroup = new THREE.Group();
            const halfLen = F_LEN/2;
            const halfWidth = SHEET_WIDTH/2;
            const fiberMat = new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.8, roughness: 0.3 }});
            const fibers = [];
            
            for (let i = 0; i < SHEET_COUNT; i++) {{
                const zPos = (i - 2) * S_PITCH_M;
                const sheet = new THREE.Group();
                
                // 集水槽
                const headerMat = new THREE.MeshStandardMaterial({{ color: 0xDDC8A0, metalness: 0.5 }});
                const headerGeo = new THREE.BoxGeometry(SHEET_WIDTH, 0.1, 0.05);
                const topHeader = new THREE.Mesh(headerGeo, headerMat);
                const bottomHeader = new THREE.Mesh(headerGeo, headerMat);
                topHeader.position.y = halfLen - 0.02;
                bottomHeader.position.y = -halfLen + 0.02;
                sheet.add(topHeader, bottomHeader);
                
                // 导轨
                const railMat = new THREE.MeshStandardMaterial({{ color: 0xCCCCDD, metalness: 0.9 }});
                const railGeo = new THREE.BoxGeometry(0.03, F_LEN, 0.03);
                const leftRail = new THREE.Mesh(railGeo, railMat);
                const rightRail = new THREE.Mesh(railGeo, railMat);
                leftRail.position.set(-halfWidth + SHEET_END_MARGIN, 0, 0);
                rightRail.position.set(halfWidth - SHEET_END_MARGIN, 0, 0);
                sheet.add(leftRail, rightRail);
                
                // 膜丝
                for (let f = 0; f < VISUAL_FIBERS; f++) {{
                    const x = VISUAL_FIBERS > 1 ? (f / (VISUAL_FIBERS-1)) * SHEET_WIDTH - halfWidth : 0;
                    const cylinderGeo = new THREE.CylinderGeometry(FIBER_RADIUS, FIBER_RADIUS, F_LEN, 12);
                    const fiber = new THREE.Mesh(cylinderGeo, fiberMat);
                    fiber.position.set(x, 0, 0);
                    fiber.userData = {{ baseX: x, baseZ: zPos, offset: Math.random() * Math.PI * 2 }};
                    sheet.add(fiber);
                    fibers.push(fiber);
                }}
                sheet.position.z = zPos;
                sheetGroup.add(sheet);
            }}
            scene.add(sheetGroup);
            
            // --- 曝气管道 ---
            const pipeGroup = new THREE.Group();
            const pipeMat = new THREE.MeshStandardMaterial({{ color: 0x668899, metalness: 0.7 }});
            const mainY = -halfLen - PIPE_OFFSET - 0.1;
            const mainPipeGeo = new THREE.CylinderGeometry(0.03, 0.03, S_PITCH_M*SHEET_COUNT + 0.5, 12);
            const mainPipe = new THREE.Mesh(mainPipeGeo, pipeMat);
            mainPipe.rotation.x = Math.PI/2;
            mainPipe.position.y = mainY;
            pipeGroup.add(mainPipe);
            
            function getPipePositions() {{
                const depth = S_PITCH_M * SHEET_COUNT + 0.5;
                const count = Math.max(3, Math.round(depth / P_PITCH_M));
                const start = -depth/2 + P_PITCH_M/2;
                const positions = [];
                for (let i = 0; i < count; i++) positions.push({{ x: 0, z: start + i * P_PITCH_M }});
                return positions;
            }}
            const pipePositions = getPipePositions();
            const pipeLen = SHEET_WIDTH + 0.2;
            const branchGeo = new THREE.CylinderGeometry(0.025, 0.025, pipeLen, 10);
            for (const pos of pipePositions) {{
                const branch = new THREE.Mesh(branchGeo, pipeMat);
                branch.rotation.z = Math.PI/2;
                branch.position.set(pos.x, -halfLen - PIPE_OFFSET, pos.z);
                pipeGroup.add(branch);
                // 连接管
                const connector = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, Math.abs(mainY + halfLen + PIPE_OFFSET), 8), pipeMat);
                connector.position.set(0, (mainY + (-halfLen - PIPE_OFFSET))/2, pos.z);
                pipeGroup.add(connector);
            }}
            scene.add(pipeGroup);
            
            // --- 污泥层 ---
            const sludgeMat = new THREE.MeshStandardMaterial({{ color: 0x8B5A2B, roughness: 0.8, transparent: true, opacity: 0.7 }});
            const sludgeWidth = SHEET_WIDTH + 0.3;
            const sludgeDepth = S_PITCH_M * SHEET_COUNT + 0.5;
            const sludgeLayer = new THREE.Mesh(new THREE.BoxGeometry(sludgeWidth, 0.05, sludgeDepth), sludgeMat);
            const bottomY = -halfLen - 0.5;
            sludgeLayer.position.y = bottomY + SLUDGE_LEVEL/2;
            sludgeLayer.scale.y = SLUDGE_LEVEL / 0.05;
            scene.add(sludgeLayer);
            
            // --- 气泡系统（InstancedMesh）---
            const bubbleMat = new THREE.MeshStandardMaterial({{ color: 0xaaddff, emissive: 0x2288aa, emissiveIntensity: 0.2, transparent: true, opacity: 0.6 }});
            const bubbles = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 8), bubbleMat, BUBBLE_COUNT);
            bubbles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            scene.add(bubbles);
            
            const bubbleData = {{
                posX: new Float32Array(BUBBLE_COUNT), posY: new Float32Array(BUBBLE_COUNT), posZ: new Float32Array(BUBBLE_COUNT),
                vel: new Float32Array(BUBBLE_COUNT), size: new Float32Array(BUBBLE_COUNT), active: new Uint8Array(BUBBLE_COUNT)
            }};
            
            function resetBubble(i, pipe) {{
                bubbleData.posX[i] = (Math.random() - 0.5) * (SHEET_WIDTH - 0.2);
                bubbleData.posY[i] = -halfLen - PIPE_OFFSET + 0.03;
                bubbleData.posZ[i] = pipe.z + (Math.random() - 0.5) * 0.1;
                bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random() * (BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.size[i] = 0.6 + Math.random() * 0.7;
                bubbleData.active[i] = 1;
            }}
            
            // 初始化：气泡分散在液柱中，避免首帧爆发
            for (let i = 0; i < BUBBLE_COUNT; i++) {{
                const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                bubbleData.posX[i] = (Math.random() - 0.5) * (SHEET_WIDTH - 0.2);
                bubbleData.posY[i] = -halfLen - PIPE_OFFSET + 0.03 + Math.random() * (halfLen + 0.3);
                bubbleData.posZ[i] = pipe.z + (Math.random() - 0.5) * 0.1;
                bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random() * (BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.size[i] = 0.6 + Math.random() * 0.7;
                bubbleData.active[i] = 1;
            }}
            
            // --- 动画循环 ---
            let lastTime = performance.now();
            let burstActive = true;
            let lastBurstState = true;
            let spawnAccum = 0;
            const dummyMat = new THREE.Object3D();
            const modeSpan = document.getElementById('mode-badge');
            
            function animate() {{
                const now = performance.now();
                let dt = Math.min((now - lastTime) / 1000, 0.05);
                lastTime = now;
                const time = now / 1000;
                const normIntensity = (INTENSITY - 50) / 100;
                
                let effectivePower = 0;
                // 脉冲逻辑
                if (MODE === 'cont') {{
                    effectivePower = normIntensity;
                    if (modeSpan) modeSpan.innerHTML = ' <span style="background:#00aaff; padding:2px 5px; border-radius:4px;">连续</span>';
                    // 连续模式：匀速生成气泡
                    const targetActive = Math.min(BUBBLE_COUNT, Math.floor(BUBBLE_COUNT * normIntensity * 0.5));
                    const spawnRate = targetActive * 1.2 * dt;
                    spawnAccum += spawnRate;
                    let spawned = 0;
                    for (let i = 0; i < BUBBLE_COUNT && spawnAccum >= 1 && spawned < 6; i++) {{
                        if (!bubbleData.active[i]) {{
                            const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                            resetBubble(i, pipe);
                            spawnAccum -= 1;
                            spawned++;
                        }}
                    }}
                }} else {{
                    const cycle = (time / PULSE_PERIOD) % 1;
                    burstActive = cycle < (1 / PULSE_PERIOD);
                    if (burstActive) {{
                        effectivePower = normIntensity * PULSE_BOOST;
                        if (modeSpan) modeSpan.innerHTML = ' <span style="background:#ff5500; padding:2px 5px; border-radius:4px; animation: blink 0.8s infinite;">脉冲 ON</span>';
                        // 脉冲开启瞬间，一次性生成一批气泡
                        if (!lastBurstState) {{
                            const burstCount = Math.floor(BUBBLE_COUNT * normIntensity * 0.4);
                            let spawned = 0;
                            for (let i = 0; i < BUBBLE_COUNT && spawned < burstCount; i++) {{
                                if (!bubbleData.active[i]) {{
                                    const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                                    resetBubble(i, pipe);
                                    spawned++;
                                }}
                            }}
                        }}
                    }} else {{
                        effectivePower = 0;
                        if (modeSpan) modeSpan.innerHTML = ' <span style="background:#333; padding:2px 5px; border-radius:4px;">脉冲 OFF</span>';
                        // 关闭阶段：不生成新气泡
                    }}
                    lastBurstState = burstActive;
                }}
                
                // 更新所有气泡位置（无论脉冲状态，上升速度正常）
                for (let i = 0; i < BUBBLE_COUNT; i++) {{
                    if (!bubbleData.active[i]) continue;
                    bubbleData.posY[i] += bubbleData.vel[i] * (1 + normIntensity * 1.2) * dt * 60;
                    if (bubbleData.posY[i] > halfLen + 0.3) {{
                        bubbleData.active[i] = 0;
                        dummyMat.scale.setScalar(0);
                        dummyMat.updateMatrix();
                        bubbles.setMatrixAt(i, dummyMat.matrix);
                        continue;
                    }}
                    const scale = H_SIZE * BUBBLE_BASE_SCALE * bubbleData.size[i] * (0.8 + Math.sin(time*12 + i)*0.2);
                    dummyMat.position.set(bubbleData.posX[i], bubbleData.posY[i], bubbleData.posZ[i]);
                    dummyMat.scale.setScalar(scale);
                    dummyMat.updateMatrix();
                    bubbles.setMatrixAt(i, dummyMat.matrix);
                }}
                bubbles.instanceMatrix.needsUpdate = true;
                
                // 膜丝摆动
                const pwr = Math.min(1.2, Math.max(0, effectivePower));
                for (const fiber of fibers) {{
                    const baseX = fiber.userData.baseX;
                    const baseZ = fiber.userData.baseZ;
                    let minDist = 100;
                    for (const pp of pipePositions) minDist = Math.min(minDist, Math.abs(baseZ - pp.z));
                    const effect = Math.exp(-minDist * EFFECT_DECAY_RATE);
                    const amp = pwr * F_LEN * SLACK * (0.15 + effect * 1.2);
                    const limitedAmp = Math.min(MAX_AMP, amp);
                    const phase = time * (WAVE_FREQ_BASE + pwr * WAVE_FREQ_INTENSITY_FACTOR) + fiber.userData.offset;
                    const shift = Math.sin(phase) * limitedAmp;
                    fiber.position.x = baseX + shift;
                    fiber.position.y = Math.sin(phase * 1.5) * limitedAmp * 0.2;
                }}
                
                // 污泥层更新
                sludgeLayer.scale.y = Math.max(SLUDGE_LEVEL, 0.01) / 0.05;
                sludgeLayer.position.y = bottomY + sludgeLayer.scale.y * 0.05 / 2;
                
                controls.update();
                renderer.render(scene, camera);
                requestAnimationFrame(animate);
            }}
            
            animate();
            
            // 全屏和复位
            document.getElementById('fullscreen').addEventListener('click', () => {{
                if (!document.fullscreenElement) document.documentElement.requestFullscreen();
                else document.exitFullscreen();
            }});
            document.getElementById('reset').addEventListener('click', () => {{
                camera.position.set(5.5, 4.0, 7.0);
                controls.target.set(0, 0, 0);
                controls.update();
            }});
            
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
            
            // 添加 CSS 动画
            const style = document.createElement('style');
            style.textContent = `@keyframes blink {{ 0%{{opacity:0.3;}}50%{{opacity:1;}}100%{{opacity:0.3;}}}}`;
            document.head.appendChild(style);
        </script>
    </body>
    </html>
    """
    return html


# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="MBR 工业仿真 v14.4", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = MBRSimulator()
sim = st.session_state.sim

# 侧边栏控件
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
    st.subheader("🧫 污泥参数")
    sim.mlss = st.slider("MLSS (mg/L)", 2000, 15000, int(sim.mlss), step=500)
    sim.srt = st.slider("污泥龄 (d)", 5, 40, int(sim.srt), step=1)
    sim.settling_rate = st.slider("沉降速率 (m/h)", 0.5, 6.0, float(sim.settling_rate), step=0.5)
    sim.return_ratio = st.slider("回流比 (%)", 50, 300, int(sim.return_ratio), step=10)
    if st.button("⬇️ 排泥", use_container_width=True):
        sim.discharge_sludge()
        st.rerun()
    if st.button("⏱️ 模拟1小时", use_container_width=True):
        sim.update_sludge_level(3600)
        st.rerun()
    sludge_percent = sim.sludge_level / 0.6 * 100
    st.progress(min(100, int(sludge_percent)), text=f"污泥层 {sim.sludge_level*1000:.0f} mm")

# 主区域
st.title("💧 MBR 工业仿真系统 v14.4")
st.caption("圆柱体膜丝 | 动态光照 | 脉冲/连续曝气 | 鼠标拖拽旋转/缩放")

metrics = sim.get_metrics()
cols = st.columns(4)
cols[0].metric("能耗 SEC", f"{metrics['sec']} kWh/m³")
cols[1].metric("平均剪切力", f"{metrics['shear_avg']} Pa")
cols[2].metric("最大剪切力", f"{metrics['shear_max']} Pa")
cols[3].metric("覆盖均匀度", f"{metrics['uniformity']} %")
cols2 = st.columns(4)
cols2[0].metric("积垢风险", metrics['risk_text'])
cols2[1].metric("SVI", f"{metrics['svi']} mL/g")
cols2[2].metric("TSS", f"{metrics['tss']} mg/L")
cols2[3].metric("总膜面积", f"{metrics['total_area']} m²")

st.markdown("### 🖥️ 3D 可视化视图")
html_code = generate_3d_html(sim)
st.components.v1.html(html_code, height=650, scrolling=False)

# 趋势图
with st.expander("📈 12小时趋势预测"):
    times = np.linspace(0, 12, 50)
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
    for _ in times:
        avg, _ = temp.calculate_shear_stress()
        shear_vals.append(avg)
        sludge_vals.append(temp.sludge_level * 1000)
        temp.update_sludge_level(3600 * 0.24)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=times, y=shear_vals, mode='lines+markers', name='剪切力 (Pa)', line=dict(color='#00f2ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=sludge_vals, mode='lines', name='污泥层 (mm)', line=dict(color='#d4a84b')), row=2, col=1)
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
