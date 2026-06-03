import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

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
    """MBR工艺仿真核心逻辑"""
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
    
    def calculate_shear_stress(self) -> tuple:
        intensity_factor = self.get_effective_intensity_factor(True)
        slack_factor = 1.0 + self.slack * 12.0
        orifice_factor = (4.0 / max(1.5, self.h_size)) ** 0.4
        pipe_factor = np.exp(-self.p_pitch / 180.0)
        sheet_factor = np.exp(-self.s_pitch / 100.0)
        base_shear = 0.3 * intensity_factor * slack_factor * orifice_factor * (pipe_factor+0.3) * (sheet_factor+0.5)
        avg_shear = np.clip(base_shear, 0.1, 3.5)
        if self.mode == "pulse":
            max_mult = 1.8
        else:
            max_mult = 1.2
        max_shear = np.clip(avg_shear * max_mult * (1.0 + self.slack * 3.0), 0.2, 6.0)
        return round(avg_shear, 3), round(max_shear, 3)
    
    def calculate_sec(self) -> float:
        norm_intensity = (self.intensity - 50.0) / 100.0
        base_sec = 0.12 + norm_intensity * 0.35
        if self.mode == "pulse":
            duty = min(1.0 / self.pulse_period, 0.5)
            base_sec *= (duty * 1.2 + (1-duty)*0.3)
        return round(base_sec, 3)
    
    def calculate_uniformity(self) -> float:
        diff = abs(self.p_pitch - self.s_pitch)
        return round(max(0.0, 100.0 - diff * CONSTANTS["UNIF_PENALTY_FACTOR"]), 1)
    
    def calculate_risk_level(self, max_shear: float):
        if max_shear < 0.8:
            return "HIGH", "risk-high"
        elif max_shear < 1.8:
            return "MEDIUM", "risk-medium"
        else:
            return "LOW", "risk-low"
    
    def calculate_svi(self) -> float:
        svi = 200.0 - self.srt * 3.0 + (self.mlss / 10000.0) * 50.0
        return round(np.clip(svi, 50.0, 280.0), 1)
    
    def calculate_tss(self, avg_shear: float) -> float:
        base_tss = 5.0 + (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 25.0
        return round(np.clip(base_tss + avg_shear * 1.5 * 2, 4.0, 35.0), 1)
    
    def update_sludge_level(self, dt: float = 1.0):
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
            "fiber_count": self.calculate_fiber_count()[0] * CONSTANTS["SHEET_COUNT"],
            "sludge_level_mm": self.sludge_level * 1000.0,
            "sludge_percent": (self.sludge_level / CONSTANTS["SLUDGE_LAYER_MAX_HEIGHT"]) * 100.0
        }


def generate_3d_html_optimized(sim) -> str:
    """生成3D场景HTML，脉冲与连续模式视觉差异显著"""
    sheet_area = sim.get_sheet_area()
    diameter_m = sim.fiber_diameter / 1000
    area_per_fiber = np.pi * diameter_m * sim.f_len
    real_fibers = max(1, int(sheet_area / area_per_fiber))
    visual_fibers = min(real_fibers, 180)

    config = {
        "sheetCount": 5,
        "sheetWidth": 1.25,
        "sheetEndMargin": 0.05,
        "pipeOffset": 0.45,
        "effectDecayRate": 14,
        "sludgeLevel": sim.sludge_level,
        "sludgeMaxHeight": 0.6,
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
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <style>
            body {{ margin: 0; overflow: hidden; font-family: 'Segoe UI', 'Roboto', sans-serif; }}
            #controls-ui {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                z-index: 100;
                background: rgba(0,0,0,0.7);
                backdrop-filter: blur(8px);
                padding: 8px 12px;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                display: flex;
                gap: 10px;
                pointer-events: auto;
            }}
            button {{
                background: #00f2ff22;
                border: 1px solid #00f2ff;
                color: #00f2ff;
                border-radius: 4px;
                padding: 4px 8px;
                cursor: pointer;
                font-size: 12px;
                transition: 0.2s;
            }}
            button:hover {{
                background: #00f2ff66;
                color: white;
            }}
            .info-panel {{
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.6);
                backdrop-filter: blur(5px);
                padding: 10px 15px;
                border-radius: 8px;
                color: #ccddff;
                font-size: 12px;
                font-family: monospace;
                pointer-events: none;
                z-index: 100;
                border-left: 3px solid #00f2ff;
            }}
            .pulse-badge {{
                display: inline-block;
                background: #ff5500;
                color: white;
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 4px;
                margin-left: 8px;
                animation: blink 0.8s infinite;
            }}
            @keyframes blink {{
                0% {{ opacity: 0.3; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.3; }}
            }}
        </style>
    </head>
    <body>
        <div class="info-panel" id="info-panel">
            MBR 3D | 圆柱体膜丝 | 污泥层 {sim.sludge_level*1000:.0f} mm | 曝气强度 {sim.intensity}
            <span id="mode-indicator"></span>
        </div>
        <div id="controls-ui">
            <button id="fullscreen-btn">⛶ 全屏</button>
            <button id="reset-cam">🎥 复位视角</button>
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
            import {{ EffectComposer }} from 'three/addons/postprocessing/EffectComposer.js';
            import {{ RenderPass }} from 'three/addons/postprocessing/RenderPass.js';
            import {{ UnrealBloomPass }} from 'three/addons/postprocessing/UnrealBloomPass.js';
            
            const CONFIG = {json.dumps(config)};
            
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
            const FIBER_SEGMENTS = 16;
            const BUBBLE_COUNT = 4000;
            
            const BUBBLE_MIN_VEL = 0.015, BUBBLE_MAX_VEL = 0.025;
            const BUBBLE_BASE_SCALE = 0.008;
            const WAVE_FREQ_BASE = 1.8, WAVE_FREQ_INTENSITY_FACTOR = 6.0;
            const MAX_BASE_AMPLITUDE = 0.045;
            const PULSE_POWER_BOOST = 1.4;
            const OFF_PHASE_POWER = 0.0;   // 脉冲关闭时完全无动力，膜丝静止，气泡不产生
            const OFF_PHASE_BUBBLE_FACTOR = 0.02; // 残留气泡快速消散
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x01050a);
            scene.fog = new THREE.FogExp2(0x01050a, 0.015);
            
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(5.5, 4.2, 7.5);
            camera.lookAt(0, 0, 0);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.toneMapping = THREE.ReinhardToneMapping;
            renderer.toneMappingExposure = 1.2;
            document.body.appendChild(renderer.domElement);
            
            const renderScene = new RenderPass(scene, camera);
            const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.5, 0.3, 0.2);
            bloomPass.threshold = 0.1;
            bloomPass.strength = 0.6;
            bloomPass.radius = 0.5;
            const effectComposer = new EffectComposer(renderer);
            effectComposer.addPass(renderScene);
            effectComposer.addPass(bloomPass);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.07;
            controls.rotateSpeed = 1.0;
            controls.zoomSpeed = 1.2;
            controls.target.set(0, 0.2, 0);
            
            // 光照
            scene.add(new THREE.AmbientLight(0x4466aa, 0.55));
            const mainLight = new THREE.DirectionalLight(0xffffff, 1.5);
            mainLight.position.set(5, 10, 4);
            scene.add(mainLight);
            scene.add(new THREE.PointLight(0x88aaff, 0.6).position.set(0, -3, 0));
            scene.add(new THREE.PointLight(0xffaa66, 0.5).position.set(-2, 2, -5));
            scene.add(new THREE.PointLight(0x00ccff, 0.8).position.set(3, 1.5, -4));
            
            const gridHelper = new THREE.GridHelper(12, 20, 0x33aaff, 0x2266aa);
            gridHelper.position.y = -F_LEN/2 - 0.6;
            gridHelper.material.transparent = true;
            gridHelper.material.opacity = 0.2;
            scene.add(gridHelper);
            
            // 膜组件
            const sheetGroup = new THREE.Group();
            const halfLen = F_LEN/2;
            const halfWidth = SHEET_WIDTH/2;
            const fiberMaterial = new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.85, roughness: 0.25, emissive: 0x222222, emissiveIntensity: 0.15 }});
            const fiberMeshes = [];
            for (let i = 0; i < SHEET_COUNT; i++) {{
                const zPos = (i - 2) * S_PITCH_M;
                const sheet = new THREE.Group();
                const headerMat = new THREE.MeshStandardMaterial({{ color: 0xE0C8A0, metalness: 0.6, roughness: 0.4 }});
                const headerGeo = new THREE.BoxGeometry(SHEET_WIDTH, 0.12, 0.04);
                const topHeader = new THREE.Mesh(headerGeo, headerMat);
                const bottomHeader = new THREE.Mesh(headerGeo, headerMat);
                topHeader.position.y = halfLen - 0.04;
                bottomHeader.position.y = -halfLen + 0.04;
                sheet.add(topHeader, bottomHeader);
                const railMat = new THREE.MeshStandardMaterial({{ color: 0xCCCCDD, metalness: 0.9, roughness: 0.2 }});
                const railGeo = new THREE.BoxGeometry(0.03, F_LEN, 0.03);
                const leftRail = new THREE.Mesh(railGeo, railMat);
                const rightRail = new THREE.Mesh(railGeo, railMat);
                leftRail.position.set(-halfWidth + SHEET_END_MARGIN, 0, 0);
                rightRail.position.set(halfWidth - SHEET_END_MARGIN, 0, 0);
                sheet.add(leftRail, rightRail);
                for (let f = 0; f < VISUAL_FIBERS; f++) {{
                    const x = VISUAL_FIBERS > 1 ? (f / (VISUAL_FIBERS-1)) * SHEET_WIDTH - halfWidth : 0;
                    const cylinderGeo = new THREE.CylinderGeometry(FIBER_RADIUS, FIBER_RADIUS, F_LEN, FIBER_SEGMENTS);
                    const fiber = new THREE.Mesh(cylinderGeo, fiberMaterial);
                    fiber.position.set(x, 0, 0);
                    fiber.castShadow = true;
                    fiber.userData = {{ baseX: x, baseZ: zPos, offset: Math.random() * Math.PI * 2 }};
                    sheet.add(fiber);
                    fiberMeshes.push(fiber);
                }}
                sheet.position.z = zPos;
                sheetGroup.add(sheet);
            }}
            scene.add(sheetGroup);
            
            // 曝气管道
            const pipeGroup = new THREE.Group();
            const pipeMatShiny = new THREE.MeshStandardMaterial({{ color: 0x557788, metalness: 0.8, roughness: 0.3 }});
            const mainY = -halfLen - PIPE_OFFSET - 0.15;
            const mainPipeGeo = new THREE.CylinderGeometry(0.032, 0.032, S_PITCH_M*SHEET_COUNT + 0.5, 16);
            const mainPipe = new THREE.Mesh(mainPipeGeo, pipeMatShiny);
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
            const pipeLen = SHEET_WIDTH + 0.3;
            const pipeGeo = new THREE.CylinderGeometry(0.028, 0.028, pipeLen, 12);
            const orificeRadius = 0.018 * (H_SIZE / 4);
            const orificeMatGlow = new THREE.MeshStandardMaterial({{ color: 0x00ccff, emissive: 0x0088aa, emissiveIntensity: 0.4 }});
            const orificeGeo = new THREE.SphereGeometry(orificeRadius, 8, 8);
            for (const pos of pipePositions) {{
                const pipe = new THREE.Mesh(pipeGeo, pipeMatShiny);
                pipe.rotation.z = Math.PI/2;
                pipe.position.set(pos.x, -halfLen - PIPE_OFFSET, pos.z);
                for (let k = 0; k < 8; k++) {{
                    const orifice = new THREE.Mesh(orificeGeo, orificeMatGlow);
                    orifice.position.set(0.032, k/7*pipeLen - pipeLen/2, 0);
                    pipe.add(orifice);
                }}
                pipeGroup.add(pipe);
                const connectorHeight = Math.abs(pos.y - mainY);
                const connector = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, connectorHeight, 8), pipeMatShiny);
                connector.position.set(0, (pos.y + mainY)/2, pos.z);
                pipeGroup.add(connector);
            }}
            scene.add(pipeGroup);
            
            // 污泥层
            const sludgeLayerMat = new THREE.MeshStandardMaterial({{ color: 0x7a4a2a, roughness: 0.8, metalness: 0.1, emissive: 0x331100, emissiveIntensity: 0.2 }});
            const sludgeLayerWidth = SHEET_WIDTH + 0.3;
            const depth = S_PITCH_M * SHEET_COUNT + 0.5;
            const sludgeLayer = new THREE.Mesh(new THREE.BoxGeometry(sludgeLayerWidth, 0.05, depth), sludgeLayerMat);
            const bottomY = -halfLen - 0.5;
            sludgeLayer.position.y = bottomY + SLUDGE_LEVEL/2;
            sludgeLayer.scale.y = SLUDGE_LEVEL / 0.05;
            scene.add(sludgeLayer);
            
            // 气泡系统
            const bubbleMat = new THREE.MeshStandardMaterial({{ color: 0x88ccff, emissive: 0x2288aa, emissiveIntensity: 0.3, transparent: true, opacity: 0.7 }});
            const bubbles = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10), bubbleMat, BUBBLE_COUNT);
            bubbles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            scene.add(bubbles);
            
            const bubbleData = {{
                posX: new Float32Array(BUBBLE_COUNT), posY: new Float32Array(BUBBLE_COUNT), posZ: new Float32Array(BUBBLE_COUNT),
                vel: new Float32Array(BUBBLE_COUNT), size: new Float32Array(BUBBLE_COUNT), active: new Uint8Array(BUBBLE_COUNT)
            }};
            function resetBubble(i, pipe) {{
                bubbleData.posX[i] = (Math.random() - 0.5) * (SHEET_WIDTH - 0.2);
                bubbleData.posY[i] = -halfLen - PIPE_OFFSET + 0.04;
                bubbleData.posZ[i] = pipe.z + (Math.random()-0.5)*0.1;
                bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random()*(BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.size[i] = 0.5 + Math.random()*0.8;
                bubbleData.active[i] = 1;
            }}
            for (let i=0; i<BUBBLE_COUNT; i++) resetBubble(i, pipePositions[0]);
            
            // 动画
            let lastTime = performance.now();
            let burstActive = true;
            let spawnAccum = 0;
            const dummyMat = new THREE.Object3D();
            const modeIndicator = document.getElementById('mode-indicator');
            
            function animate() {{
                const now = performance.now();
                let dt = Math.min((now - lastTime)/1000, 0.05);
                lastTime = now;
                const time = now / 1000;
                
                // 脉冲模式逻辑
                let effectivePower = 0;
                if (MODE === 'pulse') {{
                    const cycle = (time / PULSE_PERIOD) % 1;
                    burstActive = cycle < (1 / PULSE_PERIOD);
                    // 脉冲开启时功率倍增，关闭时功率为0
                    const normIntensity = (INTENSITY - 50) / 100;
                    if (burstActive) {{
                        effectivePower = normIntensity * PULSE_POWER_BOOST;
                    }} else {{
                        effectivePower = 0;
                    }}
                    // UI指示器
                    if (burstActive) {{
                        modeIndicator.innerHTML = '<span class="pulse-badge">⚡ 脉冲 ON</span>';
                    }} else {{
                        modeIndicator.innerHTML = '<span class="pulse-badge" style="background:#333;">⏸ 脉冲 OFF</span>';
                    }}
                }} else {{
                    burstActive = true;
                    const normIntensity = (INTENSITY - 50) / 100;
                    effectivePower = normIntensity;
                    modeIndicator.innerHTML = '<span style="background:#00aaff; padding:2px 6px; border-radius:4px;">🔘 连续曝气</span>';
                }}
                
                // 膜丝摆动：功率为0时完全静止
                const pwr = Math.min(1.2, Math.max(0, effectivePower));
                for (const fiber of fiberMeshes) {{
                    const baseX = fiber.userData.baseX;
                    const baseZ = fiber.userData.baseZ;
                    let minDist = Infinity;
                    for (const pp of pipePositions) minDist = Math.min(minDist, Math.abs(baseZ - pp.z));
                    const effect = Math.exp(-minDist * EFFECT_DECAY_RATE);
                    const amplitude = pwr * F_LEN * SLACK * (0.15 + effect*1.2);
                    const limitedAmp = Math.min(MAX_BASE_AMPLITUDE, amplitude);
                    const phase = time * (WAVE_FREQ_BASE + pwr * WAVE_FREQ_INTENSITY_FACTOR) + fiber.userData.offset;
                    const shift = Math.sin(phase) * limitedAmp;
                    fiber.position.x = baseX + shift;
                    const vertShift = Math.sin(phase * 1.7) * limitedAmp * 0.3;
                    fiber.position.y = vertShift;
                }}
                
                // 气泡更新
                const frameScale = dt * 60;
                const baseScale = H_SIZE * 0.007;
                const targetActive = burstActive ? Math.floor(BUBBLE_COUNT * ((INTENSITY-50)/100) * 0.8) : 0;
                let spawnRate = burstActive ? targetActive * 1.5 * dt : 0;
                spawnAccum += spawnRate;
                let activeCount = 0, spawns = 0;
                for (let i=0; i<BUBBLE_COUNT; i++) {{
                    if (!bubbleData.active[i] && activeCount < targetActive && spawnAccum>=1 && spawns<8) {{
                        const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                        resetBubble(i, pipe);
                        spawnAccum -= 1;
                        spawns++;
                        activeCount++;
                        continue;
                    }}
                    if (!bubbleData.active[i]) continue;
                    activeCount++;
                    // 脉冲关闭时气泡上升速度减慢且快速消亡
                    let velocityFactor = burstActive ? (1 + ((INTENSITY-50)/100)*1.2) : OFF_PHASE_BUBBLE_FACTOR;
                    bubbleData.posY[i] += bubbleData.vel[i] * velocityFactor * frameScale;
                    if (bubbleData.posY[i] > halfLen + 0.4) {{
                        bubbleData.active[i] = 0;
                        dummyMat.scale.setScalar(0);
                        dummyMat.updateMatrix();
                        bubbles.setMatrixAt(i, dummyMat.matrix);
                        continue;
                    }}
                    const scl = baseScale * bubbleData.size[i] * (0.8 + Math.sin(time*10 + i)*0.2);
                    dummyMat.position.set(bubbleData.posX[i], bubbleData.posY[i], bubbleData.posZ[i]);
                    dummyMat.scale.setScalar(scl);
                    dummyMat.updateMatrix();
                    bubbles.setMatrixAt(i, dummyMat.matrix);
                }}
                bubbles.instanceMatrix.needsUpdate = true;
                
                // 污泥层更新（从参数读取）
                sludgeLayer.scale.y = Math.max(SLUDGE_LEVEL, 0.01) / 0.05;
                sludgeLayer.position.y = bottomY + sludgeLayer.scale.y * 0.05 / 2;
                
                controls.update();
                effectComposer.render();
                requestAnimationFrame(animate);
            }}
            
            animate();
            
            document.getElementById('fullscreen-btn').addEventListener('click', () => {{
                if (!document.fullscreenElement) document.documentElement.requestFullscreen();
                else document.exitFullscreen();
            }});
            document.getElementById('reset-cam').addEventListener('click', () => {{
                camera.position.set(5.5, 4.2, 7.5);
                controls.target.set(0, 0.2, 0);
                controls.update();
            }});
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
                effectComposer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html


# ==================== Streamlit 应用主体 ====================
st.set_page_config(page_title="MBR v14.4 - 高清3D仿真", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = MBRSimulator()
sim = st.session_state.sim

with st.sidebar:
    st.markdown("## 🧪 MBR 系统控制")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌿 节能模式"): sim.apply_preset("eco"); st.rerun()
    with col2:
        if st.button("⚖️ 均衡模式"): sim.apply_preset("balanced"); st.rerun()
    with col3:
        if st.button("💨 高冲刷模式"): sim.apply_preset("flush"); st.rerun()
    st.markdown("---")
    st.markdown("### 🌊 曝气参数")
    sim.mode = st.selectbox("曝气模式", ["cont", "pulse"], format_func=lambda x: "连续曝气" if x=="cont" else "脉冲式曝气")
    sim.intensity = st.slider("曝气强度", 50, 150, int(sim.intensity), step=5)
    if sim.mode == "pulse":
        sim.pulse_period = st.slider("脉冲周期 (s)", 3.0, 6.0, float(sim.pulse_period), step=0.5)
    sim.h_size = st.slider("曝气孔径 (mm)", 1.0, 15.0, float(sim.h_size), step=0.5)
    sim.p_pitch = st.slider("曝气管间距 (mm)", 50, 300, int(sim.p_pitch), step=10)
    st.markdown("---")
    st.markdown("### 🧬 膜片参数")
    fd_opt = {1.65:"1.65 mm → 40 m²", 2.8:"2.8 mm → 25 m²"}
    sim.fiber_diameter = st.selectbox("膜丝外径", options=[1.65,2.8], format_func=lambda x: fd_opt[x])
    sim.thickness = st.slider("膜片厚度 (mm)", 10, 100, int(sim.thickness), step=5)
    sim.s_pitch = st.slider("膜片间距 (mm)", 50, 100, int(sim.s_pitch), step=5)
    sim.f_len = st.slider("膜丝长度 (m)", 0.1, 3.0, float(sim.f_len), step=0.1)
    sim.slack = st.slider("松弛度 (%)", 0.2, 5.0, float(sim.slack*100), step=0.2) / 100.0
    st.markdown("---")
    st.markdown("### 🧫 污泥参数")
    sim.mlss = st.slider("MLSS (mg/L)", 2000, 15000, int(sim.mlss), step=500)
    sim.srt = st.slider("污泥龄 (d)", 5, 40, int(sim.srt), step=1)
    sim.settling_rate = st.slider("沉降速率 (m/h)", 0.5, 6.0, float(sim.settling_rate), step=0.5)
    sim.return_ratio = st.slider("回流比 (%)", 50, 300, int(sim.return_ratio), step=10)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("⬇️ 排泥"): sim.discharge_sludge(); st.rerun()
    with col_d2:
        if st.button("⏱️ 模拟1h"): sim.update_sludge_level(3600); st.rerun()
    sludge_percent = sim.sludge_level / 0.6 * 100
    st.progress(min(100, int(sludge_percent)), text=f"污泥层 {sim.sludge_level*1000:.0f} mm")

st.title("💧 MBR 工业仿真系统 v14.4 - 高清3D引擎")
st.caption("圆柱体膜丝 | 动态光照 | 辉光特效 | 全屏支持 | 鼠标拖拽旋转/缩放")

metrics = sim.get_metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("能耗 SEC", f"{metrics['sec']} kWh/m³")
c2.metric("平均剪切力", f"{metrics['shear_avg']} Pa")
c3.metric("最大剪切力", f"{metrics['shear_max']} Pa")
c4.metric("覆盖均匀度", f"{metrics['uniformity']} %")
c5, c6, c7, c8 = st.columns(4)
c5.metric("积垢风险", metrics['risk_text'])
c6.metric("SVI", f"{metrics['svi']} mL/g")
c7.metric("TSS", f"{metrics['tss']} mg/L")
c8.metric("总膜面积", f"{metrics['total_area']} m²")

st.markdown("### 🖥️ 交互式3D视图")
html_code = generate_3d_html_optimized(sim)
st.components.v1.html(html_code, height=650, scrolling=False)

with st.expander("📈 12小时趋势预测 (剪切力 & 污泥层)"):
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
    for t in times:
        avg, _ = temp.calculate_shear_stress()
        shear_vals.append(avg)
        sludge_vals.append(temp.sludge_level*1000)
        temp.update_sludge_level(3600*0.24)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=times, y=shear_vals, mode='lines+markers', name='剪切力 (Pa)', line=dict(color='#00f2ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=sludge_vals, mode='lines', name='污泥层 (mm)', line=dict(color='#d4a84b')), row=2, col=1)
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
