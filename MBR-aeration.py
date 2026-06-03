import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ==================== 与之前相同的 MBRSimulator 类 ====================
# （为了节省篇幅，此处省略，您可直接复制之前代码中的 MBRSimulator 类及 CONSTANTS、PRESETS）
# 请确保完整包含该类。若需完整代码，请告知，我会提供全量。

# ==================== 优化后的 Three.js HTML 生成器 ====================
def generate_3d_html_optimized(sim) -> str:
    """生成高清、高对比度的 3D 场景 HTML"""
    # 参数计算
    sheet_area = sim.get_sheet_area()
    diameter_m = sim.fiber_diameter / 1000
    area_per_fiber = np.pi * diameter_m * sim.f_len
    real_fibers = max(1, int(sheet_area / area_per_fiber))
    visual_fibers = min(real_fibers, 180)  # 降低数量以保证性能，但使用圆柱体后更清晰

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
        "fiberDiameter": sim.fiber_diameter,
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
        </style>
    </head>
    <body>
        <div class="info-panel">
            MBR 3D | 膜丝圆柱体模型 | 污泥层 {sim.sludge_level*1000:.0f} mm | 曝气强度 {sim.intensity}
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
            
            // 常量
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
            const VISUAL_FIBERS = CONFIG.visualFibersPerSheet;
            
            // 增强视觉参数
            const FIBER_RADIUS = 0.008;       // 圆柱体半径 (8mm 直径，清晰可见)
            const FIBER_SEGMENTS = 16;         // 圆柱体分段数，更圆滑
            const BUBBLE_COUNT = 4000;
            const SLUDGE_PARTICLE_COUNT = 2500;
            
            // 物理模拟参数（与之前类似，但减少粒子数以提升性能）
            const BUBBLE_MIN_VEL = 0.015, BUBBLE_MAX_VEL = 0.025;
            const BUBBLE_BASE_SCALE = 0.008;
            const WAVE_FREQ_BASE = 1.8, WAVE_FREQ_INTENSITY_FACTOR = 6.0;
            const MAX_BASE_AMPLITUDE = 0.045;
            const PULSE_POWER_BOOST = 1.4, OFF_PHASE_POWER = 0.05;
            
            // 初始化场景、相机、渲染器（开启最高抗锯齿）
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
            
            // 后期特效（Bloom 增强高光）
            const renderScene = new RenderPass(scene, camera);
            const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.5, 0.3, 0.2);
            bloomPass.threshold = 0.1;
            bloomPass.strength = 0.6;
            bloomPass.radius = 0.5;
            const effectComposer = new EffectComposer(renderer);
            effectComposer.addPass(renderScene);
            effectComposer.addPass(bloomPass);
            
            // 轨道控制（带自动旋转可选）
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.07;
            controls.rotateSpeed = 1.0;
            controls.zoomSpeed = 1.2;
            controls.enableZoom = true;
            controls.autoRotate = false;
            controls.target.set(0, 0.2, 0);
            
            // 增强照明系统
            // 环境光
            const ambientLight = new THREE.AmbientLight(0x4466aa, 0.55);
            scene.add(ambientLight);
            // 主光源方向光
            const mainLight = new THREE.DirectionalLight(0xffffff, 1.5);
            mainLight.position.set(5, 10, 4);
            mainLight.castShadow = true;
            mainLight.receiveShadow = false;
            scene.add(mainLight);
            // 补光从底部
            const fillLight = new THREE.PointLight(0x88aaff, 0.6);
            fillLight.position.set(0, -3, 0);
            scene.add(fillLight);
            // 背光暖色
            const backLight = new THREE.PointLight(0xffaa66, 0.5);
            backLight.position.set(-2, 2, -5);
            scene.add(backLight);
            // 冷色侧光
            const rimLight = new THREE.PointLight(0x00ccff, 0.8);
            rimLight.position.set(3, 1.5, -4);
            scene.add(rimLight);
            
            // 辅助网格地板（半透明，增强空间感）
            const gridHelper = new THREE.GridHelper(12, 20, 0x33aaff, 0x2266aa);
            gridHelper.position.y = -F_LEN/2 - 0.6;
            gridHelper.material.transparent = true;
            gridHelper.material.opacity = 0.2;
            scene.add(gridHelper);
            
            // ---------- 创建膜组件（使用圆柱体表示膜丝）----------
            const sheetGroup = new THREE.Group();
            const halfLen = F_LEN/2;
            const halfWidth = SHEET_WIDTH/2;
            const fiberMaterial = new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.85, roughness: 0.25, emissive: 0x222222, emissiveIntensity: 0.15 }});
            
            // 存储所有膜丝对象以便动画
            const fiberMeshes = [];
            
            for (let i = 0; i < SHEET_COUNT; i++) {{
                const zPos = (i - 2) * S_PITCH_M;
                const sheet = new THREE.Group();
                
                // 上下集水槽（金属质感）
                const headerMat = new THREE.MeshStandardMaterial({{ color: 0xE0C8A0, metalness: 0.6, roughness: 0.4 }});
                const headerGeo = new THREE.BoxGeometry(SHEET_WIDTH, 0.12, 0.04);
                const topHeader = new THREE.Mesh(headerGeo, headerMat);
                const bottomHeader = new THREE.Mesh(headerGeo, headerMat);
                topHeader.position.y = halfLen - 0.04;
                bottomHeader.position.y = -halfLen + 0.04;
                sheet.add(topHeader, bottomHeader);
                
                // 两侧导轨（不锈钢）
                const railMat = new THREE.MeshStandardMaterial({{ color: 0xCCCCDD, metalness: 0.9, roughness: 0.2 }});
                const railGeo = new THREE.BoxGeometry(0.03, F_LEN, 0.03);
                const leftRail = new THREE.Mesh(railGeo, railMat);
                const rightRail = new THREE.Mesh(railGeo, railMat);
                leftRail.position.set(-halfWidth + SHEET_END_MARGIN, 0, 0);
                rightRail.position.set(halfWidth - SHEET_END_MARGIN, 0, 0);
                sheet.add(leftRail, rightRail);
                
                // 膜丝（圆柱体）
                for (let f = 0; f < VISUAL_FIBERS; f++) {{
                    const x = VISUAL_FIBERS > 1 ? (f / (VISUAL_FIBERS-1)) * SHEET_WIDTH - halfWidth : 0;
                    const cylinderGeo = new THREE.CylinderGeometry(FIBER_RADIUS, FIBER_RADIUS, F_LEN, FIBER_SEGMENTS);
                    const fiber = new THREE.Mesh(cylinderGeo, fiberMaterial);
                    fiber.position.set(x, 0, 0);
                    fiber.castShadow = true;
                    fiber.receiveShadow = false;
                    fiber.userData = {{
                        baseX: x,
                        baseZ: zPos,
                        offset: Math.random() * Math.PI * 2,
                        segments: 1  // 圆柱体整体摆动，不需要分段几何变形
                    }};
                    sheet.add(fiber);
                    fiberMeshes.push(fiber);
                }}
                sheet.position.z = zPos;
                sheetGroup.add(sheet);
            }}
            scene.add(sheetGroup);
            
            // ---------- 曝气管道（与之前类似，增加材质高光）----------
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
                // 连接竖管
                const connectorHeight = Math.abs(pos.y - mainY);
                const connector = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, connectorHeight, 8), pipeMatShiny);
                connector.position.set(0, (pos.y + mainY)/2, pos.z);
                pipeGroup.add(connector);
            }}
            scene.add(pipeGroup);
            
            // ---------- 污泥层（带纹理感）----------
            const sludgeLayerMat = new THREE.MeshStandardMaterial({{ color: 0x7a4a2a, roughness: 0.8, metalness: 0.1, emissive: 0x331100, emissiveIntensity: 0.2 }});
            const sludgeLayerWidth = SHEET_WIDTH + 0.3;
            const depth = S_PITCH_M * SHEET_COUNT + 0.5;
            const sludgeLayer = new THREE.Mesh(new THREE.BoxGeometry(sludgeLayerWidth, 0.05, depth), sludgeLayerMat);
            const bottomY = -halfLen - 0.5;
            sludgeLayer.position.y = bottomY + SLUDGE_LEVEL/2;
            sludgeLayer.scale.y = SLUDGE_LEVEL / 0.05;
            scene.add(sludgeLayer);
            
            // ---------- 气泡粒子系统（InstancedMesh，性能优化）----------
            const bubbleMat = new THREE.MeshStandardMaterial({{ color: 0x88ccff, emissive: 0x2288aa, emissiveIntensity: 0.3, transparent: true, opacity: 0.7 }});
            const bubbles = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10), bubbleMat, BUBBLE_COUNT);
            bubbles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            scene.add(bubbles);
            
            // 气泡数据
            const bubbleData = {{
                posX: new Float32Array(BUBBLE_COUNT), posY: new Float32Array(BUBBLE_COUNT), posZ: new Float32Array(BUBBLE_COUNT),
                vel: new Float32Array(BUBBLE_COUNT), phase: new Float32Array(BUBBLE_COUNT),
                size: new Float32Array(BUBBLE_COUNT), active: new Uint8Array(BUBBLE_COUNT)
            }};
            function resetBubble(i) {{
                const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                bubbleData.posX[i] = (Math.random() - 0.5) * (SHEET_WIDTH - 0.2);
                bubbleData.posY[i] = -halfLen - PIPE_OFFSET + 0.04;
                bubbleData.posZ[i] = pipe.z + (Math.random()-0.5)*0.1;
                bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random()*(BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.size[i] = 0.5 + Math.random()*0.8;
                bubbleData.active[i] = 1;
            }}
            for (let i=0; i<BUBBLE_COUNT; i++) resetBubble(i);
            
            // 动画循环
            let lastTime = performance.now();
            let burstActive = true;
            let spawnAccum = 0;
            const dummyMat = new THREE.Object3D();
            
            function animate() {{
                const now = performance.now();
                let dt = Math.min((now - lastTime)/1000, 0.05);
                lastTime = now;
                const time = now / 1000;
                
                // 脉冲逻辑
                if (MODE === 'pulse') {{
                    const cycle = (time / PULSE_PERIOD) % 1;
                    burstActive = cycle < (1 / PULSE_PERIOD);
                }} else burstActive = true;
                const intensityNorm = (INTENSITY - 50) / 100;
                const pwr = burstActive ? (MODE==='pulse' ? intensityNorm*PULSE_POWER_BOOST : intensityNorm) : OFF_PHASE_POWER;
                
                // 更新气泡
                const frameScale = dt * 60;
                const baseScale = H_SIZE * 0.007;
                const targetActive = Math.min(BUBBLE_COUNT, Math.floor(BUBBLE_COUNT * intensityNorm * 0.8));
                const spawnRate = burstActive ? targetActive * 1.5 * dt : 0;
                spawnAccum += spawnRate;
                let activeCount = 0, spawns = 0;
                for (let i=0; i<BUBBLE_COUNT; i++) {{
                    if (!bubbleData.active[i] && activeCount < targetActive && spawnAccum>=1 && spawns<8) {{
                        resetBubble(i);
                        spawnAccum -= 1;
                        spawns++;
                        activeCount++;
                        continue;
                    }}
                    if (!bubbleData.active[i]) continue;
                    activeCount++;
                    bubbleData.posY[i] += bubbleData.vel[i] * (1 + intensityNorm*1.2) * frameScale;
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
                
                // 膜丝摆动（圆柱体整体位移+轻微弯曲效果通过位置偏移模拟）
                for (const fiber of fiberMeshes) {{
                    const baseX = fiber.userData.baseX;
                    const baseZ = fiber.userData.baseZ;
                    let minDist = Infinity;
                    for (const pp of pipePositions) minDist = Math.min(minDist, Math.abs(baseZ - pp.z));
                    const effect = Math.exp(-minDist * EFFECT_DECAY_RATE);
                    const env = 1.0; // 整根纤维整体摆动
                    const amplitude = Math.min(MAX_BASE_AMPLITUDE, F_LEN * SLACK * pwr * (0.15 + effect*1.2));
                    const phase = time * (WAVE_FREQ_BASE + pwr * WAVE_FREQ_INTENSITY_FACTOR) + fiber.userData.offset;
                    const shift = Math.sin(phase) * amplitude;
                    fiber.position.x = baseX + shift;
                    // 轻微垂直偏移增加真实感
                    const vertShift = Math.sin(phase * 1.7) * amplitude * 0.3;
                    fiber.position.y = vertShift;
                }}
                
                // 污泥层动态更新（从后端参数获取，这里仅视觉更新）
                sludgeLayer.scale.y = Math.max(SLUDGE_LEVEL, 0.01) / 0.05;
                sludgeLayer.position.y = bottomY + sludgeLayer.scale.y * 0.05 / 2;
                
                controls.update();
                effectComposer.render();
                requestAnimationFrame(animate);
            }}
            
            animate();
            
            // 全屏和复位功能
            document.getElementById('fullscreen-btn').addEventListener('click', () => {{
                if (!document.fullscreenElement) {{
                    document.documentElement.requestFullscreen();
                }} else {{
                    document.exitFullscreen();
                }}
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

# 侧边栏控制（与之前相同，省略重复代码，请复制之前完整侧边栏部分）
# 注意：您需要将之前代码中的整个侧边栏 with st.sidebar 内容粘贴至此
# 为避免篇幅过长，此处使用占位注释，实际运行时请替换为完整侧边栏代码
# ...

# 主区域
st.title("💧 MBR 工业仿真系统 v14.4 - 高清3D引擎")
st.caption("圆柱体膜丝 | 动态光照 | 辉光特效 | 全屏支持")

# 指标卡片
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

st.markdown("### 🖥️ 交互式3D视图（鼠标拖拽旋转/右键平移/滚轮缩放）")
# 生成并嵌入优化后的 HTML
html_code = generate_3d_html_optimized(sim)
st.components.v1.html(html_code, height=650, scrolling=False)

# 趋势图
with st.expander("📈 12小时趋势预测"):
    # 与之前相同的趋势图代码...
    pass
