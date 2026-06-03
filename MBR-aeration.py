def generate_3d_html_optimized(sim) -> str:
    """生成高清3D场景，修复连续曝气初始爆发和脉冲模式气泡停滞问题"""
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
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <style>
            body {{ margin: 0; overflow: hidden; font-family: 'Segoe UI', sans-serif; }}
            #controls-ui {{
                position: absolute; bottom: 20px; right: 20px; z-index: 100;
                background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
                padding: 8px 12px; border-radius: 8px; color: white; font-size: 12px;
                display: flex; gap: 10px; pointer-events: auto;
            }}
            button {{
                background: #00f2ff22; border: 1px solid #00f2ff; color: #00f2ff;
                border-radius: 4px; padding: 4px 8px; cursor: pointer; font-size: 12px;
            }}
            button:hover {{ background: #00f2ff66; color: white; }}
            .info-panel {{
                position: absolute; top: 20px; left: 20px;
                background: rgba(0,0,0,0.6); backdrop-filter: blur(5px);
                padding: 10px 15px; border-radius: 8px; color: #ccddff; font-size: 12px;
                font-family: monospace; pointer-events: none; z-index: 100;
                border-left: 3px solid #00f2ff;
            }}
            .pulse-badge {{ display: inline-block; background: #ff5500; color: white;
                font-weight: bold; padding: 2px 6px; border-radius: 4px; margin-left: 8px;
                animation: blink 0.8s infinite; }}
            @keyframes blink {{ 0% {{ opacity: 0.3; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.3; }} }}
        </style>
    </head>
    <body>
        <div class="info-panel" id="info-panel">
            MBR 3D | 污泥 {sim.sludge_level*1000:.0f} mm | 曝气 {sim.intensity}
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
            import {{ EffectComposer, RenderPass, UnrealBloomPass }} from 'three/addons/postprocessing/EffectComposer.js';
            
            const CONFIG = {json.dumps(config)};
            
            // 常量定义
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
            
            // 初始化场景
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x01050a);
            scene.fog = new THREE.FogExp2(0x01050a, 0.015);
            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(5.5, 4.2, 7.5);
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
            const effectComposer = new EffectComposer(renderer);
            effectComposer.addPass(renderScene);
            effectComposer.addPass(bloomPass);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
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
            const fiberMaterial = new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.85, roughness: 0.25 }});
            const fiberMeshes = [];
            for (let i = 0; i < SHEET_COUNT; i++) {{
                const zPos = (i - 2) * S_PITCH_M;
                const sheet = new THREE.Group();
                // 集水槽
                const headerMat = new THREE.MeshStandardMaterial({{ color: 0xE0C8A0, metalness: 0.6 }});
                const headerGeo = new THREE.BoxGeometry(SHEET_WIDTH, 0.12, 0.04);
                const topHeader = new THREE.Mesh(headerGeo, headerMat);
                const bottomHeader = new THREE.Mesh(headerGeo, headerMat);
                topHeader.position.y = halfLen - 0.04;
                bottomHeader.position.y = -halfLen + 0.04;
                sheet.add(topHeader, bottomHeader);
                // 导轨
                const railMat = new THREE.MeshStandardMaterial({{ color: 0xCCCCDD, metalness: 0.9 }});
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
            const pipeMatShiny = new THREE.MeshStandardMaterial({{ color: 0x557788, metalness: 0.8 }});
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
            const sludgeLayerMat = new THREE.MeshStandardMaterial({{ color: 0x7a4a2a, roughness: 0.8, metalness: 0.1, emissive: 0x331100 }});
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
                bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random() * (BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.size[i] = 0.5 + Math.random() * 0.8;
                bubbleData.active[i] = 1;
            }}
            // 初始化：气泡分散在不同高度，避免第一帧爆发
            for (let i = 0; i < BUBBLE_COUNT; i++) {{
                const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                bubbleData.posX[i] = (Math.random() - 0.5) * (SHEET_WIDTH - 0.2);
                bubbleData.posY[i] = -halfLen - PIPE_OFFSET + 0.04 + Math.random() * (halfLen + 0.4);
                bubbleData.posZ[i] = pipe.z + (Math.random()-0.5)*0.1;
                bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random() * (BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                bubbleData.size[i] = 0.5 + Math.random() * 0.8;
                bubbleData.active[i] = 1;
            }}
            
            // 动画循环
            let lastTime = performance.now();
            let burstActive = true;
            let lastBurstState = true;
            let spawnAccum = 0;
            const dummyMat = new THREE.Object3D();
            const modeIndicator = document.getElementById('mode-indicator');
            const baseScale = H_SIZE * 0.007;
            
            function animate() {{
                const now = performance.now();
                let dt = Math.min((now - lastTime)/1000, 0.05);
                lastTime = now;
                const time = now / 1000;
                const normIntensity = (INTENSITY - 50) / 100;
                let effectivePower = 0;
                
                if (MODE === 'cont') {{
                    effectivePower = normIntensity;
                    modeIndicator.innerHTML = '<span style="background:#00aaff; padding:2px 6px; border-radius:4px;">🔘 连续曝气</span>';
                    // 连续模式：持续生成新气泡，但不超过目标数量
                    const targetActive = Math.floor(BUBBLE_COUNT * normIntensity * 0.6);
                    const spawnRate = targetActive * 1.2 * dt;
                    spawnAccum += spawnRate;
                    let spawns = 0;
                    for (let i = 0; i < BUBBLE_COUNT && spawnAccum >= 1 && spawns < 5; i++) {{
                        if (!bubbleData.active[i]) {{
                            const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                            resetBubble(i, pipe);
                            spawnAccum -= 1;
                            spawns++;
                        }}
                    }}
                }} else {{
                    // 脉冲模式
                    const cycle = (time / PULSE_PERIOD) % 1;
                    burstActive = cycle < (1 / PULSE_PERIOD);
                    if (burstActive) {{
                        effectivePower = normIntensity * PULSE_POWER_BOOST;
                        modeIndicator.innerHTML = '<span class="pulse-badge">⚡ 脉冲 ON</span>';
                    }} else {{
                        effectivePower = 0;
                        modeIndicator.innerHTML = '<span class="pulse-badge" style="background:#333;">⏸ 脉冲 OFF</span>';
                    }}
                    // 脉冲开启的瞬间，一次性生成大量气泡（burst）
                    if (burstActive && !lastBurstState) {{
                        const burstCount = Math.floor(BUBBLE_COUNT * normIntensity * 0.5);
                        let spawned = 0;
                        for (let i = 0; i < BUBBLE_COUNT && spawned < burstCount; i++) {{
                            if (!bubbleData.active[i]) {{
                                const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                                resetBubble(i, pipe);
                                spawned++;
                            }}
                        }}
                    }}
                    lastBurstState = burstActive;
                }}
                
                // 更新所有气泡（正常上升，无论脉冲开关）
                for (let i = 0; i < BUBBLE_COUNT; i++) {{
                    if (!bubbleData.active[i]) continue;
                    bubbleData.posY[i] += bubbleData.vel[i] * (1 + normIntensity * 1.2) * dt * 60;
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
                
                // 膜丝摆动
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
                    fiber.position.y = Math.sin(phase * 1.7) * limitedAmp * 0.3;
                }}
                
                // 污泥层更新
                sludgeLayer.scale.y = Math.max(SLUDGE_LEVEL, 0.01) / 0.05;
                sludgeLayer.position.y = bottomY + sludgeLayer.scale.y * 0.05 / 2;
                
                controls.update();
                effectComposer.render();
                requestAnimationFrame(animate);
            }}
            animate();
            
            // 全屏和复位
            document.getElementById('fullscreen-btn').onclick = () => {{
                if (!document.fullscreenElement) document.documentElement.requestFullscreen();
                else document.exitFullscreen();
            }};
            document.getElementById('reset-cam').onclick = () => {{
                camera.position.set(5.5, 4.2, 7.5);
                controls.target.set(0, 0.2, 0);
                controls.update();
            }};
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
