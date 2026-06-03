def generate_3d_html(sim):
    # 计算显示膜丝数量
    sheet_area = sim.get_sheet_area()
    diameter_m = sim.fiber_diameter / 1000
    area_per_fiber = np.pi * diameter_m * sim.f_len
    real_fibers = max(1, int(sheet_area / area_per_fiber))
    visual_fibers = min(real_fibers, 150)

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
                
                const headerMat = new THREE.MeshStandardMaterial({{ color: 0xDDC8A0, metalness: 0.5 }});
                const headerGeo = new THREE.BoxGeometry(SHEET_WIDTH, 0.1, 0.05);
                const topHeader = new THREE.Mesh(headerGeo, headerMat);
                const bottomHeader = new THREE.Mesh(headerGeo, headerMat);
                topHeader.position.y = halfLen - 0.02;
                bottomHeader.position.y = -halfLen + 0.02;
                sheet.add(topHeader, bottomHeader);
                
                const railMat = new THREE.MeshStandardMaterial({{ color: 0xCCCCDD, metalness: 0.9 }});
                const railGeo = new THREE.BoxGeometry(0.03, F_LEN, 0.03);
                const leftRail = new THREE.Mesh(railGeo, railMat);
                const rightRail = new THREE.Mesh(railGeo, railMat);
                leftRail.position.set(-halfWidth + SHEET_END_MARGIN, 0, 0);
                rightRail.position.set(halfWidth - SHEET_END_MARGIN, 0, 0);
                sheet.add(leftRail, rightRail);
                
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
            
            // 改进初始化：只激活 30% 的气泡，且主要分布在上半部分，避免首帧密集
            const initActiveCount = Math.floor(BUBBLE_COUNT * 0.3);
            for (let i = 0; i < BUBBLE_COUNT; i++) {{
                if (i < initActiveCount) {{
                    const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                    bubbleData.posX[i] = (Math.random() - 0.5) * (SHEET_WIDTH - 0.2);
                    // 高度范围：从中上部到顶部 (40% 到 100% 高度)
                    const t = 0.4 + Math.random() * 0.6;
                    bubbleData.posY[i] = -halfLen - PIPE_OFFSET + 0.03 + t * (halfLen + 0.3);
                    bubbleData.posZ[i] = pipe.z + (Math.random() - 0.5) * 0.1;
                    bubbleData.vel[i] = BUBBLE_MIN_VEL + Math.random() * (BUBBLE_MAX_VEL - BUBBLE_MIN_VEL);
                    bubbleData.size[i] = 0.6 + Math.random() * 0.7;
                    bubbleData.active[i] = 1;
                }} else {{
                    bubbleData.active[i] = 0;
                }}
            }}
            
            // --- 动画循环 ---
            let lastTime = performance.now();
            let burstActive = true;
            let lastBurstState = true;
            let spawnAccum = 0;
            let firstFrame = true;   // 首帧特殊标记，避免瞬间生成
            const dummyMat = new THREE.Object3D();
            const modeSpan = document.getElementById('mode-badge');
            
            function animate() {{
                const now = performance.now();
                let dt = Math.min((now - lastTime) / 1000, 0.05);
                lastTime = now;
                const time = now / 1000;
                const normIntensity = (INTENSITY - 50) / 100;
                
                let effectivePower = 0;
                if (MODE === 'cont') {{
                    effectivePower = normIntensity;
                    if (modeSpan) modeSpan.innerHTML = ' <span style="background:#00aaff; padding:2px 5px; border-radius:4px;">连续</span>';
                    
                    // 连续模式：匀速生成气泡，首帧跳过生成，避免补充过快
                    if (!firstFrame) {{
                        const targetActive = Math.min(BUBBLE_COUNT, Math.floor(BUBBLE_COUNT * normIntensity * 0.5));
                        const spawnRate = targetActive * 0.8 * dt;
                        spawnAccum += spawnRate;
                        let spawned = 0;
                        for (let i = 0; i < BUBBLE_COUNT && spawnAccum >= 1 && spawned < 4; i++) {{
                            if (!bubbleData.active[i]) {{
                                const pipe = pipePositions[Math.floor(Math.random() * pipePositions.length)];
                                resetBubble(i, pipe);
                                spawnAccum -= 1;
                                spawned++;
                            }}
                        }}
                    }}
                }} else {{
                    const cycle = (time / PULSE_PERIOD) % 1;
                    burstActive = cycle < (1 / PULSE_PERIOD);
                    if (burstActive) {{
                        effectivePower = normIntensity * PULSE_BOOST;
                        if (modeSpan) modeSpan.innerHTML = ' <span style="background:#ff5500; padding:2px 5px; border-radius:4px; animation: blink 0.8s infinite;">脉冲 ON</span>';
                        // 脉冲开启瞬间，一次性生成一批气泡（首帧除外）
                        if (!lastBurstState && !firstFrame) {{
                            const burstCount = Math.floor(BUBBLE_COUNT * normIntensity * 0.3);
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
                    }}
                    lastBurstState = burstActive;
                }}
                
                // 首帧标记清除
                if (firstFrame) firstFrame = false;
                
                // 更新所有气泡位置
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
            
            const style = document.createElement('style');
            style.textContent = `@keyframes blink {{ 0%{{opacity:0.3;}}50%{{opacity:1;}}100%{{opacity:0.3;}}}}`;
            document.head.appendChild(style);
        </script>
    </body>
    </html>
    """
    return html
