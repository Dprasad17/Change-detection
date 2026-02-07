// Space Background with Three.js
(function() {
    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true
    });
    
    // Set renderer size and position
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.insertBefore(renderer.domElement, document.body.firstChild);
    renderer.domElement.style.position = 'fixed';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.zIndex = '-1';
    
    // Add stars
    function addStars() {
        const starsGeometry = new THREE.BufferGeometry();
        const starsMaterial = new THREE.PointsMaterial({
            color: 0xFFFFFF,
            size: 0.1,
            transparent: true
        });
        
        const starsVertices = [];
        for (let i = 0; i < 10000; i++) {
            const x = (Math.random() - 0.5) * 2000;
            const y = (Math.random() - 0.5) * 2000;
            const z = (Math.random() - 0.5) * 2000;
            starsVertices.push(x, y, z);
        }
        
        starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
        const stars = new THREE.Points(starsGeometry, starsMaterial);
        scene.add(stars);
    }
    
    // Create Earth
    function createEarth() {
        const geometry = new THREE.SphereGeometry(5, 32, 32);
        const texture = new THREE.TextureLoader().load('/static/img/earth-texture.jpg');
        const material = new THREE.MeshPhongMaterial({
            map: texture,
            bumpMap: texture,
            bumpScale: 0.05,
            specular: new THREE.Color('grey'),
            shininess: 5
        });
        
        const earth = new THREE.Mesh(geometry, material);
        scene.add(earth);
        
        // Add atmosphere
        const atmosphereGeometry = new THREE.SphereGeometry(5.1, 32, 32);
        const atmosphereMaterial = new THREE.ShaderMaterial({
            uniforms: {
                'c': { value: 0.7 },
                'p': { value: 5.0 },
                glowColor: { type: 'c', value: new THREE.Color(0x00b3ff) },
                viewVector: { type: 'v3', value: camera.position }
            },
            vertexShader: `
                uniform vec3 viewVector;
                uniform float c;
                uniform float p;
                varying float intensity;
                void main() {
                    vec3 vNormal = normalize(normalMatrix * normal);
                    vec3 vNormel = normalize(normalMatrix * viewVector);
                    intensity = pow(c - dot(vNormal, vNormel), p);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 glowColor;
                varying float intensity;
                void main() {
                    vec3 glow = glowColor * intensity;
                    gl_FragColor = vec4(glow, 1.0);
                }
            `,
            side: THREE.BackSide,
            blending: THREE.AdditiveBlending,
            transparent: true
        });
        
        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        scene.add(atmosphere);
        
        return earth;
    }
    
    // Add lights
    function addLights() {
        const ambientLight = new THREE.AmbientLight(0x333333);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 3, 5);
        scene.add(directionalLight);
    }
    
    // Initialize
    function init() {
        addStars();
        const earth = createEarth();
        addLights();
        
        camera.position.z = 15;
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            // Rotate Earth
            if (earth) {
                earth.rotation.y += 0.001;
            }
            
            renderer.render(scene, camera);
        }
        
        // Handle window resize
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        window.addEventListener('resize', onWindowResize, false);
        animate();
    }
    
    // Load Earth texture
    const loader = new THREE.TextureLoader();
    loader.load('/static/img/earth-texture.jpg', function() {
        init();
    });
    
})();
