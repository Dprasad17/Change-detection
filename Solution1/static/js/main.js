document.addEventListener('DOMContentLoaded', () => {
    const t1Drop = document.getElementById('t1-drop');
    const t2Drop = document.getElementById('t2-drop');
    const t1Input = document.getElementById('t1-input');
    const t2Input = document.getElementById('t2-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('main-loader');
    const statusText = document.getElementById('analysis-status');
    const resultsSection = document.getElementById('results');
    const downloadMask = document.getElementById('download-mask');
    const connectGisBtn = document.getElementById('connect-gis-btn');
    const connectGisHeader = document.getElementById('connect-gis-header');
    const gisModal = document.getElementById('gis-modal');
    const closeModal = document.querySelector('.close-modal');
    const qgisLink = document.getElementById('qgis-project-link');

    const steps = [
        document.getElementById('step-1'),
        document.getElementById('step-2'),
        document.getElementById('step-3')
    ];

    // Initialize Leaflet Map
    let map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: -3,
        zoom: 0,
        center: [0, 0]
    });

    let layerControl = L.control.layers(null, null, { collapsed: false }).addTo(map);
    let currentLayers = [];

    // Modal logic
    const openModal = () => gisModal.style.display = 'flex';
    connectGisBtn.onclick = openModal;
    if (connectGisHeader) connectGisHeader.onclick = (e) => { e.preventDefault(); openModal(); };
    closeModal.onclick = () => gisModal.style.display = 'none';
    window.onclick = (event) => { if (event.target == gisModal) gisModal.style.display = 'none'; };

    // Handle Drop Zone Clicking
    t1Drop.addEventListener('click', () => t1Input.click());
    t2Drop.addEventListener('click', () => t2Input.click());

    // Drag and drop visual feedback
    const addDragListeners = (dropZone) => {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    };
    addDragListeners(t1Drop);
    addDragListeners(t2Drop);

    const handleFileSelect = (input, dropZone, filenameId) => {
        if (input.files.length > 0) {
            const filename = input.files[0].name;
            const filenameEl = document.getElementById(filenameId);
            if (filenameEl) {
                filenameEl.textContent = filename;
                filenameEl.style.display = 'block';
            }
            dropZone.classList.add('file-selected');
            updateWizard(1);
        }
    };

    t1Input.addEventListener('change', () => handleFileSelect(t1Input, t1Drop, 't1-filename'));
    t2Input.addEventListener('change', () => handleFileSelect(t2Input, t2Drop, 't2-filename'));

    const updateWizard = (stepIndex) => {
        steps.forEach((s, i) => {
            if (i < stepIndex) {
                s.classList.add('completed');
                s.classList.remove('active');
            } else if (i === stepIndex) {
                s.classList.add('active');
                s.classList.remove('completed');
            } else {
                s.classList.remove('active', 'completed');
            }
        });
    };

    analyzeBtn.addEventListener('click', async () => {
        if (t1Input.files.length === 0 || t2Input.files.length === 0) {
            alert('Please select both T1 and T2 images.');
            return;
        }

        const formData = new FormData();
        formData.append('t1', t1Input.files[0]);
        formData.append('t2', t2Input.files[0]);

        // Show process status section
        analyzeBtn.style.display = 'none';
        const processStatus = document.getElementById('process-status');
        if (processStatus) processStatus.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';

        // Get all UI elements
        const progressBar = document.getElementById('progress-bar');
        const stepIcon = document.getElementById('current-step-icon');
        const stepTitle = document.getElementById('current-step-title');
        const stepDescription = document.getElementById('current-step-description');
        const timelineSteps = [
            document.getElementById('timeline-1'),
            document.getElementById('timeline-2'),
            document.getElementById('timeline-3'),
            document.getElementById('timeline-4')
        ];

        // Step configuration
        const steps = [
            { icon: '📤', title: 'Uploading Files', description: 'Sending satellite imagery to the server...', progress: 10 },
            { icon: '🔧', title: 'Preprocessing', description: 'Aligning and normalizing image pairs...', progress: 35 },
            { icon: '🧠', title: 'AI Analysis', description: 'Running Siamese UNet deep learning model...', progress: 70 },
            { icon: '🗺️', title: 'Generating Results', description: 'Creating GIS layers and change detection mask...', progress: 90 }
        ];

        const setActiveStep = (index) => {
            // Update current step display
            if (stepIcon) stepIcon.textContent = steps[index].icon;
            if (stepTitle) stepTitle.textContent = steps[index].title;
            if (stepDescription) stepDescription.textContent = steps[index].description;
            if (progressBar) progressBar.style.width = steps[index].progress + '%';

            // Update timeline
            timelineSteps.forEach((step, i) => {
                if (!step) return;
                step.classList.remove('active', 'completed');
                if (i < index) step.classList.add('completed');
                else if (i === index) step.classList.add('active');
            });
        };

        // Step 1: Uploading
        setActiveStep(0);

        try {
            // Step 2: Preprocessing (after upload starts)
            setTimeout(() => setActiveStep(1), 1500);

            // Step 3: AI Inference
            setTimeout(() => setActiveStep(2), 4000);

            // Step 4: GIS Export
            setTimeout(() => setActiveStep(3), 8000);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }

            const data = await response.json();

            // Wizard Step 3: Complete
            updateWizard(3); // Mark all completed
            statusText.innerText = "Analysis Successful!";

            // Clear existing layers
            currentLayers.forEach(l => map.removeLayer(l));
            currentLayers = [];

            map.removeControl(layerControl);
            layerControl = L.control.layers(null, null, { collapsed: false }).addTo(map);

            const bounds = data.bounds || [[0, 0], [1000, 1000]];

            // Add Layers
            const t1Layer = L.imageOverlay(data.t1_url, bounds);
            const t2Layer = L.imageOverlay(data.t2_url, bounds);
            const maskLayer = L.imageOverlay(data.mask_url, bounds, { opacity: 0.8 });

            t1Layer.addTo(map);
            t2Layer.addTo(map);
            maskLayer.addTo(map);

            layerControl.addBaseLayer(t1Layer, "T1: Before");
            layerControl.addBaseLayer(t2Layer, "T2: After");
            layerControl.addOverlay(maskLayer, "Change Mask");

            map.fitBounds(bounds);

            downloadMask.href = data.download_url;
            qgisLink.href = data.qgis_url;

            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
            statusText.innerText = `Error: ${error.message}`;
            updateWizard(0);
        } finally {
            analyzeBtn.style.display = 'inline-block';
            loader.style.display = 'none';
        }
    });

    // Drag and Drop
    [t1Drop, t2Drop].forEach(zone => {
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.style.borderColor = 'var(--primary)'; });
        zone.addEventListener('dragleave', () => { zone.style.borderColor = 'var(--border)'; });
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            const input = zone === t1Drop ? t1Input : t2Input;
            input.files = e.dataTransfer.files;
            handleFileSelect(input, zone);
        });
    });
});
