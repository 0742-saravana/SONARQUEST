/**
 * SONARQUEST: Clean & Fresh Client Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    const state = {
        currentView: 'landing', // 'landing' or 'workspace'
        samples: [],
        currentSampleId: null,
        selectedFile: null,
        rawImageDataUrl: null,
        lastAnalysisResult: null,
        activeViewMode: 'annotated',
        map: null,
        mapMarkers: [],
        mapTrackLine: null
    };

    // Navigation & View Elements
    const viewLanding = document.getElementById('view-landing');
    const viewWorkspace = document.getElementById('view-workspace');
    const btnNavLanding = document.getElementById('btn-nav-landing');
    const btnNavWorkspace = document.getElementById('btn-nav-workspace');
    const btnHeaderStart = document.getElementById('btn-header-start');
    const btnHeroLaunch = document.getElementById('btn-hero-launch');
    const btnHeroSample = document.getElementById('btn-hero-sample');
    const btnBackToLanding = document.getElementById('btn-back-to-landing');
    const navGoHome = document.getElementById('nav-go-home');

    // Controls & Form Elements
    const sampleSelect = document.getElementById('sample-select');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectedFileInfo = document.getElementById('selected-file-info');
    const fileNameDisplay = document.getElementById('file-name-display');
    const btnClearFile = document.getElementById('btn-clear-file');
    
    const confSlider = document.getElementById('confidence-range');
    const confVal = document.getElementById('conf-val');
    const despeckleToggle = document.getElementById('despeckle-toggle');
    const btnRunAnalysis = document.getElementById('btn-run-analysis');
    const btnLoadSampleHero = document.getElementById('btn-load-sample-hero');

    // Download Action Buttons
    const btnDownloadJpg = document.getElementById('btn-download-jpg');
    const btnDownloadCsv = document.getElementById('btn-download-csv');
    const btnDownloadJson = document.getElementById('btn-download-json');
    const btnDownloadPdf = document.getElementById('btn-download-pdf');

    // Viewport Elements
    const viewModeTabs = document.getElementById('view-mode-tabs');
    const emptyState = document.getElementById('empty-state');
    const sonarDisplayImg = document.getElementById('sonar-display-img');
    const loadingOverlay = document.getElementById('loading-overlay');
    const imageContainer = document.getElementById('image-container');
    const hudCoords = document.getElementById('hud-coords');
    const hudDetCount = document.getElementById('hud-det-count');
    const snrBadge = document.getElementById('snr-badge');
    const detectionTableBody = document.getElementById('detection-table-body');

    // ================= 1. Initialization =================
    fetchSamples();
    setupNavigation();
    setupEventListeners();

    // ================= 2. View Switching =================
    function switchView(viewName) {
        state.currentView = viewName;
        if (viewName === 'landing') {
            viewLanding.classList.add('active');
            viewWorkspace.classList.remove('active');
            btnNavLanding.classList.add('active');
            btnNavWorkspace.classList.remove('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            viewLanding.classList.remove('active');
            viewWorkspace.classList.add('active');
            btnNavLanding.classList.remove('active');
            btnNavWorkspace.classList.add('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
            // Initialize Leaflet map if not already done
            if (!state.map) {
                setTimeout(initMap, 200);
            } else {
                setTimeout(() => state.map.invalidateSize(), 200);
            }
        }
    }

    function setupNavigation() {
        btnNavLanding.addEventListener('click', () => switchView('landing'));
        btnNavWorkspace.addEventListener('click', () => switchView('workspace'));
        navGoHome.addEventListener('click', () => switchView('landing'));
        btnBackToLanding.addEventListener('click', () => switchView('landing'));
        
        btnHeaderStart.addEventListener('click', () => switchView('workspace'));
        btnHeroLaunch.addEventListener('click', () => switchView('workspace'));
        btnHeroSample.addEventListener('click', () => {
            switchView('workspace');
            triggerQuickDemo(0);
        });
    }

    // ================= 3. Leaflet Ocean Map Setup =================
    function initMap() {
        // Deep Offshore Ocean Sector (Bay of Bengal Marine Basin, 35+ nautical miles offshore)
        const defaultLat = 13.1500;
        const defaultLon = 80.6500;

        // Base Layer 1: Esri Ocean Basemap & Bathymetric Seabed Depth Contours
        const esriOceanBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 16,
            attribution: 'Esri, GEBCO, NOAA, National Geographic'
        });

        const esriOceanRef = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 16
        });

        // Base Layer 2: Satellite Ocean Surface
        const satelliteOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 18,
            attribution: 'Esri, Maxar, Earthstar Geographics'
        });

        // Base Layer 3: Nautical Light Chart
        const nauticalChart = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            maxZoom: 19
        });

        state.map = L.map('seafloor-map', {
            zoomControl: true,
            layers: [esriOceanBase, esriOceanRef]
        }).setView([defaultLat, defaultLon], 13);

        // Add Layer Control
        const baseMaps = {
            "🌊 Ocean Bathymetry (Seabed)": L.layerGroup([esriOceanBase, esriOceanRef]),
            "🛰️ Satellite Marine View": satelliteOcean,
            "⚓ Nautical Chart": nauticalChart
        };
        L.control.layers(baseMaps, null, { position: 'topright' }).addTo(state.map);

        drawAuvTrack(defaultLat, defaultLon);
    }

    function drawAuvTrack(baseLat, baseLon) {
        if (state.mapTrackLine) {
            state.map.removeLayer(state.mapTrackLine);
        }

        const trackPoints = [
            [baseLat - 0.008, baseLon - 0.008],
            [baseLat, baseLon],
            [baseLat + 0.008, baseLon + 0.008]
        ];

        state.mapTrackLine = L.polyline(trackPoints, {
            color: '#0284c7',
            weight: 3,
            dashArray: '6, 8',
            opacity: 0.9
        }).addTo(state.map);
    }

    function clearMapMarkers() {
        state.mapMarkers.forEach(m => state.map.removeLayer(m));
        state.mapMarkers = [];
    }

    function plotDetectionsOnMap(detections, baseLat = 13.1500, baseLon = 80.6500) {
        clearMapMarkers();
        drawAuvTrack(baseLat, baseLon);

        if (!detections || detections.length === 0) return;

        const bounds = [];
        bounds.push([baseLat, baseLon]);

        const markerColors = {
            ghost_net: '#ef4444',
            shipwreck: '#f97316',
            pipe_cylinder: '#10b981',
            debris: '#8b5cf6',
            natural_anomaly: '#94a3b8'
        };

        detections.forEach(det => {
            const geo = det.geotag || {};
            const lat = geo.latitude || baseLat;
            const lon = geo.longitude || baseLon;
            bounds.push([lat, lon]);

            const color = markerColors[det.type] || '#2563eb';

            const customIcon = L.divIcon({
                className: 'custom-sonar-pin',
                html: `<div style="background-color: ${color}; width: 14px; height: 14px; border-radius: 50%; border: 2px solid #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.3);"></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });

            const marker = L.marker([lat, lon], { icon: customIcon }).addTo(state.map);
            
            const popupContent = `
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; color: #0f172a; line-height: 1.4;">
                    <strong style="color: ${color}; font-size: 13px;">${det.id}: ${det.label}</strong><br>
                    <strong>Confidence:</strong> ${(det.confidence * 100).toFixed(0)}%<br>
                    <strong>Threat Level:</strong> ${det.hazard_level}<br>
                    <strong>Slant Range:</strong> ${geo.slant_range_m || det.slant_range_m} m<br>
                    <strong>GPS:</strong> ${lat.toFixed(5)}, ${lon.toFixed(5)}
                </div>
            `;
            marker.bindPopup(popupContent);
            state.mapMarkers.push(marker);
        });

        if (bounds.length > 0) {
            state.map.fitBounds(bounds, { padding: [25, 25] });
        }
    }

    // ================= 4. Load Sample Presets =================
    async function fetchSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            state.samples = data;

            sampleSelect.innerHTML = '<option value="">-- Choose Sample Survey Log --</option>';
            data.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s.id;
                opt.textContent = `${s.title}`;
                sampleSelect.appendChild(opt);
            });
        } catch (err) {
            console.error('Failed to load sample presets:', err);
        }
    }

    // ================= 5. Event Listeners =================
    function setupEventListeners() {
        // Preset select
        sampleSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            if (val) {
                state.currentSampleId = val;
                clearSelectedFile();
                const selectedSample = state.samples.find(s => s.id === val);
                if (selectedSample) {
                    state.rawImageDataUrl = selectedSample.url;
                    displayRawPreview(selectedSample.url);
                }
            }
        });

        btnLoadSampleHero.addEventListener('click', () => triggerQuickDemo(0));

        // Drag and drop upload
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        btnClearFile.addEventListener('click', clearSelectedFile);

        // Slider
        confSlider.addEventListener('input', (e) => {
            confVal.textContent = `${e.target.value}%`;
        });

        // Run Analysis
        btnRunAnalysis.addEventListener('click', executeDetection);

        // View Mode Switcher
        viewModeTabs.addEventListener('click', (e) => {
            const btn = e.target.closest('.tab-pill');
            if (!btn) return;
            
            document.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            state.activeViewMode = btn.dataset.mode;
            updateDisplayedImage();
        });

        // Cursor Coordinate Tracking
        imageContainer.addEventListener('mousemove', (e) => {
            const rect = imageContainer.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);
            const normX = (x - rect.width / 2) / (rect.width / 2);
            const rangeM = Math.abs(normX * 60).toFixed(1);
            hudCoords.innerHTML = `<i class="fa-regular fa-compass"></i> Cursor: X: ${x}px | Y: ${y}px | Range: ${rangeM}m`;
        });

        // Download Event Listeners
        btnDownloadJpg.addEventListener('click', downloadBoxedJpg);
        btnDownloadCsv.addEventListener('click', downloadCsv);
        btnDownloadJson.addEventListener('click', downloadJson);
        btnDownloadPdf.addEventListener('click', downloadPdf);
    }

    function triggerQuickDemo(index = 0) {
        if (state.samples.length > index) {
            const targetSample = state.samples[index];
            sampleSelect.value = targetSample.id;
            state.currentSampleId = targetSample.id;
            clearSelectedFile();
            state.rawImageDataUrl = targetSample.url;
            displayRawPreview(targetSample.url);
            executeDetection();
        }
    }

    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid Side-Scan Sonar image (PNG, JPG, TIFF).');
            return;
        }

        state.selectedFile = file;
        state.currentSampleId = null;
        sampleSelect.value = '';

        fileNameDisplay.textContent = file.name;
        selectedFileInfo.classList.remove('hidden');

        const reader = new FileReader();
        reader.onload = (e) => {
            state.rawImageDataUrl = e.target.result;
            displayRawPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedFile() {
        state.selectedFile = null;
        fileInput.value = '';
        selectedFileInfo.classList.add('hidden');
    }

    function displayRawPreview(imgUrl) {
        emptyState.classList.add('hidden');
        sonarDisplayImg.classList.remove('hidden');
        sonarDisplayImg.src = imgUrl;
    }

    // ================= 6. Detection Execution =================
    async function executeDetection() {
        if (!state.selectedFile && !state.currentSampleId) {
            alert('Please select a sample survey log or upload a sonar image first.');
            return;
        }

        loadingOverlay.classList.remove('hidden');

        const formData = new FormData();
        if (state.selectedFile) {
            formData.append('file', state.selectedFile);
        } else if (state.currentSampleId) {
            formData.append('sample_id', state.currentSampleId);
        }

        const confThresh = parseFloat(confSlider.value) / 100.0;
        formData.append('confidence_threshold', confThresh);
        formData.append('enable_despeckle', despeckleToggle.checked);

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Detection failed');
            }

            const data = await res.json();
            state.lastAnalysisResult = data;

            renderResults(data);

        } catch (err) {
            console.error('Detection error:', err);
            alert(`Analysis failed: ${err.message}`);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    function renderResults(data) {
        updateDisplayedImage();

        const detCount = data.detections.length;
        hudDetCount.textContent = `${detCount} Hazard${detCount === 1 ? '' : 's'} Detected`;
        snrBadge.innerHTML = `<i class="fa-solid fa-wave-square"></i> SNR: ${data.telemetry.snr_db} dB`;

        // Enable Download Action Buttons
        btnDownloadJpg.disabled = false;
        btnDownloadCsv.disabled = false;
        btnDownloadJson.disabled = false;
        btnDownloadPdf.disabled = false;

        // Render Table Rows
        renderTableRows(data.detections);

        // Plot map pins in deep ocean survey coordinates
        const baseCoords = data.mission_metadata?.auv_start_coords || { lat: 13.1500, lon: 80.6500 };
        plotDetectionsOnMap(data.detections, baseCoords.lat, baseCoords.lon);
    }

    function updateDisplayedImage() {
        if (!state.lastAnalysisResult) {
            if (state.rawImageDataUrl) {
                sonarDisplayImg.src = state.rawImageDataUrl;
            }
            return;
        }

        const res = state.lastAnalysisResult;
        switch (state.activeViewMode) {
            case 'annotated':
                sonarDisplayImg.src = res.annotated_image;
                break;
            case 'segmentation':
                sonarDisplayImg.src = res.segmentation_image;
                break;
            case 'despeckled':
                sonarDisplayImg.src = res.despeckled_image;
                break;
            case 'raw':
                sonarDisplayImg.src = state.rawImageDataUrl || res.despeckled_image;
                break;
        }
    }

    function renderTableRows(detections) {
        if (!detections || detections.length === 0) {
            detectionTableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No hazards detected matching current confidence threshold.</td></tr>';
            return;
        }

        detectionTableBody.innerHTML = '';
        detections.forEach(d => {
            const geo = d.geotag || {};
            const tr = document.createElement('tr');
            const sevColor = d.hazard_level === 'High' ? 'text-red' : (d.hazard_level === 'Medium' ? 'text-amber' : 'text-sec');

            tr.innerHTML = `
                <td><strong>${d.id}</strong></td>
                <td>${d.label}</td>
                <td><strong>${(d.confidence * 100).toFixed(0)}%</strong></td>
                <td><span class="${sevColor} font-semibold">${d.hazard_level}</span></td>
                <td>${d.slant_range_m} m</td>
                <td>${geo.latitude ? geo.latitude.toFixed(5) : '--'}, ${geo.longitude ? geo.longitude.toFixed(5) : '--'}</td>
            `;
            detectionTableBody.appendChild(tr);
        });
    }

    // ================= 7. Download Action Implementations =================
    // 1. Download Annotated JPG with Bounding Boxes
    async function downloadBoxedJpg() {
        if (!state.lastAnalysisResult || !state.lastAnalysisResult.annotated_image) return;
        
        try {
            const res = await fetch('/api/export/image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: state.lastAnalysisResult.annotated_image })
            });
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sonarquest_detected_objects_${Date.now()}.jpg`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            console.error('Image export failed:', err);
        }
    }

    // 2. Download CSV Metadata Report
    async function downloadCsv() {
        if (!state.lastAnalysisResult) return;
        try {
            const res = await fetch('/api/export/csv', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detections: state.lastAnalysisResult.detections })
            });
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sonarquest_metadata_report_${Date.now()}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            console.error('CSV export failed:', err);
        }
    }

    // 3. Download JSON Manifest
    async function downloadJson() {
        if (!state.lastAnalysisResult) return;
        try {
            const res = await fetch('/api/export/json', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mission_metadata: state.lastAnalysisResult.mission_metadata,
                    detections: state.lastAnalysisResult.detections
                })
            });
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sonarquest_inspection_manifest_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            console.error('JSON export failed:', err);
        }
    }

    // 4. Download PDF Mission Inspection Report
    async function downloadPdf() {
        if (!state.lastAnalysisResult) return;
        try {
            const res = await fetch('/api/export/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mission_metadata: state.lastAnalysisResult.mission_metadata,
                    telemetry: state.lastAnalysisResult.telemetry,
                    detections: state.lastAnalysisResult.detections,
                    annotated_image: state.lastAnalysisResult.annotated_image
                })
            });
            if (!res.ok) throw new Error("PDF generation failed");
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sonarquest_mission_report_${Date.now()}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            console.error('PDF export failed:', err);
            alert('PDF generation failed: ' + err.message);
        }
    }
});
