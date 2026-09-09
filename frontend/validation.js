const urlParams = new URLSearchParams(window.location.search);
const jobId = urlParams.get('job_id');

if (!jobId) {
    alert("No job_id provided.");
    window.location.href = "/";
}

let hrFile = null;
const mapImages = {
    reconstruction_error: null,
    confidence: null,
    sam_error: null,
    ndvi_error: null
};

let activeTab = 'confidence';

// Viewer state
const viewerState = {
    scale: 1, panX: 0, panY: 0, w: 0, h: 0,
    container: document.getElementById('val-container'),
    img: document.getElementById('val-img')
};

const interpretations = {
    confidence: {
        title: "Reference-Based Confidence",
        text: "Measures pixel-level agreement between the SR output and your HR reference. <strong>Green = high confidence</strong> (SR closely matches HR). <strong>Red = low confidence</strong> (SR diverges from HR). Derived from the inverse of the 4-band reconstruction error with 2nd–98th percentile normalization. This is a reference-based quality score, not a model-uncertainty estimate."
    },
    reconstruction_error: {
        title: "Multispectral Reconstruction Error",
        text: "Shows the mean absolute error across all 4 bands (B2, B3, B4, B8) between the SR output and the HR reference, computed in the model's normalized space (/3000). <strong>Brighter/hotter regions</strong> (yellow-white) indicate higher error. <strong>Darker regions</strong> indicate close agreement. This captures both spatial and spectral reconstruction fidelity."
    },
    sam_error: {
        title: "Spectral Angle Mapper (SAM)",
        text: "Measures the angular difference between the 4-band spectral vectors of SR and HR at each pixel. <strong>Lower values (dark)</strong> = spectral fidelity preserved. <strong>Higher values (bright)</strong> = spectral distortion. SAM is invariant to brightness — it captures pure spectral shape changes, making it particularly sensitive to band-ratio errors at mixed-class boundaries and edges."
    },
    ndvi_error: {
        title: "NDVI Reconstruction Error",
        text: "Compares the Normalized Difference Vegetation Index (B8−B4)/(B8+B4) between SR and HR. NDVI is the most widely used vegetation health indicator in remote sensing. <strong>Small differences (dark)</strong> = vegetation patterns well-preserved. <strong>Large differences (bright)</strong> = the SR model has altered the Red/NIR band ratio, potentially distorting vegetation analysis downstream."
    }
};

// UI Elements
const dropZone = document.getElementById('hr-drop-zone');
const fileInput = document.getElementById('hr-file-input');
const startBtn = document.getElementById('validate-start-btn');
const errorText = document.getElementById('upload-error');

const uploadPanel = document.getElementById('upload-panel');
const processingPanel = document.getElementById('processing-panel');
const metricsPanel = document.getElementById('metrics-panel');
const interpPanel = document.getElementById('interpretation-panel');
const controls = document.getElementById('val-controls');
const stageEmpty = document.getElementById('stage-empty');
const container = document.getElementById('val-container');

// Tabs
const tabs = document.querySelectorAll('.val-tab');
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        activeTab = tab.dataset.tab;
        
        if (mapImages[activeTab]) {
            viewerState.img.src = mapImages[activeTab];
        }
        
        document.getElementById('interp-title').innerText = interpretations[activeTab].title;
        document.getElementById('interp-text').innerHTML = interpretations[activeTab].text;
    });
});

// File Handling
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
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.tif') && !name.endsWith('.tiff')) {
        errorText.innerText = "Only .tif or .tiff files are supported for HR reference.";
        errorText.classList.remove('hidden');
        hrFile = null;
        startBtn.disabled = true;
        dropZone.classList.remove('has-file');
        return;
    }
    
    hrFile = file;
    errorText.classList.add('hidden');
    dropZone.classList.add('has-file');
    dropZone.querySelector('p').textContent = file.name;
    startBtn.disabled = false;
}

// Start Validation
startBtn.addEventListener('click', async () => {
    if (!hrFile) return;
    
    uploadPanel.classList.add('hidden');
    processingPanel.classList.remove('hidden');
    
    const formData = new FormData();
    formData.append('hr_file', hrFile);
    formData.append('job_id', jobId);
    
    try {
        const response = await fetch('/validate', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Validation failed.");
        }
        
        populateUI(data);
    } catch (err) {
        alert(err.message);
        uploadPanel.classList.remove('hidden');
        processingPanel.classList.add('hidden');
    }
});

function populateUI(data) {
    processingPanel.classList.add('hidden');
    metricsPanel.classList.remove('hidden');
    interpPanel.classList.remove('hidden');
    
    // Store maps
    mapImages.reconstruction_error = "data:image/png;base64," + data.maps.reconstruction_error;
    mapImages.confidence = "data:image/png;base64," + data.maps.confidence;
    mapImages.sam_error = "data:image/png;base64," + data.maps.sam_error;
    mapImages.ndvi_error = "data:image/png;base64," + data.maps.ndvi_error;
    
    // Set initial image
    viewerState.img.src = mapImages[activeTab];
    viewerState.w = data.dimensions.width;
    viewerState.h = data.dimensions.height;
    
    // Show canvas
    stageEmpty.classList.add('hidden');
    container.classList.remove('hidden');
    controls.style.display = 'flex';
    
    // Populate metrics
    document.getElementById('val-psnr').innerText = data.metrics.psnr.toFixed(2);
    document.getElementById('val-ssim').innerText = data.metrics.ssim.toFixed(4);
    document.getElementById('val-sam').innerText = data.metrics.sam_mean_degrees.toFixed(2) + "°";
    document.getElementById('val-ndvi').innerText = data.metrics.ndvi_mean_error.toFixed(4);
    document.getElementById('val-rmse').innerText = data.metrics.rmse.toFixed(4);
    document.getElementById('val-conf').innerText = (data.metrics.confidence_mean * 100).toFixed(1) + "%";
    
    // Auto-fit
    setTimeout(fitToScreen, 100);
}

// Viewer Logic
const pane = document.getElementById('val-pane');
let isDragging = false;
let startX, startY;

pane.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX - viewerState.panX;
    startY = e.clientY - viewerState.panY;
    pane.style.cursor = 'grabbing';
});

window.addEventListener('mouseup', () => {
    isDragging = false;
    pane.style.cursor = 'grab';
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    e.preventDefault();
    viewerState.panX = e.clientX - startX;
    viewerState.panY = e.clientY - startY;
    renderTransform();
});

pane.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = pane.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const delta = e.deltaY < 0 ? 1.1 : 0.9;
    const prevScale = viewerState.scale;
    viewerState.scale = Math.max(0.01, Math.min(viewerState.scale * delta, 50));
    
    viewerState.panX = x - (x - viewerState.panX) * (viewerState.scale / prevScale);
    viewerState.panY = y - (y - viewerState.panY) * (viewerState.scale / prevScale);
    
    renderTransform();
});

document.getElementById('ctrl-zoom-in').onclick = () => {
    const rect = pane.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    
    const prevScale = viewerState.scale;
    viewerState.scale = Math.max(0.01, Math.min(viewerState.scale * 1.2, 50));
    viewerState.panX = cx - (cx - viewerState.panX) * (viewerState.scale / prevScale);
    viewerState.panY = cy - (cy - viewerState.panY) * (viewerState.scale / prevScale);
    renderTransform();
};

document.getElementById('ctrl-zoom-out').onclick = () => {
    const rect = pane.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    
    const prevScale = viewerState.scale;
    viewerState.scale = Math.max(0.01, Math.min(viewerState.scale * 0.8, 50));
    viewerState.panX = cx - (cx - viewerState.panX) * (viewerState.scale / prevScale);
    viewerState.panY = cy - (cy - viewerState.panY) * (viewerState.scale / prevScale);
    renderTransform();
};

document.getElementById('ctrl-fit').onclick = fitToScreen;
document.getElementById('ctrl-1to1').onclick = () => {
    const rect = pane.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    
    const prevScale = viewerState.scale;
    viewerState.scale = 1.0;
    viewerState.panX = cx - (cx - viewerState.panX) * (viewerState.scale / prevScale);
    viewerState.panY = cy - (cy - viewerState.panY) * (viewerState.scale / prevScale);
    renderTransform();
};

function fitToScreen() {
    if (viewerState.w === 0) return;
    const padding = 0.9;
    const rect = pane.getBoundingClientRect();
    const scaleX = rect.width * padding / viewerState.w;
    const scaleY = rect.height * padding / viewerState.h;
    
    viewerState.scale = Math.min(scaleX, scaleY);
    viewerState.panX = (rect.width - viewerState.w * viewerState.scale) / 2;
    viewerState.panY = (rect.height - viewerState.h * viewerState.scale) / 2;
    
    renderTransform();
}

function renderTransform() {
    viewerState.container.style.transform = `translate(${viewerState.panX}px, ${viewerState.panY}px) scale(${viewerState.scale})`;
    document.getElementById('zoom-level').innerText = Math.round(viewerState.scale * 100) + "%";
}
