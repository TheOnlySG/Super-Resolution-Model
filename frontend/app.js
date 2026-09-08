// DOM Elements
const lrInput = document.getElementById('lr-file-input');
const lrZone = document.getElementById('lr-drop-zone');
const startBtn = document.getElementById('start-btn');

const uploadPanel = document.getElementById('upload-panel');
const processingPanel = document.getElementById('processing-panel');
const metadataPanel = document.getElementById('metadata-panel');

const stageEmpty = document.getElementById('stage-empty');
const viewerContainer = document.getElementById('viewer-container');

// Viewers
const workspace = document.getElementById('workspace');
const paneOriginal = document.getElementById('pane-original');
const paneSR = document.getElementById('pane-sr');
const imgOriginal = document.getElementById('img-original');
const imgSR = document.getElementById('img-sr');
const containerOriginal = document.getElementById('container-original');
const containerSR = document.getElementById('container-sr');

// Controls
const modeBtns = document.querySelectorAll('.mode-btn');
const syncCheckbox = document.getElementById('sync-checkbox');
const activeIndicator = document.getElementById('active-indicator');
const splitDivider = document.getElementById('split-divider');
const overlayControls = document.getElementById('overlay-controls');
const opacitySlider = document.getElementById('opacity-slider');
const zoomLevelText = document.getElementById('zoom-level-text');

// State
let lrFile = null;
let currentJobId = null;

let currentMode = 'side'; // 'side', 'split', 'overlay'
let isSync = true;
let activeViewer = 'original'; // 'original' or 'sr'

const state = {
    original: { scale: 1, panX: 0, panY: 0, w: 0, h: 0, pane: paneOriginal, container: containerOriginal, img: imgOriginal },
    sr: { scale: 1, panX: 0, panY: 0, w: 0, h: 0, pane: paneSR, container: containerSR, img: imgSR }
};

// -----------------------------------------
// FILE UPLOAD & PREVIEW
// -----------------------------------------
function handleFileSelect(files, zone) {
    if (files && files.length > 0) {
        const file = files[0];
        const allowedExt = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jp2'];
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!allowedExt.includes(ext)) {
            alert('Invalid file type. Please upload .png, .jpg, .jpeg, .tif, .tiff, or .jp2');
            return;
        }

        lrFile = file;
        zone.querySelector('p').innerText = lrFile.name;
        zone.classList.add('has-file');
        startBtn.disabled = false;
        
        const browserNative = ['.png', '.jpg', '.jpeg'];
        const needsServerPreview = !browserNative.includes(ext);
        
        if (needsServerPreview) {
            zone.querySelector('p').innerText = 'Converting preview...';
            const previewForm = new FormData();
            previewForm.append('file', file);
            
            fetch('/preview', { method: 'POST', body: previewForm })
                .then(res => res.json())
                .then(data => {
                    const previewSrc = "data:image/png;base64," + data.preview;
                    imgOriginal.src = previewSrc;
                    imgSR.src = previewSrc;
                    state.original.w = data.width;
                    state.original.h = data.height;
                    state.sr.w = data.width;
                    state.sr.h = data.height;
                    
                    zone.querySelector('p').innerText = lrFile.name;
                    stageEmpty.classList.add('hidden');
                    viewerContainer.classList.remove('hidden');
                    
                    fitToScreen('original');
                    if (isSync) syncFrom('original');
                    else fitToScreen('sr');
                })
                .catch(err => {
                    alert("Preview error: " + err.message);
                    zone.querySelector('p').innerText = 'Preview failed';
                });
        } else {
            // Show local preview immediately
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    imgOriginal.src = e.target.result;
                    imgSR.src = e.target.result; // Temp until SR finishes
                    state.original.w = img.naturalWidth;
                    state.original.h = img.naturalHeight;
                    state.sr.w = img.naturalWidth; // Temp
                    state.sr.h = img.naturalHeight; // Temp
                    
                    stageEmpty.classList.add('hidden');
                    viewerContainer.classList.remove('hidden');
                    
                    fitToScreen('original');
                    if (isSync) syncFrom('original');
                    else fitToScreen('sr');
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    }
}

lrInput.addEventListener('change', () => handleFileSelect(lrInput.files, lrZone));

// Drag and drop setup
function setupDragAndDrop(zone, input) {
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.style.borderColor = 'var(--accent-hover)'; });
    zone.addEventListener('dragleave', () => { zone.style.borderColor = 'var(--border)'; });
    zone.addEventListener('drop', (e) => {
        e.preventDefault(); zone.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files; 
            handleFileSelect(e.dataTransfer.files, zone);
        }
    });
    zone.addEventListener('click', (e) => { if (e.target !== input) input.click(); });
}
setupDragAndDrop(lrZone, lrInput);

// -----------------------------------------
// BACKEND INFERENCE PIPELINE
// -----------------------------------------
startBtn.addEventListener('click', async () => {
    if (!lrFile) return;
    uploadPanel.classList.add('hidden');
    processingPanel.classList.remove('hidden');
    
    const setStep = (id, status) => {
        const step = document.getElementById(id);
        if(status === 'active') { step.classList.add('active'); step.classList.remove('done'); }
        if(status === 'done') { step.classList.add('done'); step.classList.remove('active'); }
    };
    setStep('step-upload', 'active');

    try {
        const formData = new FormData();
        formData.append('lr_file', lrFile);
        setStep('step-upload', 'done'); setStep('step-infer', 'active');
        
        const response = await fetch('/predict', { method: 'POST', body: formData });
        setStep('step-infer', 'done'); setStep('step-output', 'active');

        if (!response.ok) {
            const err = await response.json();
            alert("Error: " + (err.detail || "Unknown error")); location.reload(); return;
        }

        const data = await response.json();
        populateUI(data);

        setStep('step-output', 'done');
        setTimeout(() => {
            processingPanel.classList.add('hidden');
            metadataPanel.classList.remove('hidden');
        }, 800);
    } catch (e) {
        alert("Connection error: " + e.message); location.reload();
    }
});

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024; const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function populateUI(data) {
    currentJobId = data.job_id;
    const m = data.metadata;
    
    // Set actual output dimensions
    state.sr.w = m.output.width;
    state.sr.h = m.output.height;
    
    imgSR.onload = () => {
        if (isSync) syncFrom('original');
        else fitToScreen('sr');
    };
    imgSR.src = "data:image/png;base64," + data.visualizations.sr_rgb;

    // Stats
    document.getElementById('metadata-tbody').innerHTML = `
        <tr><td colspan="2" style="color:var(--accent-hover); font-weight:bold; border-bottom:1px solid #1f2937; padding-bottom:0.5rem; padding-top:0.5rem;">INPUT</td></tr>
        <tr><td>Width</td><td>${m.input.width} px</td></tr>
        <tr><td>Height</td><td>${m.input.height} px</td></tr>
        <tr><td>Channels</td><td>${m.input.channels}</td></tr>
        <tr><td>File Size</td><td>${formatBytes(m.input.size_bytes)}</td></tr>
        <tr><td>Format</td><td>${m.input.format}</td></tr>
        
        <tr><td colspan="2" style="color:var(--accent-hover); font-weight:bold; border-bottom:1px solid #1f2937; padding-bottom:0.5rem; padding-top:1.5rem;">OUTPUT</td></tr>
        <tr><td>Width</td><td>${m.output.width} px</td></tr>
        <tr><td>Height</td><td>${m.output.height} px</td></tr>
        <tr><td>Channels</td><td>${m.output.channels}</td></tr>
        <tr><td>Scale Factor</td><td>${m.output.scale_factor}×</td></tr>
        <tr><td>Format</td><td>${m.output.format}</td></tr>
    `;
    
    document.getElementById('download-image-btn').onclick = () => {
        window.location.href = "/download_image?job_id=" + currentJobId;
    };
    
    document.getElementById('validate-btn').onclick = () => {
        window.location.href = "/validation.html?job_id=" + currentJobId;
    };

    // --- Segmentation ---
    populateSegmentation(data.segmentation);
}

function populateSegmentation(seg) {
    const section = document.getElementById('segmentation-section');
    const loading = document.getElementById('seg-loading');
    const error = document.getElementById('seg-error');
    const results = document.getElementById('seg-results');

    // Always show the section after SR completes
    section.classList.remove('hidden');
    loading.classList.add('hidden');
    error.classList.add('hidden');
    results.classList.add('hidden');

    if (!seg || !seg.available) {
        // Show error state
        error.classList.remove('hidden');
        document.getElementById('seg-error-msg').textContent = 
            (seg && seg.error) ? seg.error : 'Segmentation unavailable.';
        return;
    }

    // Show results
    results.classList.remove('hidden');

    // Set images
    document.getElementById('seg-mask-img').src = "data:image/png;base64," + seg.mask;
    document.getElementById('seg-overlay-img').src = "data:image/png;base64," + seg.overlay;

    // Build legend
    const legendEl = document.getElementById('seg-legend');
    legendEl.innerHTML = seg.classes.map(c => `
        <div class="seg-legend-item">
            <div class="seg-legend-swatch" style="background-color: ${c.color};"></div>
            <span class="seg-legend-label">${c.label}</span>
        </div>
    `).join('');

    // Build stats with percentage bars
    const statsEl = document.getElementById('seg-stats');
    const maxPct = Math.max(...seg.classes.map(c => c.percentage), 1);
    statsEl.innerHTML = seg.classes.map(c => `
        <div class="seg-stat-row">
            <span class="seg-stat-label">${c.label}</span>
            <div class="seg-stat-bar-bg">
                <div class="seg-stat-bar-fill" style="width: ${(c.percentage / maxPct) * 100}%; background-color: ${c.color};"></div>
            </div>
            <span class="seg-stat-value">${c.percentage.toFixed(1)}%</span>
        </div>
    `).join('');

    // Scroll to segmentation section smoothly
    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

// -----------------------------------------
// VIEWER MODES & SYNC LOGIC
// -----------------------------------------

function setMode(mode) {
    currentMode = mode;
    workspace.className = 'workspace mode-' + mode;
    
    if (isSync) workspace.classList.remove('sync-off');
    else workspace.classList.add('sync-off');

    // UI Toggles
    splitDivider.classList.toggle('hidden', mode !== 'split');
    overlayControls.classList.toggle('hidden', mode !== 'overlay');
    
    if (mode === 'split') updateSplitClip(splitPercent);
    if (mode === 'overlay') paneSR.style.opacity = opacitySlider.value;
    else paneSR.style.opacity = 1;
    
    // Fit views on mode switch just in case pane sizes drastically changed
    // Use requestAnimationFrame so CSS layout applies first
    requestAnimationFrame(() => {
        if (isSync) {
            fitToScreen('original');
            syncFrom('original');
        } else {
            fitToScreen('original');
            fitToScreen('sr');
        }
    });
}

modeBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        modeBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        setMode(e.target.getAttribute('data-mode'));
    });
});

syncCheckbox.addEventListener('change', (e) => {
    isSync = e.target.checked;
    if (isSync) {
        workspace.classList.remove('sync-off');
        activeIndicator.classList.add('hidden');
        syncFrom('original'); // Force sync state
    } else {
        workspace.classList.add('sync-off');
        activeIndicator.classList.remove('hidden');
        updateActiveIndicator();
    }
});

function setActiveViewer(v) {
    activeViewer = v;
    paneOriginal.classList.toggle('is-active', v === 'original');
    paneSR.classList.toggle('is-active', v === 'sr');
    if (!isSync) updateActiveIndicator();
}

function updateActiveIndicator() {
    activeIndicator.innerText = "ACTIVE: " + (activeViewer === 'original' ? 'ORIGINAL' : 'SUPER-RESOLVED');
}

// Click to set active
paneOriginal.addEventListener('mousedown', () => { if (!isSync && currentMode === 'side') setActiveViewer('original'); });
paneSR.addEventListener('mousedown', () => { if (!isSync && currentMode === 'side') setActiveViewer('sr'); });

// -----------------------------------------
// ZOOM & PAN LOGIC
// -----------------------------------------

function renderTransforms() {
    // Request animation frame is implicit when we call this
    ['original', 'sr'].forEach(v => {
        const s = state[v];
        s.container.style.transform = `translate(${s.panX}px, ${s.panY}px) scale(${s.scale})`;
    });
    
    // Update zoom label based on active viewer (or original if synced)
    const target = (isSync || currentMode !== 'side') ? 'original' : activeViewer;
    zoomLevelText.innerText = Math.round(state[target].scale * 100) + '%';
}

function syncFrom(source) {
    if (!isSync) return;
    const dest = source === 'original' ? 'sr' : 'original';
    
    // We synchronize the VISUAL viewport, not raw absolute pixels, because original and SR might have different native resolutions.
    // E.g. SR might be 4x larger. We match the scaling relative to the image size.
    // If SR is 4x larger, scale of SR should be 1/4 of scale of Original to look the same size on screen?
    // Wait, if both are displayed such that they fill the screen, their bounding boxes match.
    // Let's use relative pan/zoom based on the center of the image.
    
    // Ratio of natural dimensions
    const ratioW = state[dest].w / (state[source].w || 1);
    const ratioH = state[dest].h / (state[source].h || 1);
    
    // Dest scale needs to be adjusted so that it maps to the same screen size
    // Actually, image `width/height` attributes aren't set, we are transforming the natural size.
    // The `img` element takes up its natural width/height in pixels.
    // So if Original is 100x100 and SR is 400x400.
    // To make them look identical on screen, SR scale must be (Original scale) / 4.
    
    const scaleRatio = 1 / ratioW;
    state[dest].scale = state[source].scale * scaleRatio;
    
    // Pan needs to point to the same relative center point.
    // Since images are scaled to match identically, the panning is identical on screen space!
    // Yes! The container's screen-space transform `translate(panX, panY)` applies after the scale, wait...
    // transform is `translate(X, Y) scale(S)`. So X, Y are in screen pixels.
    // If they occupy the same screen space footprint, panX and panY are exactly the same!
    
    state[dest].panX = state[source].panX;
    state[dest].panY = state[source].panY;
    
    renderTransforms();
}

// Fit to screen
function fitToScreen(v) {
    const s = state[v];
    if (s.w === 0 || s.h === 0) return;
    
    const rect = s.pane.getBoundingClientRect();
    if (rect.width === 0) return; // not visible
    
    // Calculate scale to fit inside rect with some padding (10%)
    const padding = 0.9;
    const scaleX = (rect.width * padding) / s.w;
    const scaleY = (rect.height * padding) / s.h;
    s.scale = Math.min(scaleX, scaleY);
    
    // Center it
    s.panX = (rect.width - s.w * s.scale) / 2;
    s.panY = (rect.height - s.h * s.scale) / 2;
    
    renderTransforms();
}

function zoom(v, delta, clientX, clientY) {
    const s = state[v];
    const rect = s.pane.getBoundingClientRect();
    
    // Get cursor position relative to pane
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    // Calculate new scale
    const prevScale = s.scale;
    s.scale *= delta;
    s.scale = Math.max(0.01, Math.min(s.scale, 50)); // limits
    
    // Adjust pan so the point under cursor remains exactly under cursor
    // screenX = panX + imageX * prevScale
    // screenX = newPanX + imageX * newScale
    // newPanX = screenX - (screenX - panX) * (newScale / prevScale)
    s.panX = x - (x - s.panX) * (s.scale / prevScale);
    s.panY = y - (y - s.panY) * (s.scale / prevScale);
    
    if (isSync) syncFrom(v);
    renderTransforms();
}

function pan(v, dx, dy) {
    const s = state[v];
    s.panX += dx;
    s.panY += dy;
    if (isSync) syncFrom(v);
    renderTransforms();
}

// Mouse Wheel Zoom
['original', 'sr'].forEach(v => {
    state[v].pane.addEventListener('wheel', (e) => {
        // If sync is off and we are in Split/Overlay, we need to know which viewer we are interacting with.
        // Actually, in Split/Overlay, pane-original is on bottom.
        // If mode is side, we hover over the specific pane.
        // If mode is split/overlay, just assume we affect the 'activeViewer' or both if synced.
        
        let target = v;
        if (!isSync && currentMode !== 'side') target = activeViewer;
        if (isSync) target = 'original'; // arbitrary source for sync
        
        e.preventDefault();
        const delta = e.deltaY < 0 ? 1.1 : 0.9;
        zoom(target, delta, e.clientX, e.clientY);
    });
});

// Drag Pan
let isDragging = false;
let dragStartX, dragStartY;
let dragTarget = null;

workspace.addEventListener('mousedown', (e) => {
    if (e.target.closest('.split-divider') || e.target.closest('.overlay-controls')) return;
    
    let target = 'original';
    if (currentMode === 'side') {
        if (e.target.closest('#pane-sr')) target = 'sr';
    }
    if (!isSync && currentMode !== 'side') target = activeViewer;
    if (isSync) target = 'original';
    
    isDragging = true;
    dragTarget = target;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    
    if (!isSync && currentMode === 'side') setActiveViewer(target);
});

window.addEventListener('mouseup', () => { isDragging = false; dragTarget = null; });
window.addEventListener('mousemove', (e) => {
    if (isDragging && dragTarget) {
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        pan(dragTarget, dx, dy);
    }
});

// -----------------------------------------
// BOTTOM CONTROLS
// -----------------------------------------

function getTargetViewers() {
    if (isSync || currentMode !== 'side') return isSync ? ['original', 'sr'] : [activeViewer];
    return [activeViewer];
}

document.getElementById('ctrl-zoom-in').onclick = () => {
    const targets = getTargetViewers();
    targets.forEach(v => {
        const s = state[v];
        const rect = s.pane.getBoundingClientRect();
        zoom(v, 1.2, rect.left + rect.width/2, rect.top + rect.height/2);
    });
};
document.getElementById('ctrl-zoom-out').onclick = () => {
    const targets = getTargetViewers();
    targets.forEach(v => {
        const s = state[v];
        const rect = s.pane.getBoundingClientRect();
        zoom(v, 1/1.2, rect.left + rect.width/2, rect.top + rect.height/2);
    });
};
document.getElementById('ctrl-fit').onclick = () => {
    if (isSync) {
        fitToScreen('original'); syncFrom('original');
    } else {
        getTargetViewers().forEach(v => fitToScreen(v));
    }
};
document.getElementById('ctrl-1to1').onclick = () => {
    const targets = getTargetViewers();
    targets.forEach(v => {
        const s = state[v];
        const rect = s.pane.getBoundingClientRect();
        // Calculate point to center around (current center)
        const cx = rect.left + rect.width/2;
        const cy = rect.top + rect.height/2;
        
        // 1:1 scale is 1.0 (actual image pixels)
        const prevScale = s.scale;
        s.scale = 1.0;
        
        // Adjust pan to keep center
        const x = cx - rect.left;
        const y = cy - rect.top;
        s.panX = x - (x - s.panX) * (s.scale / prevScale);
        s.panY = y - (y - s.panY) * (s.scale / prevScale);
        
        if (isSync) syncFrom(v);
        renderTransforms();
    });
};
document.getElementById('ctrl-reset').onclick = () => {
    document.getElementById('ctrl-fit').click();
};

document.getElementById('ctrl-fullscreen').onclick = () => {
    const viewer = document.getElementById('main-viewer');
    viewer.classList.toggle('fullscreen');
    // Re-fit on layout change
    setTimeout(() => document.getElementById('ctrl-fit').click(), 50);
};

// -----------------------------------------
// SPLIT VIEW SLIDER
// -----------------------------------------
let isDraggingSplit = false;
let splitPercent = 50;

function updateSplitClip(percent) {
    splitDivider.style.left = percent + '%';
    // Clip the SR pane so it only shows the RIGHT side of the divider
    // Polygon: top-left, top-right, bottom-right, bottom-left
    paneSR.style.clipPath = `polygon(${percent}% 0, 100% 0, 100% 100%, ${percent}% 100%)`;
}

splitDivider.addEventListener('mousedown', (e) => {
    isDraggingSplit = true;
    e.stopPropagation();
});

window.addEventListener('mousemove', (e) => {
    if (isDraggingSplit && currentMode === 'split') {
        const rect = workspace.getBoundingClientRect();
        let x = e.clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        splitPercent = (x / rect.width) * 100;
        updateSplitClip(splitPercent);
    }
});
window.addEventListener('mouseup', () => { isDraggingSplit = false; });

// -----------------------------------------
// OVERLAY SLIDER
// -----------------------------------------
opacitySlider.addEventListener('input', (e) => {
    if (currentMode === 'overlay') {
        paneSR.style.opacity = e.target.value;
    }
});

// Init
setActiveViewer('original');
