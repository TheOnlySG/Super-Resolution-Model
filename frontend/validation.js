const urlParams = new URLSearchParams(window.location.search);
const jobId = urlParams.get('job_id');

if (!jobId) {
    alert("No job_id provided.");
    window.location.href = "/";
}

let hrFile = null;
const mapKeys = ['confidence', 'reconstruction_error', 'sam_error', 'ndvi_error'];

// 1. Comparison Viewer States (3 independent panes: lr, sr, hr)
const compStates = {
    lr: { scale: 1, panX: 0, panY: 0, w: 0, h: 0 },
    sr: { scale: 1, panX: 0, panY: 0, w: 0, h: 0 },
    hr: { scale: 1, panX: 0, panY: 0, w: 0, h: 0 }
};
let activeCompPane = null;
let cStartX = 0, cStartY = 0;

// Setup image onload listeners to get exact natural dimensions for pixel-perfect fitting
const lrImgEl = document.getElementById('comp-lr-img');
const srImgEl = document.getElementById('comp-sr-img');
const hrImgEl = document.getElementById('comp-hr-img');

lrImgEl.onload = () => {
    compStates.lr.w = lrImgEl.naturalWidth;
    compStates.lr.h = lrImgEl.naturalHeight;
    fitCompViewer('lr');
};

srImgEl.onload = () => {
    compStates.sr.w = srImgEl.naturalWidth;
    compStates.sr.h = srImgEl.naturalHeight;
    fitCompViewer('sr');
};

hrImgEl.onload = () => {
    compStates.hr.w = hrImgEl.naturalWidth;
    compStates.hr.h = hrImgEl.naturalHeight;
    fitCompViewer('hr');
};

// Load initial LR and SR images from DB (key is 'current_job')
let srJobData = null;
const request = indexedDB.open("sr_db", 1);
request.onsuccess = (event) => {
    const db = event.target.result;
    if (!db.objectStoreNames.contains("jobs")) return;
    const transaction = db.transaction(["jobs"], "readonly");
    const store = transaction.objectStore("jobs");
    const getReq = store.get('current_job');
    getReq.onsuccess = () => {
        if (getReq.result) {
            srJobData = getReq.result;
            const lrBase64 = srJobData.visualizations?.lr_rgb || srJobData.preview;
            const srBase64 = srJobData.visualizations?.sr_rgb;

            if (lrBase64) {
                lrImgEl.src = "data:image/png;base64," + lrBase64;
            } else {
                const emptyEl = document.getElementById('comp-lr-empty');
                if (emptyEl) {
                    emptyEl.innerText = "LR preview unavailable for this job.";
                    emptyEl.classList.remove('hidden');
                }
            }
            if (srBase64) srImgEl.src = "data:image/png;base64," + srBase64;
        }
    };
};

// Mouse & Wheel events for each comparison pane
document.querySelectorAll('.comparison-pane').forEach(pane => {
    const key = pane.dataset.pane;
    if (!key) return;

    pane.addEventListener('mousedown', (e) => {
        // Prevent drag initiation if clicking a button inside pane-controls
        if (e.target.closest('.pane-controls')) return;
        activeCompPane = key;
        const state = compStates[key];
        cStartX = e.clientX - state.panX;
        cStartY = e.clientY - state.panY;
    });

    pane.addEventListener('wheel', (e) => {
        e.preventDefault();
        const state = compStates[key];
        if (!state || !state.w) return;
        const rect = pane.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const delta = e.deltaY < 0 ? 1.1 : 0.9;
        const prevScale = state.scale;
        state.scale = Math.max(0.01, Math.min(state.scale * delta, 50));

        state.panX = x - (x - state.panX) * (state.scale / prevScale);
        state.panY = y - (y - state.panY) * (state.scale / prevScale);
        renderComp(key);
    });
});

function fitCompViewer(key) {
    const state = compStates[key];
    if (!state || !state.w) return;
    const pane = document.getElementById(`comp-${key}`);
    if (!pane) return;
    const rect = pane.getBoundingClientRect();
    const padding = 0.9;
    const scaleX = (rect.width * padding) / state.w;
    const scaleY = (rect.height * padding) / state.h;

    state.scale = Math.min(scaleX, scaleY);
    state.panX = (rect.width - state.w * state.scale) / 2;
    state.panY = (rect.height - state.h * state.scale) / 2;
    renderComp(key);
}

function resetCompViewer(key) {
    const state = compStates[key];
    if (!state || !state.w) return;
    const pane = document.getElementById(`comp-${key}`);
    if (!pane) return;
    const rect = pane.getBoundingClientRect();

    state.scale = 1;
    state.panX = (rect.width - state.w) / 2;
    state.panY = (rect.height - state.h) / 2;
    renderComp(key);
}

function renderComp(key) {
    const state = compStates[key];
    if (!state) return;
    const container = document.getElementById(`comp-${key}-container`);
    if (container) {
        container.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
    }
    const zoomText = document.getElementById(`zoom-comp-${key}`);
    if (zoomText) {
        zoomText.innerText = Math.round(state.scale * 100) + "%";
    }
}

// Comparison pane FIT and RESET buttons
document.querySelectorAll('.comp-fit-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        fitCompViewer(e.target.dataset.target);
    });
});
document.querySelectorAll('.comp-reset-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetCompViewer(e.target.dataset.target);
    });
});

// Fullscreen toggle for comparison panes and error cards
document.querySelectorAll('.fullscreen-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const targetId = e.target.dataset.target;
        toggleFullscreen(targetId);
    });
});

function toggleFullscreen(targetId) {
    const el = document.getElementById(targetId);
    if (!el) return;

    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else if (el.requestFullscreen) {
        el.requestFullscreen().catch(() => el.classList.toggle('fullscreen-mode'));
    } else {
        el.classList.toggle('fullscreen-mode');
    }
}

window.addEventListener('resize', () => {
    ['lr', 'sr', 'hr'].forEach(fitCompViewer);
    mapKeys.forEach(fitGrid);
});
document.addEventListener('fullscreenchange', () => {
    setTimeout(() => {
        ['lr', 'sr', 'hr'].forEach(fitCompViewer);
        mapKeys.forEach(fitGrid);
    }, 150);
});

// 2. Upload HR
const dropZone = document.getElementById('hr-drop-zone');
const fileInput = document.getElementById('hr-file-input');
const startBtn = document.getElementById('validate-start-btn');
const errorText = document.getElementById('upload-error');

dropZone.addEventListener('click', (e) => { if (e.target !== fileInput) fileInput.click(); });
fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) handleFile(e.target.files[0]);
});
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
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.tif') && !name.endsWith('.tiff')) {
        errorText.innerText = "Only .tif or .tiff supported.";
        errorText.classList.remove('hidden');
        hrFile = null;
        startBtn.disabled = true;
        return;
    }
    hrFile = file;
    errorText.classList.add('hidden');
    dropZone.querySelector('p').textContent = file.name;
    startBtn.disabled = false;
}

// 3. Validation execution
startBtn.addEventListener('click', async () => {
    if (!hrFile) return;
    startBtn.disabled = true;
    startBtn.innerText = "Processing...";

    const formData = new FormData();
    formData.append('hr_file', hrFile);
    formData.append('job_id', jobId);

    try {
        const response = await fetch('/validate', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Validation failed.");
        populateUI(data);
        startBtn.innerText = "Validation Complete";
    } catch (err) {
        alert(err.message);
        startBtn.disabled = false;
        startBtn.innerText = "Run Validation";
    }
});

function populateUI(data) {
    // Show HR preview in third pane
    document.getElementById('comp-hr-empty').classList.add('hidden');
    hrImgEl.src = "data:image/png;base64," + data.maps.hr_rgb;
    hrImgEl.classList.remove('hidden');

    // Populate metrics
    document.getElementById('metrics-panel').classList.remove('hidden');
    document.getElementById('val-psnr').innerText = data.metrics.psnr.toFixed(2);
    document.getElementById('val-ssim').innerText = data.metrics.ssim.toFixed(4);
    document.getElementById('val-sam').innerText = data.metrics.sam_mean_degrees.toFixed(2) + "°";
    document.getElementById('val-ndvi').innerText = data.metrics.ndvi_mean_error.toFixed(4);
    document.getElementById('val-rmse').innerText = data.metrics.rmse.toFixed(4);
    document.getElementById('val-conf').innerText = (data.metrics.confidence_mean * 100).toFixed(1) + "%";

    // Grid maps
    document.getElementById('error-grid').classList.remove('hidden');

    mapKeys.forEach(key => {
        const img = document.getElementById(`img-${key}`);
        if (img && data.maps[key]) {
            img.onload = () => {
                if (gridStates[key]) {
                    gridStates[key].w = img.naturalWidth;
                    gridStates[key].h = img.naturalHeight;
                    fitGrid(key);
                }
            };
            img.src = "data:image/png;base64," + data.maps[key];
        }
        gridStates[key] = {
            scale: 1, panX: 0, panY: 0,
            w: data.dimensions.width, h: data.dimensions.height,
            container: document.getElementById(`container-${key}`),
            zoomText: document.getElementById(`zoom-${key}`)
        };
        setTimeout(() => fitGrid(key), 100);
    });
}

// 4. Grid Viewers Logic
const gridStates = {};
let activeGrid = null;
let gStartX, gStartY;

document.querySelectorAll('.error-card-viewer').forEach(viewer => {
    viewer.addEventListener('mousedown', (e) => {
        activeGrid = viewer.dataset.map;
        if (!gridStates[activeGrid]) return;
        gStartX = e.clientX - gridStates[activeGrid].panX;
        gStartY = e.clientY - gridStates[activeGrid].panY;
    });
    viewer.addEventListener('wheel', (e) => {
        e.preventDefault();
        const key = viewer.dataset.map;
        if (!gridStates[key]) return;
        const state = gridStates[key];
        const rect = viewer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const delta = e.deltaY < 0 ? 1.1 : 0.9;
        const prevScale = state.scale;
        state.scale = Math.max(0.01, Math.min(state.scale * delta, 50));

        state.panX = x - (x - state.panX) * (state.scale / prevScale);
        state.panY = y - (y - state.panY) * (state.scale / prevScale);
        renderGrid(key);
    });
});

// Single consolidated mouse move and mouse up listeners for both drag systems
window.addEventListener('mouseup', () => {
    activeCompPane = null;
    activeGrid = null;
});

window.addEventListener('mousemove', (e) => {
    if (activeCompPane && compStates[activeCompPane]) {
        const state = compStates[activeCompPane];
        state.panX = e.clientX - cStartX;
        state.panY = e.clientY - cStartY;
        renderComp(activeCompPane);
    } else if (activeGrid && gridStates[activeGrid]) {
        const state = gridStates[activeGrid];
        state.panX = e.clientX - gStartX;
        state.panY = e.clientY - gStartY;
        renderGrid(activeGrid);
    }
});

function fitGrid(key) {
    const state = gridStates[key];
    if (!state || !state.w) return;
    const viewer = document.getElementById(`viewer-${key}`);
    const rect = viewer.getBoundingClientRect();
    const padding = 0.9;
    const scaleX = (rect.width * padding) / state.w;
    const scaleY = (rect.height * padding) / state.h;

    state.scale = Math.min(scaleX, scaleY);
    state.panX = (rect.width - state.w * state.scale) / 2;
    state.panY = (rect.height - state.h * state.scale) / 2;
    renderGrid(key);
}

function renderGrid(key) {
    const state = gridStates[key];
    if (!state) return;
    state.container.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
    state.zoomText.innerText = Math.round(state.scale * 100) + "%";
}

document.querySelectorAll('.fit-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        fitGrid(e.target.dataset.target);
    });
});
document.querySelectorAll('.reset-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const key = e.target.dataset.target;
        const state = gridStates[key];
        if (!state) return;
        const viewer = document.getElementById(`viewer-${key}`);
        const rect = viewer.getBoundingClientRect();
        state.scale = 1;
        state.panX = (rect.width - state.w) / 2;
        state.panY = (rect.height - state.h) / 2;
        renderGrid(key);
    });
});

// Info Modal Handlers for Validation Page
const infoModalBackdrop = document.getElementById('info-modal-backdrop');
const infoModalTitle = document.getElementById('info-modal-title');
const infoModalBody = document.getElementById('info-modal-body');
const infoModalClose = document.getElementById('info-modal-close');

if (infoModalBackdrop && infoModalClose) {
    document.querySelectorAll('.info-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const title = btn.dataset.infoTitle || "Information";
            const content = btn.dataset.infoContent || "";
            infoModalTitle.innerText = title;
            infoModalBody.innerText = content;
            infoModalBackdrop.classList.remove('hidden');
        });
    });

    infoModalClose.addEventListener('click', () => {
        infoModalBackdrop.classList.add('hidden');
    });

    infoModalBackdrop.addEventListener('click', (e) => {
        if (e.target === infoModalBackdrop) infoModalBackdrop.classList.add('hidden');
    });
}
