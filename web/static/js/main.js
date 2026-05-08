/* ============================================================
   PlantDoc Disease Detector — Dashboard JS
   ============================================================ */

const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const btnYolo    = document.getElementById('btn-yolo');
const btnVit     = document.getElementById('btn-vit');
const btnBoth    = document.getElementById('btn-both');
const resultsEl  = document.getElementById('results-section');

let currentFile = null;

// ---- Drag & drop ----
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = 'block';
  [btnYolo, btnVit, btnBoth].forEach(b => b.disabled = false);
  resultsEl.innerHTML = '';
}

// ---- Button handlers ----
btnYolo.addEventListener('click', () => runPrediction('/predict/yolo', 'yolo'));
btnVit.addEventListener('click',  () => runPrediction('/predict/vit',  'vit'));
btnBoth.addEventListener('click', () => runPrediction('/predict/both', 'both'));

async function runPrediction(endpoint, mode) {
  if (!currentFile) return;

  setLoading(true);
  resultsEl.innerHTML = '';

  const form = new FormData();
  form.append('file', currentFile);

  try {
    const res = await fetch(endpoint, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Request failed');
    }
    const data = await res.json();
    renderResults(data, mode);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

// ---- Render results ----
function renderResults(data, mode) {
  resultsEl.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'results-grid';

  if (mode === 'yolo' || mode === 'both' && data.yolo) {
    const src = mode === 'both' ? data.yolo : data;
    grid.appendChild(buildYoloCard(src));
  }

  if (mode === 'vit' || mode === 'both' && data.vit) {
    const src = mode === 'both' ? data.vit : data;
    const origSrc = data.original_image || src.original_image;
    grid.appendChild(buildVitCard(src, origSrc));
  }

  resultsEl.appendChild(grid);
  grid.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- YOLO card ----
function buildYoloCard(data) {
  const card = document.createElement('div');
  card.className = 'card';

  const dets = data.detections || [];
  const diseaseClasses = dets.filter(d => !isHealthy(d.cls_name));
  const healthyClasses = dets.filter(d => isHealthy(d.cls_name));

  card.innerHTML = `
    <div class="card-header yolo">
      <div class="model-icon">🎯</div>
      <div>
        <h3>YOLOv11 Detection</h3>
        <small>${dets.length} annotation${dets.length !== 1 ? 's' : ''} found</small>
      </div>
    </div>
    <div class="card-body">
      <div class="result-img-wrap">
        <img src="data:image/jpeg;base64,${data.annotated_image}" alt="YOLO output">
      </div>
      ${dets.length ? buildDetectionList(dets) : '<div class="empty-state"><div class="icon">🌿</div>No detections above threshold</div>'}
    </div>
  `;
  return card;
}

function buildDetectionList(dets) {
  const items = dets.map(d => {
    const hue    = hashColor(d.cls_name);
    const conf   = (d.conf * 100).toFixed(1);
    const tag    = isHealthy(d.cls_name) ? '✅ Healthy' : '⚠️ Disease';
    return `
      <li class="detection-item">
        <span class="dot" style="background:${hue}"></span>
        <span class="cls">${d.cls_name}</span>
        <span class="conf">${conf}%</span>
        <span style="font-size:0.75rem;margin-left:auto">${tag}</span>
      </li>`;
  }).join('');
  return `<ul class="detection-list">${items}</ul>`;
}

// ---- ViT card ----
function buildVitCard(data, originalB64) {
  const card = document.createElement('div');
  card.className = 'card';

  const preds = data.predictions || [];
  const top1  = preds[0] || null;

  card.innerHTML = `
    <div class="card-header vit">
      <div class="model-icon">🔬</div>
      <div>
        <h3>Vision Transformer</h3>
        <small>${top1 ? `Top prediction: ${top1.class_name}` : 'No prediction'}</small>
      </div>
    </div>
    <div class="card-body">
      ${data.attention_image ? buildTabView(data.attention_image, originalB64) : ''}
      <div class="prediction-list" style="margin-top:14px">
        ${preds.map(p => buildPredBar(p)).join('')}
      </div>
      ${buildDiseaseInfo(top1)}
    </div>
  `;

  // Animate bars after DOM insert
  requestAnimationFrame(() => {
    card.querySelectorAll('.bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });

  // Tab switching
  const tabs    = card.querySelectorAll('.tab-btn');
  const panels  = card.querySelectorAll('.tab-panel');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.style.display = 'none');
      tab.classList.add('active');
      card.querySelector(`#panel-${tab.dataset.tab}`).style.display = 'block';
    });
  });

  return card;
}

function buildTabView(attnB64, origB64) {
  const hasOrig = !!origB64;
  return `
    <div class="tab-bar">
      <button class="tab-btn active" data-tab="attn">Attention Map</button>
      ${hasOrig ? `<button class="tab-btn" data-tab="orig">Original</button>` : ''}
    </div>
    <div id="panel-attn" class="tab-panel result-img-wrap">
      <img src="data:image/jpeg;base64,${attnB64}" alt="Attention heatmap">
    </div>
    ${hasOrig ? `
    <div id="panel-orig" class="tab-panel result-img-wrap" style="display:none">
      <img src="data:image/jpeg;base64,${origB64}" alt="Original image">
    </div>` : ''}
  `;
}

function buildPredBar(p) {
  const pct  = (p.confidence * 100).toFixed(1);
  const w    = `${Math.max(p.confidence * 100, 2)}%`;
  return `
    <div class="prediction-item">
      <div class="label-row">
        <span>${p.class_name}</span>
        <span style="color:var(--text-muted)">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:2%" data-width="${w}"></div>
      </div>
    </div>`;
}

// Disease info panel (top prediction only)
const DISEASE_INFO = {
  "Apple Scab Leaf":            "Fungal disease (Venturia inaequalis). Use captan or mancozeb fungicide.",
  "Apple rust leaf":            "Cedar-apple rust (Gymnosporangium). Apply myclobutanil at bud break.",
  "Bell_pepper leaf spot":      "Bacterial spot (Xanthomonas). Use copper-based bactericides.",
  "Corn Gray leaf spot":        "Cercospora fungal disease. Rotate crops; use strobilurin fungicides.",
  "Corn leaf blight":           "Helminthosporium or Erwinia. Plant resistant hybrids.",
  "Corn rust leaf":             "Puccinia fungi. Apply triazole fungicides early.",
  "Potato leaf early blight":   "Alternaria solani. Apply chlorothalonil or azoxystrobin.",
  "Potato leaf late blight":    "Phytophthora infestans. Use mancozeb; destroy infected plants.",
  "Squash Powdery mildew leaf": "Erysiphales. Apply sulfur or potassium bicarbonate spray.",
  "Tomato Early blight leaf":   "Alternaria linariae. Remove lower infected leaves; use copper spray.",
  "Tomato Septoria leaf spot":  "Septoria lycopersici. Apply chlorothalonil fungicide.",
  "Tomato leaf bacterial spot": "Xanthomonas vesicatoria. Copper bactericides; use disease-free seeds.",
  "Tomato leaf late blight":    "Phytophthora infestans. Metalaxyl + mancozeb; remove infected plants.",
  "Tomato leaf mosaic virus":   "TMV — no chemical cure. Remove infected plants; control aphids.",
  "Tomato leaf yellow virus":   "TYLCV spread by whiteflies. Control vector; use resistant varieties.",
  "Tomato mold leaf":           "Cladosporium fulvum. Improve air circulation; apply fungicide.",
  "grape leaf black rot":       "Guignardia bidwellii. Apply myclobutanil at bud swell.",
};

function buildDiseaseInfo(pred) {
  if (!pred) return '';
  const info = DISEASE_INFO[pred.class_name];
  const isHlth = isHealthy(pred.class_name);
  if (!info && !isHlth) return '';
  return `
    <div style="margin-top:16px; padding:14px; border-radius:10px; border:1px solid var(--border);
                background:rgba(255,255,255,0.03); font-size:0.83rem;">
      <div style="font-weight:700; margin-bottom:6px;">
        ${isHlth ? '✅ Healthy leaf — no treatment needed' : `⚠️ ${pred.class_name}`}
      </div>
      ${info ? `<div style="color:var(--text-muted);">${info}</div>` : ''}
    </div>`;
}

// ---- Utilities ----
function isHealthy(name) {
  const n = name.toLowerCase();
  return n.includes('leaf') && !n.includes('spot') && !n.includes('blight') &&
    !n.includes('rust') && !n.includes('mold') && !n.includes('virus') &&
    !n.includes('mildew') && !n.includes('rot') && !n.includes('bacterial');
}

function hashColor(str) {
  let hash = 0;
  for (const ch of str) hash = ch.charCodeAt(0) + ((hash << 5) - hash);
  const h = Math.abs(hash) % 360;
  return `hsl(${h}, 70%, 60%)`;
}

function setLoading(on) {
  [btnYolo, btnVit, btnBoth].forEach(b => {
    b.disabled = on;
    const sp = b.querySelector('.spinner');
    if (sp) sp.style.display = on ? 'block' : 'none';
  });
}

// ---- Demo gallery ----
async function loadDemoResult(imageId) {
  // Highlight active tile
  document.querySelectorAll('.demo-thumb-btn').forEach(b => b.classList.remove('active'));
  const btn = document.querySelector(`.demo-thumb-btn[data-id="${imageId}"]`);
  if (btn) btn.classList.add('active');

  resultsEl.innerHTML = '';
  setLoading(true);
  try {
    const res = await fetch(`/demo/result/${encodeURIComponent(imageId)}`);
    if (!res.ok) throw new Error('Cached result not found');
    const data = await res.json();

    // Determine mode from what's available in cache
    const mode = (data.yolo && data.vit) ? 'both' : data.yolo ? 'yolo' : 'vit';
    renderResults(data, mode);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

function showError(msg) {
  document.querySelectorAll('.error-toast').forEach(e => e.remove());
  const toast = document.createElement('div');
  toast.className = 'error-toast';
  toast.textContent = `Error: ${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}
