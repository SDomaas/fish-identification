#!/usr/bin/env python3
"""
bbox_editor.py — Interactive bounding box editor for YOLO detection_frames.

Serves a web UI where you can:
  - Browse images in a detection_frames folder
  - View, draw, resize, and delete bounding boxes
  - Save changes back to YOLO .txt label files
  - Mark images as done (skipped from default view)

Usage
-----
python scripts/bbox_editor.py \
    --images /data/.../training_crops/detection_frames \
    --port 5000

Then open http://localhost:5000 in your browser.

Controls
--------
  Draw    : Click and drag on empty area
  Select  : Click an existing box
  Delete  : Select a box then press Delete or Backspace
  Save    : Ctrl+S  or  Save button
  Navigate: Arrow keys ← → or Prev/Next buttons
  Filter  : Show only unlabelled / all images via dropdown
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_file, render_template_string

app = Flask(__name__)
IMAGES_DIR: Path = None
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BBox Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: sans-serif; background: #1a1a2e; color: #eee; display: flex; flex-direction: column; height: 100vh; }
#toolbar { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #16213e; flex-shrink: 0; flex-wrap: wrap; }
#toolbar h1 { font-size: 14px; color: #a0c4ff; margin-right: 8px; }
button { padding: 5px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
#btnPrev, #btnNext { background: #0f3460; color: #eee; }
#btnSave { background: #e94560; color: #fff; font-weight: bold; }
#btnDelete { background: #555; color: #fff; }
#btnClear { background: #333; color: #fff; }
select { padding: 5px; border-radius: 4px; background: #0f3460; color: #eee; border: none; font-size: 13px; }
#imgLabel { font-size: 12px; color: #a0c4ff; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
#counter { font-size: 12px; color: #aaa; }
#status { font-size: 12px; color: #6fcf97; min-width: 80px; }
#main { display: flex; flex: 1; overflow: hidden; }
#sidebar { width: 200px; background: #16213e; overflow-y: auto; flex-shrink: 0; border-right: 1px solid #333; }
#sidebar div { padding: 6px 10px; font-size: 12px; cursor: pointer; border-bottom: 1px solid #222; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#sidebar div:hover { background: #0f3460; }
#sidebar div.active { background: #e94560; color: #fff; }
#sidebar div.done { color: #6fcf97; }
#canvasWrap { flex: 1; overflow: auto; display: flex; align-items: flex-start; justify-content: flex-start; background: #111; }
canvas { cursor: crosshair; display: block; }
#info { position: fixed; bottom: 8px; right: 12px; font-size: 11px; color: #555; }
</style>
</head>
<body>
<div id="toolbar">
  <h1>BBox Editor</h1>
  <button id="btnPrev">◀ Prev</button>
  <button id="btnNext">Next ▶</button>
  <button id="btnSave">💾 Save (Ctrl+S)</button>
  <button id="btnDelete">🗑 Delete box</button>
  <button id="btnClear">✖ Clear all</button>
  <select id="filterSel">
    <option value="all">All images</option>
    <option value="unlabelled">Unlabelled only</option>
    <option value="labelled">Labelled only</option>
  </select>
  <span id="imgLabel">—</span>
  <span id="counter"></span>
  <span id="status"></span>
</div>
<div id="main">
  <div id="sidebar"></div>
  <div id="canvasWrap"><canvas id="c"></canvas></div>
</div>
<div id="info">Draw: drag | Select: click box | Delete: Del key | Save: Ctrl+S</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let images = [], filteredImages = [], currentIdx = 0;
let boxes = [];       // [{x,y,w,h}] normalised 0-1, class always 0
let img = new Image();
let imgW = 1, imgH = 1;
let scale = 1;
let drawing = false, startX, startY, dragBox = null;
let selectedBox = -1;
let dirty = false;
let resizing = false, resizeHandle = null;
const HANDLE = 8;

function setStatus(msg, color='#6fcf97') {
  const s = document.getElementById('status');
  s.style.color = color;
  s.textContent = msg;
  if (msg) setTimeout(() => { if (s.textContent === msg) s.textContent = ''; }, 2000);
}

async function loadImageList() {
  const filter = document.getElementById('filterSel').value;
  const res = await fetch('/api/images?filter=' + filter);
  filteredImages = await res.json();
  renderSidebar();
  if (filteredImages.length > 0) {
    currentIdx = 0;
    loadImage(filteredImages[currentIdx]);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('imgLabel').textContent = 'No images';
  }
}

function renderSidebar() {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  filteredImages.forEach((name, i) => {
    const d = document.createElement('div');
    d.textContent = name;
    d.title = name;
    if (i === currentIdx) d.classList.add('active');
    d.addEventListener('click', () => { currentIdx = i; loadImage(filteredImages[i]); });
    sb.appendChild(d);
  });
  document.getElementById('counter').textContent = `${currentIdx + 1} / ${filteredImages.length}`;
}

async function loadImage(name) {
  if (dirty) {
    const ok = confirm('Unsaved changes — save before leaving?');
    if (ok) await saveBoxes();
  }
  dirty = false;
  selectedBox = -1;
  document.getElementById('imgLabel').textContent = name;
  renderSidebar();

  const res = await fetch('/api/labels/' + encodeURIComponent(name));
  boxes = await res.json();

  img = new Image();
  img.onload = () => {
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    const wrap = document.getElementById('canvasWrap');
    const maxW = wrap.clientWidth - 20;
    const maxH = wrap.clientHeight - 20;
    scale = Math.min(maxW / imgW, maxH / imgH, 1);
    canvas.width  = Math.round(imgW * scale);
    canvas.height = Math.round(imgH * scale);
    draw();
  };
  img.src = '/api/image/' + encodeURIComponent(name) + '?t=' + Date.now();
}

function toCanvas(nx, ny) { return [nx * imgW * scale, ny * imgH * scale]; }
function toNorm(cx, cy)   { return [cx / (imgW * scale), cy / (imgH * scale)]; }

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  boxes.forEach((b, i) => {
    const [x1, y1] = toCanvas(b.x, b.y);
    const [x2, y2] = toCanvas(b.x + b.w, b.y + b.h);
    ctx.strokeStyle = i === selectedBox ? '#ff0' : '#0f0';
    ctx.lineWidth = i === selectedBox ? 2.5 : 1.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (i === selectedBox) {
      // resize handles
      [[x1,y1],[x2,y1],[x1,y2],[x2,y2]].forEach(([hx,hy]) => {
        ctx.fillStyle = '#ff0';
        ctx.fillRect(hx - HANDLE/2, hy - HANDLE/2, HANDLE, HANDLE);
      });
    }
  });
  if (dragBox) {
    ctx.strokeStyle = '#f80';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(dragBox.x, dragBox.y, dragBox.w, dragBox.h);
  }
}

function hitBox(cx, cy) {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const [x1, y1] = toCanvas(boxes[i].x, boxes[i].y);
    const [x2, y2] = toCanvas(boxes[i].x + boxes[i].w, boxes[i].y + boxes[i].h);
    if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) return i;
  }
  return -1;
}

function hitHandle(cx, cy) {
  if (selectedBox < 0) return null;
  const b = boxes[selectedBox];
  const [x1, y1] = toCanvas(b.x, b.y);
  const [x2, y2] = toCanvas(b.x + b.w, b.y + b.h);
  const corners = [{cx:x1,cy:y1,name:'tl'},{cx:x2,cy:y1,name:'tr'},{cx:x1,cy:y2,name:'bl'},{cx:x2,cy:y2,name:'br'}];
  for (const h of corners) {
    if (Math.abs(cx - h.cx) <= HANDLE && Math.abs(cy - h.cy) <= HANDLE) return h.name;
  }
  return null;
}

canvas.addEventListener('mousedown', e => {
  const r = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const handle = hitHandle(cx, cy);
  if (handle) {
    resizing = true; resizeHandle = handle;
    startX = cx; startY = cy;
    return;
  }
  const hit = hitBox(cx, cy);
  if (hit >= 0) { selectedBox = hit; draw(); return; }
  selectedBox = -1;
  drawing = true;
  startX = cx; startY = cy;
  dragBox = { x: cx, y: cy, w: 0, h: 0 };
});

canvas.addEventListener('mousemove', e => {
  const r = canvas.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  if (resizing && selectedBox >= 0) {
    const b = boxes[selectedBox];
    const [nx, ny] = toNorm(cx, cy);
    if (resizeHandle === 'tl') { const ox = b.x + b.w, oy = b.y + b.h; b.x = Math.min(nx, ox); b.y = Math.min(ny, oy); b.w = Math.abs(ox - nx); b.h = Math.abs(oy - ny); }
    if (resizeHandle === 'tr') { const oy = b.y + b.h; b.y = Math.min(ny, oy); b.w = Math.abs(nx - b.x); b.h = Math.abs(oy - ny); }
    if (resizeHandle === 'bl') { const ox = b.x + b.w; b.x = Math.min(nx, ox); b.w = Math.abs(ox - nx); b.h = Math.abs(ny - b.y); }
    if (resizeHandle === 'br') { b.w = Math.abs(nx - b.x); b.h = Math.abs(ny - b.y); }
    dirty = true; draw(); return;
  }
  if (drawing) {
    dragBox = { x: Math.min(cx, startX), y: Math.min(cy, startY), w: Math.abs(cx - startX), h: Math.abs(cy - startY) };
    draw();
  }
});

canvas.addEventListener('mouseup', e => {
  if (resizing) { resizing = false; resizeHandle = null; return; }
  if (!drawing) return;
  drawing = false;
  if (dragBox && dragBox.w > 5 && dragBox.h > 5) {
    const [nx, ny] = toNorm(dragBox.x, dragBox.y);
    const [nw, nh] = toNorm(dragBox.w, dragBox.h);
    boxes.push({ x: Math.max(0, nx), y: Math.max(0, ny), w: Math.min(nw, 1 - nx), h: Math.min(nh, 1 - ny) });
    selectedBox = boxes.length - 1;
    dirty = true;
  }
  dragBox = null;
  draw();
});

async function saveBoxes() {
  const name = filteredImages[currentIdx];
  await fetch('/api/labels/' + encodeURIComponent(name), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(boxes),
  });
  dirty = false;
  setStatus('Saved ✓');
}

function deleteSelected() {
  if (selectedBox < 0) return;
  boxes.splice(selectedBox, 1);
  selectedBox = -1;
  dirty = true;
  draw();
}

document.getElementById('btnSave').addEventListener('click', saveBoxes);
document.getElementById('btnDelete').addEventListener('click', deleteSelected);
document.getElementById('btnClear').addEventListener('click', () => { boxes = []; selectedBox = -1; dirty = true; draw(); });
document.getElementById('btnPrev').addEventListener('click', () => {
  if (currentIdx > 0) { currentIdx--; loadImage(filteredImages[currentIdx]); }
});
document.getElementById('btnNext').addEventListener('click', () => {
  if (currentIdx < filteredImages.length - 1) { currentIdx++; loadImage(filteredImages[currentIdx]); }
});
document.getElementById('filterSel').addEventListener('change', loadImageList);

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveBoxes(); }
  if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();
  if (e.key === 'ArrowRight') document.getElementById('btnNext').click();
  if (e.key === 'ArrowLeft') document.getElementById('btnPrev').click();
});

loadImageList();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/images")
def list_images():
    filter_mode = request.args.get("filter", "all")
    images = sorted(
        p.name for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in EXTENSIONS
    )
    if filter_mode == "unlabelled":
        images = [n for n in images if not _has_labels(n)]
    elif filter_mode == "labelled":
        images = [n for n in images if _has_labels(n)]
    return jsonify(images)


def _has_labels(name: str) -> bool:
    txt = IMAGES_DIR / (Path(name).stem + ".txt")
    if not txt.exists():
        return False
    return txt.stat().st_size > 0


@app.route("/api/image/<name>")
def get_image(name):
    p = IMAGES_DIR / name
    if not p.exists():
        return "not found", 404
    return send_file(str(p))


@app.route("/api/labels/<name>", methods=["GET"])
def get_labels(name):
    txt = IMAGES_DIR / (Path(name).stem + ".txt")
    boxes = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                boxes.append({"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h})
    return jsonify(boxes)


@app.route("/api/labels/<name>", methods=["POST"])
def save_labels(name):
    boxes = request.get_json()
    txt = IMAGES_DIR / (Path(name).stem + ".txt")
    lines = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cx = x + w / 2
        cy = y + h / 2
        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    return jsonify({"ok": True})


def parse_args():
    p = argparse.ArgumentParser(description="Interactive bounding box editor for YOLO detection_frames.")
    p.add_argument("--images", required=True, help="Folder containing images and .txt label files.")
    p.add_argument("--port", type=int, default=5000, help="Port to serve on (default: 5000).")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0).")
    return p.parse_args()


def main():
    global IMAGES_DIR
    args = parse_args()
    IMAGES_DIR = Path(args.images)
    if not IMAGES_DIR.exists():
        print(f"Error: images folder not found: {IMAGES_DIR}")
        sys.exit(1)
    print(f"\n🐟 BBox Editor")
    print(f"   Images : {IMAGES_DIR}")
    print(f"   Open   : http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
