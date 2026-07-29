/*
 * Mermaid Flowchart Editor
 * ------------------------
 * A dependency-free front end around mermaid-js (https://github.com/mermaid-js/mermaid).
 * Everything runs in the browser: diagrams are stored in localStorage, and exports
 * (PNG / SVG / .mmd / shareable link) are produced client side. No account, no server.
 */

/* ------------------------------------------------------------------ *
 * Loading mermaid
 * ------------------------------------------------------------------ */

// Tried in order. The first entry lets you work fully offline once
// ./download-offline-copy.sh has been run; the CDNs are the fallback.
const MERMAID_SOURCES = [
  './vendor/mermaid.min.js',
  'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js',
  'https://unpkg.com/mermaid@11/dist/mermaid.min.js',
];

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.async = true;
    el.onload = () => (window.mermaid ? resolve(src) : reject(new Error(`no mermaid global from ${src}`)));
    el.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(el);
  });
}

async function loadMermaid() {
  const failures = [];
  for (const src of MERMAID_SOURCES) {
    try {
      await loadScript(src);
      return { mermaid: window.mermaid, src };
    } catch (err) {
      failures.push(err.message);
    }
  }
  throw new Error(`Could not load Mermaid.\n${failures.join('\n')}`);
}

/* ------------------------------------------------------------------ *
 * Storage
 * ------------------------------------------------------------------ */

const STORE_KEY = 'mermaid-editor.v1';

const defaultStore = () => ({
  docs: [],
  draft: null,          // unsaved working buffer: { id, name, code }
  settings: { mermaidTheme: 'default', uiTheme: 'light', editorWidth: 42, sidebarHidden: false },
});

function readStore() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return defaultStore();
    const parsed = JSON.parse(raw);
    return {
      ...defaultStore(),
      ...parsed,
      docs: Array.isArray(parsed.docs) ? parsed.docs : [],
      settings: { ...defaultStore().settings, ...(parsed.settings || {}) },
    };
  } catch {
    return defaultStore();
  }
}

function writeStore(next) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(next));
    return true;
  } catch (err) {
    toast('Could not save — browser storage is full or blocked.', true);
    console.error(err);
    return false;
  }
}

let store = readStore();

const persist = () => writeStore(store);
const newId = () => `d${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;

/* ------------------------------------------------------------------ *
 * Templates
 * ------------------------------------------------------------------ */

const TEMPLATES = {
  'Flowchart (top-down)': `flowchart TD
    A[Start] --> B{Is it working?}
    B -- Yes --> C[Ship it]
    B -- No --> D[Debug]
    D --> B
    C --> E[End]`,

  'Flowchart (left-right, grouped)': `flowchart LR
    subgraph Collect
        A[Raw data] --> B[Clean]
    end
    subgraph Analyse
        B --> C[Model]
        C --> D{Fit good?}
    end
    D -- No --> C
    D -- Yes --> E[(Results)]
    E --> F[Write up]`,

  'Flowchart (shapes cheat sheet)': `flowchart TD
    A[Rectangle] --> B(Rounded)
    B --> C([Stadium])
    C --> D[[Subroutine]]
    D --> E[(Database)]
    E --> F((Circle))
    F --> G{Diamond}
    G --> H{{Hexagon}}
    H --> I[/Parallelogram/]
    I --> J[\\Trapezoid/]`,

  'Sequence diagram': `sequenceDiagram
    participant U as User
    participant S as Server
    participant D as Database
    U->>S: Request page
    activate S
    S->>D: Query
    D-->>S: Rows
    S-->>U: HTML
    deactivate S
    Note over U,S: Round trip complete`,

  'Class diagram': `classDiagram
    class Animal {
        +String name
        +int age
        +speak() void
    }
    class Dog {
        +fetch() void
    }
    class Cat {
        +scratch() void
    }
    Animal <|-- Dog
    Animal <|-- Cat`,

  'State diagram': `stateDiagram-v2
    [*] --> Idle
    Idle --> Running : start
    Running --> Paused : pause
    Paused --> Running : resume
    Running --> [*] : finish`,

  'Entity relationship': `erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE_ITEM : contains
    PRODUCT ||--o{ LINE_ITEM : "ordered in"
    CUSTOMER {
        string name
        string email
    }`,

  'Gantt chart': `gantt
    title Project timeline
    dateFormat YYYY-MM-DD
    axisFormat %b %d
    section Fieldwork
        Data collection   :done,    a1, 2026-01-05, 30d
        Cleaning          :active,  a2, after a1, 14d
    section Analysis
        Modelling         :         a3, after a2, 21d
        Write up          :         a4, after a3, 28d`,

  'Pie chart': `pie title Time spent
    "Reading" : 25
    "Analysis" : 40
    "Writing" : 30
    "Coffee" : 5`,

  'Mind map': `mindmap
  root((Thesis))
    Background
      Literature
      Gaps
    Methods
      Data
      Models
    Results
    Discussion`,

  'Git graph': `gitGraph
    commit id: "init"
    branch feature
    commit
    commit
    checkout main
    commit
    merge feature
    commit id: "release"`,
};

const STARTER = TEMPLATES['Flowchart (top-down)'];

/* ------------------------------------------------------------------ *
 * Element handles
 * ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

const el = {
  html: document.documentElement,
  layout: $('layout'),
  sidebar: $('sidebar'),
  toggleSidebar: $('toggleSidebar'),
  docTitle: $('docTitle'),
  dirtyDot: $('dirtyDot'),
  btnNew: $('btnNew'),
  btnSave: $('btnSave'),
  btnExport: $('btnExport'),
  exportMenu: $('exportMenu'),
  btnMore: $('btnMore'),
  moreMenu: $('moreMenu'),
  mermaidTheme: $('mermaidTheme'),
  btnUiTheme: $('btnUiTheme'),
  searchBox: $('searchBox'),
  diagramList: $('diagramList'),
  storageInfo: $('storageInfo'),
  templateSelect: $('templateSelect'),
  code: $('code'),
  gutter: $('gutter'),
  editorPane: $('editorPane'),
  splitter: $('splitter'),
  viewport: $('viewport'),
  stage: $('stage'),
  loading: $('loading'),
  errorPanel: $('errorPanel'),
  errorText: $('errorText'),
  status: $('status'),
  lineInfo: $('lineInfo'),
  zoomIn: $('zoomIn'),
  zoomOut: $('zoomOut'),
  zoomReset: $('zoomReset'),
  zoomLabel: $('zoomLabel'),
  infoDialog: $('infoDialog'),
  infoTitle: $('infoTitle'),
  infoBody: $('infoBody'),
  filePicker: $('filePicker'),
  backupPicker: $('backupPicker'),
  toast: $('toast'),
};

/* ------------------------------------------------------------------ *
 * Session state
 * ------------------------------------------------------------------ */

const state = {
  mermaid: null,
  currentId: null,      // id of the saved diagram being edited, or null for a new one
  dirty: false,
  lastGoodSvg: '',
  renderSeq: 0,
  view: { scale: 1, x: 0, y: 0 },
  fitPending: true,
};

/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */

let toastTimer = null;
function toast(message, isError = false) {
  el.toast.textContent = message;
  el.toast.classList.toggle('bad', isError);
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, isError ? 4200 : 2200);
}

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.className = isError ? 'bad' : 'ok';
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function formatDate(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
}

function safeFileName(name) {
  return (name || 'diagram').replace(/[^\w\-. ]+/g, '_').replace(/\s+/g, '-').slice(0, 80) || 'diagram';
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard API needs a secure context; fall back to a hidden textarea.
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  }
}

/* ------------------------------------------------------------------ *
 * Shareable links  (#z=deflated | #b=plain, both base64url)
 * ------------------------------------------------------------------ */

const b64urlEncode = (bytes) => {
  let bin = '';
  bytes.forEach((b) => { bin += String.fromCharCode(b); });
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};

const b64urlDecode = (str) => {
  const padded = str.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((str.length + 3) % 4);
  const bin = atob(padded);
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
};

async function encodeShare(code) {
  const bytes = new TextEncoder().encode(code);
  if (typeof CompressionStream === 'function') {
    try {
      const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
      const packed = new Uint8Array(await new Response(stream).arrayBuffer());
      return `z=${b64urlEncode(packed)}`;
    } catch { /* fall through to the uncompressed form */ }
  }
  return `b=${b64urlEncode(bytes)}`;
}

async function decodeShare(hash) {
  const match = /^#?([zb])=(.+)$/.exec(hash);
  if (!match) return null;
  const [, kind, payload] = match;
  const bytes = b64urlDecode(payload);
  if (kind === 'b') return new TextDecoder().decode(bytes);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new TextDecoder().decode(await new Response(stream).arrayBuffer());
}

/* ------------------------------------------------------------------ *
 * Editor: gutter, indentation, cursor readout
 * ------------------------------------------------------------------ */

let errorLine = 0;

function refreshGutter() {
  const lines = el.code.value.split('\n').length;
  const parts = [];
  for (let i = 1; i <= lines; i += 1) {
    parts.push(i === errorLine ? `<span class="err-line">${i}</span>` : String(i));
  }
  el.gutter.innerHTML = parts.join('\n');
  el.gutter.scrollTop = el.code.scrollTop;
}

function refreshCursorInfo() {
  const upto = el.code.value.slice(0, el.code.selectionStart);
  const line = upto.split('\n').length;
  const col = upto.length - upto.lastIndexOf('\n');
  el.lineInfo.textContent = `Ln ${line}, Col ${col}`;
}

function handleEditorKeys(event) {
  const ta = el.code;

  if (event.key === 'Tab') {
    event.preventDefault();
    const { selectionStart: start, selectionEnd: end, value } = ta;
    const lineStart = value.lastIndexOf('\n', start - 1) + 1;

    if (start !== end || event.shiftKey) {
      // Indent or dedent every line touched by the selection.
      const lineEnd = value.indexOf('\n', end) === -1 ? value.length : value.indexOf('\n', end);
      const block = value.slice(lineStart, lineEnd);
      const shifted = event.shiftKey
        ? block.replace(/^ {1,2}/gm, '')
        : block.replace(/^/gm, '  ');
      ta.setRangeText(shifted, lineStart, lineEnd, 'select');
    } else {
      ta.setRangeText('  ', start, end, 'end');
    }
    onCodeChanged();
    return;
  }

  if (event.key === 'Enter') {
    // Keep the indentation of the current line on the new one.
    const { selectionStart: start, value } = ta;
    const lineStart = value.lastIndexOf('\n', start - 1) + 1;
    const indent = (/^[ \t]*/.exec(value.slice(lineStart, start)) || [''])[0];
    if (indent) {
      event.preventDefault();
      ta.setRangeText(`\n${indent}`, ta.selectionStart, ta.selectionEnd, 'end');
      onCodeChanged();
    }
  }
}

/* ------------------------------------------------------------------ *
 * Rendering
 * ------------------------------------------------------------------ */

function configureMermaid() {
  state.mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: store.settings.mermaidTheme,
    // SVG <text> labels instead of embedded HTML: this is what makes PNG and
    // SVG exports faithful, since browsers refuse to rasterise foreignObject.
    htmlLabels: false,
    flowchart: { htmlLabels: false, useMaxWidth: false },
    sequence: { useMaxWidth: false },
    class: { useMaxWidth: false },
    state: { useMaxWidth: false },
    er: { useMaxWidth: false },
    gantt: { useMaxWidth: false },
    pie: { useMaxWidth: false },
    journey: { useMaxWidth: false },
    mindmap: { useMaxWidth: false },
    gitGraph: { useMaxWidth: false },
  });
}

function parseErrorLine(err) {
  // The message carries a human-facing 1-based line number; the parser's own
  // hash.line is 0-based, so it needs shifting when we fall back to it.
  const match = /line[: ]+(\d+)/i.exec((err && err.message) || '');
  if (match) return Number(match[1]);
  const raw = err && err.hash && Number(err.hash.line);
  return Number.isFinite(raw) && raw >= 0 ? raw + 1 : 0;
}

function showError(err) {
  errorLine = parseErrorLine(err);
  const message = (err && err.message ? err.message : String(err)).trim();
  el.errorText.textContent = errorLine
    ? `${message}\n\n(see line ${errorLine} in the editor)`
    : message;
  el.errorPanel.hidden = false;
  setStatus(errorLine ? `Error on line ${errorLine}` : 'Diagram error', true);
  refreshGutter();
}

function clearError() {
  errorLine = 0;
  el.errorPanel.hidden = true;
  el.errorText.textContent = '';
  refreshGutter();
}

async function render() {
  if (!state.mermaid) return;
  const code = el.code.value.trim();
  const seq = ++state.renderSeq;

  if (!code) {
    el.stage.innerHTML = '';
    clearError();
    setStatus('Empty diagram — type some Mermaid code to get started.');
    return;
  }

  try {
    await state.mermaid.parse(code);
    const { svg } = await state.mermaid.render(`mg${seq}`, code);
    if (seq !== state.renderSeq) return;   // a newer render already won

    el.stage.innerHTML = svg;
    const node = el.stage.querySelector('svg');
    if (node) {
      node.removeAttribute('width');
      node.removeAttribute('height');
      node.style.maxWidth = 'none';
      const { width, height } = svgSize(node);
      node.setAttribute('width', width);
      node.setAttribute('height', height);
    }
    state.lastGoodSvg = svg;
    clearError();
    setStatus('Rendered');
    if (state.fitPending) {
      state.fitPending = false;
      fitToView();
    } else {
      applyTransform();
    }
  } catch (err) {
    if (seq !== state.renderSeq) return;
    // Mermaid drops a scratch node into <body> when a render throws.
    document.querySelectorAll(`#dmg${seq}, #mg${seq}`).forEach((n) => n.remove());
    showError(err);
  }
}

const renderSoon = debounce(render, 280);

function svgSize(node) {
  const viewBox = (node.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number);
  if (viewBox.length === 4 && viewBox.every(Number.isFinite) && viewBox[2] > 0) {
    return { width: Math.ceil(viewBox[2]), height: Math.ceil(viewBox[3]) };
  }
  const box = node.getBoundingClientRect();
  return { width: Math.ceil(box.width) || 800, height: Math.ceil(box.height) || 600 };
}

/* ------------------------------------------------------------------ *
 * Pan and zoom
 * ------------------------------------------------------------------ */

function applyTransform() {
  const { scale, x, y } = state.view;
  el.stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  el.zoomLabel.textContent = `${Math.round(scale * 100)}%`;
}

function fitToView() {
  const node = el.stage.querySelector('svg');
  if (!node) return;
  const { width, height } = svgSize(node);
  const box = el.viewport.getBoundingClientRect();
  const scale = Math.min((box.width - 40) / width, (box.height - 40) / height, 1.6);
  state.view.scale = Math.max(0.05, scale);
  state.view.x = (box.width - width * state.view.scale) / 2;
  state.view.y = (box.height - height * state.view.scale) / 2;
  applyTransform();
}

function zoomAt(factor, originX, originY) {
  const next = Math.min(8, Math.max(0.05, state.view.scale * factor));
  const ratio = next / state.view.scale;
  state.view.x = originX - (originX - state.view.x) * ratio;
  state.view.y = originY - (originY - state.view.y) * ratio;
  state.view.scale = next;
  applyTransform();
}

function initPanZoom() {
  el.viewport.addEventListener('wheel', (event) => {
    event.preventDefault();
    const box = el.viewport.getBoundingClientRect();
    if (event.shiftKey) {
      state.view.x -= event.deltaY;
      applyTransform();
      return;
    }
    zoomAt(Math.exp(-event.deltaY * 0.0016), event.clientX - box.left, event.clientY - box.top);
  }, { passive: false });

  let panning = null;
  el.viewport.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 && event.button !== 1) return;
    panning = { id: event.pointerId, x: event.clientX, y: event.clientY };
    el.viewport.setPointerCapture(event.pointerId);
    el.viewport.classList.add('panning');
  });

  el.viewport.addEventListener('pointermove', (event) => {
    if (!panning || panning.id !== event.pointerId) return;
    state.view.x += event.clientX - panning.x;
    state.view.y += event.clientY - panning.y;
    panning.x = event.clientX;
    panning.y = event.clientY;
    applyTransform();
  });

  const endPan = (event) => {
    if (!panning || panning.id !== event.pointerId) return;
    panning = null;
    el.viewport.classList.remove('panning');
  };
  el.viewport.addEventListener('pointerup', endPan);
  el.viewport.addEventListener('pointercancel', endPan);

  el.zoomIn.addEventListener('click', () => zoomCentre(1.25));
  el.zoomOut.addEventListener('click', () => zoomCentre(0.8));
  el.zoomReset.addEventListener('click', fitToView);
}

function zoomCentre(factor) {
  const box = el.viewport.getBoundingClientRect();
  zoomAt(factor, box.width / 2, box.height / 2);
}

/* ------------------------------------------------------------------ *
 * Documents: open, save, delete
 * ------------------------------------------------------------------ */

function markDirty(dirty) {
  state.dirty = dirty;
  el.dirtyDot.hidden = !dirty;
}

function onCodeChanged() {
  markDirty(true);
  refreshGutter();
  refreshCursorInfo();
  saveDraft();
  renderSoon();
}

const saveDraft = debounce(() => {
  store.draft = { id: state.currentId, name: el.docTitle.value, code: el.code.value };
  persist();
}, 500);

function loadDoc(doc, { fit = true } = {}) {
  state.currentId = doc.id || null;
  el.docTitle.value = doc.name || 'Untitled diagram';
  el.code.value = doc.code || '';
  markDirty(Boolean(doc.dirty));
  state.fitPending = fit;
  refreshGutter();
  refreshCursorInfo();
  renderList();
  render();
}

function newDoc() {
  if (state.dirty && !confirm('Discard unsaved changes to the current diagram?')) return;
  loadDoc({ id: null, name: 'Untitled diagram', code: STARTER });
  store.draft = null;
  persist();
  el.docTitle.focus();
  el.docTitle.select();
}

function saveDoc() {
  const name = el.docTitle.value.trim() || 'Untitled diagram';
  el.docTitle.value = name;
  const now = Date.now();
  const existing = store.docs.find((d) => d.id === state.currentId);

  if (existing) {
    existing.name = name;
    existing.code = el.code.value;
    existing.updatedAt = now;
  } else {
    const doc = { id: newId(), name, code: el.code.value, createdAt: now, updatedAt: now };
    store.docs.push(doc);
    state.currentId = doc.id;
  }

  store.draft = null;
  if (!persist()) return;
  markDirty(false);
  renderList();
  toast(existing ? 'Saved' : 'Saved as a new diagram');
}

function duplicateDoc(id) {
  const source = store.docs.find((d) => d.id === id);
  if (!source) return;
  const now = Date.now();
  const copy = { ...source, id: newId(), name: `${source.name} (copy)`, createdAt: now, updatedAt: now };
  store.docs.push(copy);
  persist();
  loadDoc({ ...copy });
  toast('Duplicated');
}

function deleteDoc(id) {
  const doc = store.docs.find((d) => d.id === id);
  if (!doc) return;
  if (!confirm(`Delete “${doc.name}”? This cannot be undone.`)) return;
  store.docs = store.docs.filter((d) => d.id !== id);
  if (state.currentId === id) {
    state.currentId = null;
    markDirty(true);
  }
  persist();
  renderList();
  toast('Deleted');
}

function renameDoc(id) {
  const doc = store.docs.find((d) => d.id === id);
  if (!doc) return;
  const name = prompt('New name', doc.name);
  if (name === null) return;
  doc.name = name.trim() || doc.name;
  doc.updatedAt = Date.now();
  persist();
  if (state.currentId === id) el.docTitle.value = doc.name;
  renderList();
}

function openDoc(id) {
  if (state.dirty && !confirm('Discard unsaved changes to the current diagram?')) return;
  const doc = store.docs.find((d) => d.id === id);
  if (doc) loadDoc({ ...doc });
}

function renderList() {
  const query = el.searchBox.value.trim().toLowerCase();
  const docs = [...store.docs]
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
    .filter((d) => !query
      || d.name.toLowerCase().includes(query)
      || (d.code || '').toLowerCase().includes(query));

  el.diagramList.innerHTML = '';

  if (!docs.length) {
    const note = document.createElement('li');
    note.className = 'empty-note';
    note.textContent = store.docs.length
      ? 'No diagrams match that search.'
      : 'No saved diagrams yet. Press Save to keep the current one in this browser.';
    note.style.cursor = 'default';
    el.diagramList.appendChild(note);
  }

  for (const doc of docs) {
    const li = document.createElement('li');
    li.className = doc.id === state.currentId ? 'active' : '';

    const main = document.createElement('div');
    main.className = 'item-main';
    main.innerHTML = '<span class="item-name"></span><span class="item-date"></span>';
    main.querySelector('.item-name').textContent = doc.name;
    main.querySelector('.item-date').textContent = formatDate(doc.updatedAt);
    main.addEventListener('click', () => openDoc(doc.id));
    li.appendChild(main);

    for (const [label, title, action] of [
      ['✎', 'Rename', renameDoc],
      ['⧉', 'Duplicate', duplicateDoc],
      ['🗑', 'Delete', deleteDoc],
    ]) {
      const btn = document.createElement('button');
      btn.className = 'row-btn';
      btn.textContent = label;
      btn.title = title;
      btn.addEventListener('click', (event) => { event.stopPropagation(); action(doc.id); });
      li.appendChild(btn);
    }

    el.diagramList.appendChild(li);
  }

  const count = store.docs.length;
  const bytes = new Blob([JSON.stringify(store.docs)]).size;
  el.storageInfo.textContent = `${count} diagram${count === 1 ? '' : 's'} · ${(bytes / 1024).toFixed(1)} KB in this browser`;
}

/* ------------------------------------------------------------------ *
 * Export
 * ------------------------------------------------------------------ */

function currentSvgElement() {
  const node = el.stage.querySelector('svg');
  if (!node) {
    toast('Nothing to export — fix the diagram error first.', true);
    return null;
  }
  return node;
}

function serialiseSvg(node) {
  const clone = node.cloneNode(true);
  const { width, height } = svgSize(node);
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  clone.setAttribute('width', width);
  clone.setAttribute('height', height);
  clone.style.maxWidth = 'none';
  return { markup: new XMLSerializer().serializeToString(clone), width, height };
}

function exportSvg() {
  const node = currentSvgElement();
  if (!node) return;
  const { markup } = serialiseSvg(node);
  downloadBlob(new Blob([markup], { type: 'image/svg+xml;charset=utf-8' }),
    `${safeFileName(el.docTitle.value)}.svg`);
  toast('SVG downloaded');
}

async function exportPng(scale = 2) {
  const node = currentSvgElement();
  if (!node) return;
  const { markup, width, height } = serialiseSvg(node);
  const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;

  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('the browser could not rasterise the SVG'));
      image.src = url;
    });

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = store.settings.mermaidTheme === 'dark' ? '#1e1e1e' : '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    downloadBlob(blob, `${safeFileName(el.docTitle.value)}.png`);
    toast(`PNG downloaded (${canvas.width}×${canvas.height})`);
  } catch (err) {
    toast(`PNG export failed: ${err.message}`, true);
  }
}

function exportSource() {
  downloadBlob(new Blob([el.code.value], { type: 'text/plain;charset=utf-8' }),
    `${safeFileName(el.docTitle.value)}.mmd`);
  toast('Source downloaded');
}

async function copyShareLink() {
  const payload = await encodeShare(el.code.value);
  const link = `${location.origin}${location.pathname}#${payload}`;
  history.replaceState(null, '', `#${payload}`);
  if (link.length > 8000) {
    toast('Link copied, but it is very long — some apps may truncate it.', true);
  } else {
    toast('Shareable link copied');
  }
  await copyText(link);
}

function backupAll() {
  const payload = {
    kind: 'mermaid-editor-backup',
    version: 1,
    exportedAt: new Date().toISOString(),
    docs: store.docs,
  };
  downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
    `mermaid-diagrams-${new Date().toISOString().slice(0, 10)}.json`);
  toast(`Backed up ${store.docs.length} diagram${store.docs.length === 1 ? '' : 's'}`);
}

function restoreBackup(text) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    toast('That file is not valid JSON.', true);
    return;
  }

  const incoming = Array.isArray(payload) ? payload : payload.docs;
  if (!Array.isArray(incoming)) {
    toast('No diagrams found in that backup file.', true);
    return;
  }

  const known = new Set(store.docs.map((d) => d.id));
  let added = 0;
  for (const doc of incoming) {
    if (!doc || typeof doc.code !== 'string') continue;
    const id = known.has(doc.id) || !doc.id ? newId() : doc.id;
    known.add(id);
    store.docs.push({
      id,
      name: String(doc.name || 'Restored diagram'),
      code: doc.code,
      createdAt: doc.createdAt || Date.now(),
      updatedAt: doc.updatedAt || Date.now(),
    });
    added += 1;
  }

  persist();
  renderList();
  toast(added ? `Restored ${added} diagram${added === 1 ? '' : 's'}` : 'Nothing to restore', !added);
}

/* ------------------------------------------------------------------ *
 * Menus, dialogs, settings
 * ------------------------------------------------------------------ */

function closeMenus() {
  for (const [btn, menu] of [[el.btnExport, el.exportMenu], [el.btnMore, el.moreMenu]]) {
    menu.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  }
}

function toggleMenu(btn, menu) {
  const willOpen = menu.hidden;
  closeMenus();
  menu.hidden = !willOpen;
  btn.setAttribute('aria-expanded', String(willOpen));
}

function showInfo(title, html) {
  el.infoTitle.textContent = title;
  el.infoBody.innerHTML = html;
  el.infoDialog.showModal();
}

const SHORTCUTS_HTML = `
  <ul>
    <li><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>S</kbd> — save the current diagram</li>
    <li><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Enter</kbd> — re-render now</li>
    <li><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>B</kbd> — show/hide the saved diagram list</li>
    <li><kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd> — indent or dedent the selected lines</li>
    <li>Scroll to zoom the preview, drag to pan, <kbd>Shift</kbd>+scroll to pan sideways</li>
    <li>Drop a <code>.mmd</code> file anywhere on the page to open it</li>
  </ul>`;

const ABOUT_HTML = `
  <p>A free front end for <a href="https://github.com/mermaid-js/mermaid" target="_blank"
     rel="noopener">mermaid-js</a>. Everything happens in your browser — nothing is uploaded,
     and there is no account or subscription.</p>
  <p><strong>Where your diagrams live:</strong> in this browser's <code>localStorage</code>,
     under the key <code>mermaid-editor.v1</code>. That means they are tied to this browser on
     this machine, and clearing site data will remove them.</p>
  <p><strong>So keep real backups:</strong> use <em>More → Back up all diagrams</em> for a JSON
     file you can restore anywhere, or export the <code>.mmd</code> source of anything you care
     about and commit it alongside your work.</p>`;

function applyUiTheme() {
  el.html.setAttribute('data-ui-theme', store.settings.uiTheme);
}

function applyEditorWidth() {
  el.editorPane.style.flexBasis = `${store.settings.editorWidth}%`;
}

function applySidebarVisibility() {
  el.layout.classList.toggle('sidebar-hidden', Boolean(store.settings.sidebarHidden));
}

function initSplitter() {
  let dragging = false;

  el.splitter.addEventListener('pointerdown', (event) => {
    dragging = true;
    el.splitter.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  el.splitter.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const box = el.layout.getBoundingClientRect();
    const sidebar = store.settings.sidebarHidden ? 0 : el.sidebar.getBoundingClientRect().width;
    const usable = box.width - sidebar;
    const pct = ((event.clientX - box.left - sidebar) / usable) * 100;
    store.settings.editorWidth = Math.min(80, Math.max(15, pct));
    applyEditorWidth();
  });

  el.splitter.addEventListener('pointerup', (event) => {
    if (!dragging) return;
    dragging = false;
    el.splitter.releasePointerCapture(event.pointerId);
    persist();
  });

  el.splitter.addEventListener('keydown', (event) => {
    const step = event.key === 'ArrowLeft' ? -3 : event.key === 'ArrowRight' ? 3 : 0;
    if (!step) return;
    event.preventDefault();
    store.settings.editorWidth = Math.min(80, Math.max(15, store.settings.editorWidth + step));
    applyEditorWidth();
    persist();
  });
}

/* ------------------------------------------------------------------ *
 * Wiring
 * ------------------------------------------------------------------ */

function initTemplates() {
  for (const name of Object.keys(TEMPLATES)) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    el.templateSelect.appendChild(option);
  }

  el.templateSelect.addEventListener('change', () => {
    const name = el.templateSelect.value;
    el.templateSelect.value = '';
    if (!name) return;
    if (el.code.value.trim() && !confirm(`Replace the current code with the “${name}” template?`)) return;
    el.code.value = TEMPLATES[name];
    state.fitPending = true;
    onCodeChanged();
  });
}

function initEvents() {
  el.code.addEventListener('input', onCodeChanged);
  el.code.addEventListener('keydown', handleEditorKeys);
  el.code.addEventListener('scroll', () => { el.gutter.scrollTop = el.code.scrollTop; });
  el.code.addEventListener('click', refreshCursorInfo);
  el.code.addEventListener('keyup', refreshCursorInfo);

  el.docTitle.addEventListener('input', () => { markDirty(true); saveDraft(); });

  el.btnNew.addEventListener('click', newDoc);
  el.btnSave.addEventListener('click', saveDoc);
  el.searchBox.addEventListener('input', renderList);

  el.toggleSidebar.addEventListener('click', () => {
    store.settings.sidebarHidden = !store.settings.sidebarHidden;
    applySidebarVisibility();
    persist();
  });

  el.btnUiTheme.addEventListener('click', () => {
    store.settings.uiTheme = store.settings.uiTheme === 'dark' ? 'light' : 'dark';
    applyUiTheme();
    persist();
  });

  el.mermaidTheme.addEventListener('change', () => {
    store.settings.mermaidTheme = el.mermaidTheme.value;
    persist();
    configureMermaid();
    render();
  });

  el.btnExport.addEventListener('click', (e) => { e.stopPropagation(); toggleMenu(el.btnExport, el.exportMenu); });
  el.btnMore.addEventListener('click', (e) => { e.stopPropagation(); toggleMenu(el.btnMore, el.moreMenu); });
  document.addEventListener('click', closeMenus);

  el.exportMenu.addEventListener('click', async (event) => {
    const act = event.target.dataset && event.target.dataset.act;
    if (!act) return;
    closeMenus();
    if (act === 'png') exportPng();
    if (act === 'svg') exportSvg();
    if (act === 'mmd') exportSource();
    if (act === 'copy-svg') {
      const node = currentSvgElement();
      if (node) {
        const ok = await copyText(serialiseSvg(node).markup);
        toast(ok ? 'SVG markup copied' : 'Copy failed', !ok);
      }
    }
    if (act === 'copy-code') {
      const ok = await copyText(el.code.value);
      toast(ok ? 'Code copied' : 'Copy failed', !ok);
    }
    if (act === 'copy-link') copyShareLink();
  });

  el.moreMenu.addEventListener('click', (event) => {
    const act = event.target.dataset && event.target.dataset.act;
    if (!act) return;
    closeMenus();
    if (act === 'open-file') el.filePicker.click();
    if (act === 'backup') backupAll();
    if (act === 'restore') el.backupPicker.click();
    if (act === 'shortcuts') showInfo('Keyboard shortcuts', SHORTCUTS_HTML);
    if (act === 'about') showInfo('About this editor', ABOUT_HTML);
  });

  el.filePicker.addEventListener('change', async () => {
    const file = el.filePicker.files[0];
    if (!file) return;
    loadDoc({ id: null, name: file.name.replace(/\.[^.]+$/, ''), code: await file.text(), dirty: true });
    el.filePicker.value = '';
  });

  el.backupPicker.addEventListener('change', async () => {
    const file = el.backupPicker.files[0];
    if (!file) return;
    restoreBackup(await file.text());
    el.backupPicker.value = '';
  });

  // Drag and drop a .mmd file anywhere.
  document.addEventListener('dragover', (event) => { event.preventDefault(); });
  document.addEventListener('drop', async (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (!file) return;
    event.preventDefault();
    if (/\.json$/i.test(file.name)) {
      restoreBackup(await file.text());
    } else {
      loadDoc({ id: null, name: file.name.replace(/\.[^.]+$/, ''), code: await file.text(), dirty: true });
    }
  });

  document.addEventListener('keydown', (event) => {
    const mod = event.ctrlKey || event.metaKey;
    if (!mod) return;
    const key = event.key.toLowerCase();
    if (key === 's') { event.preventDefault(); saveDoc(); }
    if (key === 'b') { event.preventDefault(); el.toggleSidebar.click(); }
    if (event.key === 'Enter') { event.preventDefault(); render(); }
  });

  window.addEventListener('beforeunload', (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = '';
  });

  window.addEventListener('resize', debounce(() => { if (el.stage.querySelector('svg')) applyTransform(); }, 150));
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

async function boot() {
  applyUiTheme();
  applyEditorWidth();
  applySidebarVisibility();
  el.mermaidTheme.value = store.settings.mermaidTheme;

  initTemplates();
  initEvents();
  initPanZoom();
  initSplitter();

  // Decide what to open: a shared link wins, then an unsaved draft, then the starter.
  let initial = { id: null, name: 'Untitled diagram', code: STARTER };
  if (location.hash.length > 3) {
    try {
      const shared = await decodeShare(location.hash);
      if (shared) initial = { id: null, name: 'Shared diagram', code: shared, dirty: true };
    } catch {
      toast('That shared link could not be decoded.', true);
    }
  } else if (store.draft && typeof store.draft.code === 'string') {
    initial = { id: store.draft.id, name: store.draft.name, code: store.draft.code, dirty: true };
  }

  state.currentId = initial.id || null;
  el.docTitle.value = initial.name;
  el.code.value = initial.code;
  markDirty(Boolean(initial.dirty));
  refreshGutter();
  refreshCursorInfo();
  renderList();

  el.loading.hidden = false;
  setStatus('Loading Mermaid…');
  try {
    const { mermaid, src } = await loadMermaid();
    state.mermaid = mermaid;
    configureMermaid();
    el.loading.hidden = true;
    const where = src.startsWith('.') ? 'local copy' : new URL(src).hostname;
    const version = typeof mermaid.version === 'function' ? mermaid.version() : (mermaid.version || '');
    setStatus(`Mermaid ${version} ready (${where})`.replace(/\s+/g, ' '));
    await render();
  } catch (err) {
    el.loading.hidden = true;
    setStatus('Mermaid failed to load', true);
    el.errorPanel.hidden = false;
    el.errorText.textContent =
      `${err.message}\n\nThe editor needs the Mermaid library. Either connect to the internet, `
      + 'or run ./download-offline-copy.sh next to this page to keep a local copy in ./vendor/.';
  }
}

boot();
