/* Mermaid Flowchart Editor — no build step, no server, no account.
 * Diagrams are stored in this browser's localStorage; use the .mmd export
 * for durable backups or to move diagrams between machines. */
(function () {
  "use strict";

  var STORE_KEY = "mermaidEditor.diagrams.v1";
  var DRAFT_KEY = "mermaidEditor.draft.v1";
  var SEEDED_KEY = "mermaidEditor.seeded.v1";
  var PNG_SCALE = 2;

  // Built-in diagrams generated from methodology-flowcharts.md (library.js).
  // Seeded into the saved list on first run and always available from the
  // Examples menu, so they can be restored after a delete.
  var LIBRARY = window.MERMAID_EDITOR_LIBRARY || [];

  // ---------- DOM ----------
  var $ = function (id) { return document.getElementById(id); };
  var codeEl = $("code");
  var nameEl = $("diagram-name");
  var previewEl = $("preview");
  var previewScrollEl = $("preview-scroll");
  var errorBar = $("error-bar");
  var savedList = $("saved-list");
  var savedCount = $("saved-count");
  var sidebarEmpty = $("sidebar-empty");
  var saveStatus = $("save-status");
  var saveBtn = $("btn-save");
  var zoomResetBtn = $("btn-zoom-reset");
  var examplesSel = $("examples");

  // ---------- examples ----------
  var EXAMPLES = [
    { name: "Flowchart — basic", code:
"flowchart TD\n" +
"    A([Start]) --> B{Is the data ready?}\n" +
"    B -- Yes --> C[Clean the data]\n" +
"    B -- No --> D[(Fetch from source)]\n" +
"    D --> C\n" +
"    C --> E[Run analysis]\n" +
"    E --> F([Finish])\n" +
"\n" +
"    style A fill:#d3f9d8,stroke:#2b8a3e,color:#1b4332\n" +
"    style F fill:#ffe3e3,stroke:#c92a2a,color:#641220\n" },
    { name: "Flowchart — subgraphs", code:
"flowchart LR\n" +
"    subgraph Inputs\n" +
"        A[Satellite imagery] --> C\n" +
"        B[Field survey] --> C\n" +
"    end\n" +
"    C{Merge &\\nvalidate} --> D[Model]\n" +
"    subgraph Outputs\n" +
"        D --> E[Maps]\n" +
"        D --> F[Report]\n" +
"    end\n" },
    { name: "Sequence diagram", code:
"sequenceDiagram\n" +
"    participant U as User\n" +
"    participant S as Server\n" +
"    participant DB as Database\n" +
"    U->>S: Request page\n" +
"    S->>DB: Query data\n" +
"    DB-->>S: Rows\n" +
"    S-->>U: Rendered page\n" },
    { name: "State diagram", code:
"stateDiagram-v2\n" +
"    [*] --> Draft\n" +
"    Draft --> Review : submit\n" +
"    Review --> Draft : request changes\n" +
"    Review --> Published : approve\n" +
"    Published --> [*]\n" },
    { name: "Class diagram", code:
"classDiagram\n" +
"    Animal <|-- Duck\n" +
"    Animal <|-- Fish\n" +
"    Animal : +String name\n" +
"    Animal : +move()\n" +
"    class Duck {\n" +
"        +swim()\n" +
"        +quack()\n" +
"    }\n" },
    { name: "Gantt chart", code:
"gantt\n" +
"    title Project plan\n" +
"    dateFormat  YYYY-MM-DD\n" +
"    section Research\n" +
"    Literature review     :a1, 2026-01-06, 30d\n" +
"    Data collection       :a2, after a1, 45d\n" +
"    section Writing\n" +
"    Draft chapters        :b1, after a2, 60d\n" +
"    Revisions             :b2, after b1, 30d\n" }
  ];

  // ---------- storage ----------
  // localStorage throws in private windows, sandboxed frames, and when site
  // data is blocked. Fall back to memory so the editor still works; boot()
  // warns the user that saves will not survive a refresh in that case.
  var storage = (function () {
    try {
      var probe = "__mmd_probe__";
      localStorage.setItem(probe, "1");
      localStorage.removeItem(probe);
      return {
        persistent: true,
        get: function (k) { return localStorage.getItem(k); },
        set: function (k, v) { localStorage.setItem(k, v); }
      };
    } catch (e) {
      var mem = Object.create(null);
      return {
        persistent: false,
        get: function (k) { return k in mem ? mem[k] : null; },
        set: function (k, v) { mem[k] = String(v); }
      };
    }
  })();

  // ---------- state ----------
  var currentId = null;      // id of the open saved diagram, or null if never saved
  var savedSnapshot = null;  // {name, code} as of last save/load, for dirty checking
  var renderSeq = 0;
  var zoom = 1;
  var pendingFit = false;  // fit the next render to the pane (set on open, not on keystrokes)

  // ---------- mermaid setup ----------
  var darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function initMermaid() {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: darkQuery.matches ? "dark" : "default",
      // Plain SVG text labels (no <foreignObject>) so exported SVGs open
      // correctly in Word, Inkscape, LaTeX pipelines, etc.
      htmlLabels: false,
      flowchart: { htmlLabels: false },
      class: { htmlLabels: false }
    });
  }

  // ---------- rendering ----------
  // Mermaid emits width:100% + max-width, which squeezes a large diagram into
  // the pane and makes the zoom percentage meaningless. Pin the SVG to its
  // real dimensions instead and let the zoom transform do all the scaling.
  function svgSize(svg) {
    var vb = svg.viewBox && svg.viewBox.baseVal;
    if (vb && vb.width && vb.height) return { w: vb.width, h: vb.height };
    var box = svg.getBoundingClientRect();
    return { w: box.width || 800, h: box.height || 600 };
  }

  function sizeNaturally(svg) {
    var size = svgSize(svg);
    svg.setAttribute("width", Math.ceil(size.w));
    svg.setAttribute("height", Math.ceil(size.h));
    svg.style.maxWidth = "none";
    svg.style.width = Math.ceil(size.w) + "px";
    svg.style.height = Math.ceil(size.h) + "px";
  }

  // Largest zoom that shows the whole diagram, never magnifying past 100%.
  function fitZoom() {
    var svg = previewEl.querySelector("svg");
    if (!svg) return 1;
    var size = svgSize(svg);
    var pad = 72; // scroll padding plus the SVG's own frame
    var availW = previewScrollEl.clientWidth - pad;
    var availH = previewScrollEl.clientHeight - pad;
    if (availW <= 0 || availH <= 0) return 1;
    return Math.min(1, availW / size.w, availH / size.h);
  }

  function showError(message) {
    errorBar.textContent = message;
    errorBar.hidden = false;
  }

  function hideError() {
    errorBar.hidden = true;
  }

  function render() {
    var code = codeEl.value;
    var seq = ++renderSeq;

    if (!code.trim()) {
      previewEl.innerHTML = '<div class="placeholder">Start typing on the left, or pick an example from the toolbar.</div>';
      hideError();
      return;
    }

    var renderId = "mmd-" + seq;
    mermaid.parse(code)
      .then(function () { return mermaid.render(renderId, code); })
      .then(function (result) {
        if (seq !== renderSeq) return; // a newer render superseded this one
        previewEl.innerHTML = result.svg;
        var svg = previewEl.querySelector("svg");
        if (svg) sizeNaturally(svg);
        if (pendingFit) {
          pendingFit = false;
          setZoom(fitZoom());
        }
        hideError();
      })
      .catch(function (err) {
        // Clean up the temp node mermaid can leave behind on failed renders
        var orphan = document.getElementById("d" + renderId);
        if (orphan) orphan.remove();
        if (seq !== renderSeq) return;
        // Keep the last good diagram visible; just surface the message
        showError(String(err && err.message || err));
      });
  }

  var renderTimer = null;
  function scheduleRender() {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(render, 250);
  }

  // ---------- storage ----------
  function loadStore() {
    try {
      var raw = storage.get(STORE_KEY);
      var list = raw ? JSON.parse(raw) : [];
      return Array.isArray(list) ? list : [];
    } catch (e) {
      return [];
    }
  }

  function persistStore(list) {
    try {
      storage.set(STORE_KEY, JSON.stringify(list));
      return true;
    } catch (e) {
      alert("Could not save to this browser's storage:\n" + e.message);
      return false;
    }
  }

  function newId() {
    return (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : "id-" + Date.now() + "-" + Math.random().toString(36).slice(2);
  }

  function saveDraft() {
    try {
      storage.set(DRAFT_KEY, JSON.stringify({
        id: currentId, name: nameEl.value, code: codeEl.value
      }));
    } catch (e) { /* draft is best-effort */ }
  }

  var draftTimer = null;
  function scheduleDraft() {
    clearTimeout(draftTimer);
    draftTimer = setTimeout(saveDraft, 400);
  }

  // ---------- dirty tracking / status ----------
  function isDirty() {
    if (!savedSnapshot) return codeEl.value.trim().length > 0;
    return codeEl.value !== savedSnapshot.code || nameEl.value !== savedSnapshot.name;
  }

  function fmtTime(ts) {
    var d = new Date(ts);
    var today = new Date();
    var sameDay = d.toDateString() === today.toDateString();
    return sameDay
      ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
  }

  function refreshStatus() {
    var dirty = isDirty();
    saveBtn.classList.toggle("dirty", dirty);
    saveStatus.classList.remove("saved");
    if (!currentId) {
      saveStatus.textContent = codeEl.value.trim() ? "Not saved yet — press Save (Ctrl+S)" : "Not saved yet";
    } else if (dirty) {
      saveStatus.textContent = "Unsaved changes";
    } else {
      var doc = loadStore().find(function (d) { return d.id === currentId; });
      saveStatus.textContent = "Saved ✓ " + (doc ? fmtTime(doc.updatedAt) : "");
      saveStatus.classList.add("saved");
    }
  }

  // ---------- sidebar ----------
  function renderSidebar() {
    var list = loadStore().slice().sort(function (a, b) { return b.updatedAt - a.updatedAt; });
    savedCount.textContent = String(list.length);
    sidebarEmpty.hidden = list.length > 0;
    savedList.innerHTML = "";

    list.forEach(function (doc) {
      var li = document.createElement("li");
      if (doc.id === currentId) li.classList.add("active");

      var open = document.createElement("button");
      open.className = "open-btn";
      open.title = "Open “" + doc.name + "”";
      var nm = document.createElement("span");
      nm.className = "doc-name";
      nm.textContent = doc.name;
      var dt = document.createElement("span");
      dt.className = "doc-date";
      dt.textContent = fmtTime(doc.updatedAt);
      open.appendChild(nm);
      open.appendChild(dt);
      open.addEventListener("click", function () { openDiagram(doc.id); });

      var del = document.createElement("button");
      del.className = "del-btn";
      del.textContent = "×";
      del.title = "Delete “" + doc.name + "”";
      del.addEventListener("click", function (ev) {
        ev.stopPropagation();
        deleteDiagram(doc.id, doc.name);
      });

      li.appendChild(open);
      li.appendChild(del);
      savedList.appendChild(li);
    });
  }

  // ---------- document operations ----------
  function confirmDiscard() {
    return !isDirty() || confirm("You have unsaved changes. Discard them?");
  }

  function setBuffer(id, name, code, snapshot) {
    currentId = id;
    nameEl.value = name;
    codeEl.value = code;
    savedSnapshot = snapshot ? { name: name, code: code } : null;
    saveDraft();
    refreshStatus();
    renderSidebar();
    pendingFit = true;   // a freshly opened diagram should arrive fully visible
    render();
  }

  function saveCurrent() {
    var name = nameEl.value.trim() || "Untitled diagram";
    nameEl.value = name;
    var list = loadStore();
    var now = Date.now();
    var doc = currentId ? list.find(function (d) { return d.id === currentId; }) : null;

    if (doc) {
      doc.name = name;
      doc.code = codeEl.value;
      doc.updatedAt = now;
    } else {
      doc = { id: newId(), name: name, code: codeEl.value, createdAt: now, updatedAt: now };
      list.push(doc);
      currentId = doc.id;
    }

    if (!persistStore(list)) return;
    savedSnapshot = { name: name, code: codeEl.value };
    saveDraft();
    refreshStatus();
    renderSidebar();
  }

  function openDiagram(id) {
    if (id === currentId && !isDirty()) return;
    if (!confirmDiscard()) return;
    var doc = loadStore().find(function (d) { return d.id === id; });
    if (!doc) return;
    setBuffer(doc.id, doc.name, doc.code, true);
  }

  function deleteDiagram(id, name) {
    if (!confirm('Delete "' + name + '"? This cannot be undone.')) return;
    var list = loadStore().filter(function (d) { return d.id !== id; });
    persistStore(list);
    if (id === currentId) {
      // Keep the text in the buffer, but it is no longer linked to a saved doc
      currentId = null;
      savedSnapshot = null;
      saveDraft();
    }
    refreshStatus();
    renderSidebar();
  }

  // Load the built-in methodology diagrams into the saved list. Runs once per
  // browser: a diagram deleted afterwards stays deleted, and re-opening the
  // editor never resurrects it or creates duplicates. The Examples menu keeps
  // a pristine copy of each for restoring one by hand.
  function seedLibrary() {
    if (!LIBRARY.length || storage.get(SEEDED_KEY)) return;

    var list = loadStore();
    var existing = Object.create(null);
    list.forEach(function (d) { existing[d.name] = true; });

    // The sidebar sorts newest first, so give diagram 1 the latest timestamp
    // and count downwards. That lists them 1..13 in document order and makes
    // diagram 1 the one the editor opens on a first visit.
    var base = Date.now() - LIBRARY.length;
    LIBRARY.forEach(function (item, i) {
      if (existing[item.name]) return;
      var when = base + (LIBRARY.length - i);
      list.push({ id: newId(), name: item.name, code: item.code, createdAt: when, updatedAt: when });
    });

    if (persistStore(list)) {
      try { storage.set(SEEDED_KEY, "1"); } catch (e) { /* best effort */ }
    }
  }

  function newDiagram() {
    if (!confirmDiscard()) return;
    setBuffer(null, "", "flowchart TD\n    A[Start] --> B[Next step]\n", false);
    codeEl.focus();
  }

  // ---------- export / import ----------
  function slugify(name) {
    var s = (name || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    return s || "diagram";
  }

  function download(blob, filename) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
  }

  function getExportSvg() {
    var svg = previewEl.querySelector("svg");
    if (!svg) {
      alert("Nothing to export yet — the preview is empty or has an error.");
      return null;
    }
    var clone = svg.cloneNode(true);
    // Give the file an explicit pixel size (mermaid emits width:100% + max-width)
    var vb = svg.viewBox && svg.viewBox.baseVal;
    var w = (vb && vb.width) || svg.getBoundingClientRect().width || 800;
    var h = (vb && vb.height) || svg.getBoundingClientRect().height || 600;
    clone.setAttribute("width", Math.ceil(w));
    clone.setAttribute("height", Math.ceil(h));
    clone.style.maxWidth = "";
    if (!clone.getAttribute("xmlns")) clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    return { node: clone, width: w, height: h };
  }

  function exportSVG() {
    var ex = getExportSvg();
    if (!ex) return;
    var src = '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(ex.node);
    download(new Blob([src], { type: "image/svg+xml;charset=utf-8" }), slugify(nameEl.value) + ".svg");
  }

  function exportPNG() {
    var ex = getExportSvg();
    if (!ex) return;
    var src = new XMLSerializer().serializeToString(ex.node);
    var url = URL.createObjectURL(new Blob([src], { type: "image/svg+xml;charset=utf-8" }));
    var img = new Image();
    img.onload = function () {
      try {
        var canvas = document.createElement("canvas");
        canvas.width = Math.ceil(ex.width * PNG_SCALE);
        canvas.height = Math.ceil(ex.height * PNG_SCALE);
        var ctx = canvas.getContext("2d");
        ctx.fillStyle = darkQuery.matches ? "#1d2027" : "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function (blob) {
          if (blob) download(blob, slugify(nameEl.value) + ".png");
          else alert("PNG export failed in this browser — try the SVG export instead.");
        }, "image/png");
      } catch (e) {
        alert("PNG export failed (" + e.message + ") — try the SVG export instead.");
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
      alert("PNG export failed — try the SVG export instead.");
    };
    img.src = url;
  }

  function exportMMD() {
    download(new Blob([codeEl.value], { type: "text/plain;charset=utf-8" }), slugify(nameEl.value) + ".mmd");
  }

  function importFile(file) {
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      if (!confirmDiscard()) return;
      var name = file.name.replace(/\.(mmd|mermaid|txt)$/i, "");
      setBuffer(null, name, String(reader.result), false);
    };
    reader.readAsText(file);
  }

  // ---------- zoom ----------
  function applyZoom() {
    previewEl.style.transform = "scale(" + zoom + ")";
    zoomResetBtn.textContent = Math.round(zoom * 100) + "%";
  }

  function setZoom(z) {
    zoom = Math.min(4, Math.max(0.2, z));
    applyZoom();
  }

  // ---------- wire up ----------
  codeEl.addEventListener("input", function () {
    scheduleRender();
    scheduleDraft();
    refreshStatus();
  });

  codeEl.addEventListener("keydown", function (e) {
    if (e.key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      if (!document.execCommand || !document.execCommand("insertText", false, "    ")) {
        var s = codeEl.selectionStart;
        codeEl.setRangeText("    ", s, codeEl.selectionEnd, "end");
        codeEl.dispatchEvent(new Event("input"));
      }
    }
  });

  nameEl.addEventListener("input", function () {
    scheduleDraft();
    refreshStatus();
    document.title = (nameEl.value.trim() || "Mermaid Flowchart Editor") + " — Mermaid Editor";
  });

  saveBtn.addEventListener("click", saveCurrent);
  $("btn-new").addEventListener("click", newDiagram);
  $("btn-export-svg").addEventListener("click", exportSVG);
  $("btn-export-png").addEventListener("click", exportPNG);
  $("btn-export-mmd").addEventListener("click", exportMMD);
  $("btn-import").addEventListener("click", function () { $("file-input").click(); });
  $("file-input").addEventListener("change", function (e) {
    importFile(e.target.files[0]);
    e.target.value = "";
  });
  $("btn-sidebar").addEventListener("click", function () {
    $("sidebar").classList.toggle("hidden");
  });

  $("btn-zoom-fit").addEventListener("click", function () { setZoom(fitZoom()); });
  $("btn-zoom-in").addEventListener("click", function () { setZoom(zoom * 1.2); });
  $("btn-zoom-out").addEventListener("click", function () { setZoom(zoom / 1.2); });
  zoomResetBtn.addEventListener("click", function () { setZoom(1); });
  previewScrollEl.addEventListener("wheel", function (e) {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    setZoom(zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
  }, { passive: false });

  examplesSel.addEventListener("change", function () {
    var parts = String(examplesSel.value).split(":");
    var source = parts[0] === "lib" ? LIBRARY : EXAMPLES;
    var ex = source[Number(parts[1])];
    examplesSel.selectedIndex = 0;
    if (!ex) return;
    if (!confirmDiscard()) return;
    setBuffer(null, ex.name, ex.code, false);
  });

  window.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      saveCurrent();
    }
  });

  // Resizable split between editor and preview
  (function () {
    var divider = $("divider");
    var editorPane = $("editor-pane");
    divider.addEventListener("pointerdown", function (e) {
      e.preventDefault();
      divider.setPointerCapture(e.pointerId);
      divider.classList.add("dragging");
      var split = editorPane.parentElement.getBoundingClientRect();
      function onMove(ev) {
        var frac = (ev.clientX - split.left) / split.width;
        frac = Math.min(0.85, Math.max(0.15, frac));
        editorPane.style.flexBasis = (frac * 100) + "%";
      }
      function onUp(ev) {
        divider.releasePointerCapture(ev.pointerId);
        divider.classList.remove("dragging");
        divider.removeEventListener("pointermove", onMove);
        divider.removeEventListener("pointerup", onUp);
      }
      divider.addEventListener("pointermove", onMove);
      divider.addEventListener("pointerup", onUp);
    });
  })();

  darkQuery.addEventListener("change", function () {
    initMermaid();
    render();
  });

  window.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") saveDraft();
  });

  // ---------- boot ----------
  function boot() {
    if (typeof mermaid === "undefined") {
      showError("Could not load vendor/mermaid.min.js — make sure the vendor folder sits next to index.html.");
      return;
    }
    initMermaid();

    function addOptions(label, items, prefix) {
      if (!items.length) return;
      var group = document.createElement("optgroup");
      group.label = label;
      items.forEach(function (ex, i) {
        var opt = document.createElement("option");
        opt.value = prefix + ":" + i;
        opt.textContent = ex.name;
        group.appendChild(opt);
      });
      examplesSel.appendChild(group);
    }
    addOptions("Methodology flowcharts", LIBRARY, "lib");
    addOptions("Mermaid basics", EXAMPLES, "ex");

    seedLibrary();

    if (!storage.persistent) {
      showError("This browser is blocking site storage, so saved diagrams will disappear when you close the tab. " +
                "Use the .mmd export button to keep your work.");
    }

    var draft = null;
    try { draft = JSON.parse(storage.get(DRAFT_KEY) || "null"); } catch (e) { /* ignore */ }
    var store = loadStore();

    if (draft && typeof draft.code === "string" && draft.code.trim()) {
      // Resume exactly where the last session left off
      var linked = draft.id ? store.find(function (d) { return d.id === draft.id; }) : null;
      currentId = linked ? linked.id : null;
      nameEl.value = draft.name || "";
      codeEl.value = draft.code;
      savedSnapshot = linked ? { name: linked.name, code: linked.code } : null;
    } else if (store.length) {
      // Open the most recently edited saved diagram
      var latest = store.slice().sort(function (a, b) { return b.updatedAt - a.updatedAt; })[0];
      currentId = latest.id;
      nameEl.value = latest.name;
      codeEl.value = latest.code;
      savedSnapshot = { name: latest.name, code: latest.code };
    } else {
      // First visit: show a worked example
      nameEl.value = "My first flowchart";
      codeEl.value = EXAMPLES[0].code;
      savedSnapshot = null;
    }

    refreshStatus();
    renderSidebar();
    applyZoom();
    pendingFit = true;
    render();
  }

  boot();
})();
