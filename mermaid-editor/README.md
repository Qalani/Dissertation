# Mermaid Flowchart Editor

A free, self-contained replacement for the paid Mermaid Live Editor. Write
[Mermaid](https://mermaid.js.org) diagram code on the left, see the rendered
diagram on the right, and save as many diagrams as you like — no account, no
server, no internet connection required.

## How to open it

Use **`mermaid-editor.html`** — one file with everything already inside it.

1. Download it: on GitHub open `mermaid-editor.html`, then click the
   **Download raw file** button (the ⤓ icon, top right of the file view).
   *Do not* copy text out of GitHub's file viewer — that page shows source
   code and never runs it.
2. **Right-click** the downloaded file → **Open with** → pick a browser
   (Chrome, Edge, Firefox, Safari).

> **If double-clicking shows you a wall of HTML code instead of the editor,**
> a text editor (VS Code, Notepad, TextEdit) is registered as the default app
> for `.html` files on your machine. That's the whole problem — the file is
> fine. Use the right-click → *Open with* → browser route above.
>
> To make it stick: **Windows** — right-click → *Open with* → *Choose another
> app* → pick your browser → tick *Always use this app*. **macOS** —
> right-click → *Get Info* → under *Open with* pick your browser → *Change All…*

Once it's open, bookmark the page (`Ctrl+D` / `⌘D`) so you can get back to it
without hunting for the file.

## The methodology flowcharts are already in it

The 13 diagrams from `methodology-flowcharts.md` are built in. On first open
they appear in the sidebar, numbered in document order, with diagram 1 loaded
and ready to edit or export.

They behave like any other saved diagram: edit them, rename them, delete the
ones you don't need. Seeding runs **once per browser**, so a diagram you
delete stays deleted and reopening the editor never creates duplicates. To
bring one back, pick it from **Examples → Methodology flowcharts**, which
always holds a pristine copy.

Editing a seeded diagram and pressing Save updates it in place, exactly like
any editor. To keep the original as well, hit **New** first, or rename before
saving.

### Changing the flowcharts

`methodology-flowcharts.md` is the source of truth. Edit the ```mermaid blocks
there, then regenerate:

```bash
python3 build_library.py       # markdown -> library.js
python3 build_single_file.py   # refresh mermaid-editor.html
```

Each `## ` heading becomes a diagram title, so keep those unique.

Note that the built-ins only re-seed in a browser that has never seeded them.
If you want updated copies in a browser you've already used, pull them from
the Examples menu.

## Features

- **Live preview** — the diagram re-renders as you type; syntax errors show in
  a banner while the last good diagram stays visible.
- **Save / load** — named diagrams are stored in the browser's localStorage.
  The sidebar lists them; click to open, `×` to delete. `Ctrl+S` (or `⌘S`) saves.
- **Autosave draft** — closing or refreshing the tab never loses the text
  you were working on.
- **Export** — download the current diagram as **SVG**, **PNG** (2×
  resolution), or the raw **.mmd** source. SVG exports use plain text labels,
  so they open cleanly in Word, Inkscape, and LaTeX pipelines.
- **Import** — open any `.mmd` / `.mermaid` / `.txt` file.
- **Examples** — the methodology flowcharts, plus starter templates for
  flowcharts, sequence, state, class, and Gantt diagrams.
- **Zoom** — **Fit** scales a diagram to the pane, **100%** shows it at true
  size, and `Ctrl` + scroll wheel zooms freely. Large diagrams open fitted so
  you see the whole thing first. Drag the divider to resize the panes.
- **Light / dark** — the toolbar button cycles **match system → light →
  dark**, and remembers the choice. It restyles the diagram as well as the
  editor, so **switch to light before exporting** anything headed for a white
  dissertation page: PNG exports use the current theme's background, and a
  dark-theme diagram draws light text.

## Where are my diagrams stored?

In the browser's localStorage, keyed to wherever the page is opened from
(file path or URL). They are private to your machine and browser profile.
Two consequences worth knowing:

- Clearing browser data deletes them, so use the **.mmd** export button to
  keep file backups of anything important. You can re-import those any time.
- Moving the HTML file to a different folder changes the file path, and the
  browser treats that as a different site — your saved list will look empty.
  Pick a permanent home for the file before you start saving work.

## Optional: use it from a URL instead of a file

Serving over HTTP sidesteps the "which app opens .html" question entirely.

```bash
cd mermaid-editor
python3 -m http.server 8000
# then visit http://localhost:8000
```

Or publish it with GitHub Pages (repo **Settings → Pages → Deploy from a
branch**, pick `main` and `/root`) and use it from any device at
`https://qalani.github.io/Dissertation/mermaid-editor/`.

## Repository layout

| File | Purpose |
| --- | --- |
| `mermaid-editor.html` | **The thing to open.** Standalone build, everything inlined. |
| `index.html`, `styles.css`, `app.js` | Editable sources. Work here, not in the standalone build. |
| `methodology-flowcharts.md` | Source of truth for the 13 built-in diagrams. |
| `library.js` | Generated from that markdown — don't edit by hand. |
| `vendor/mermaid.min.js` | Mermaid v11.16.0 ([MIT](https://github.com/mermaid-js/mermaid/blob/develop/LICENSE)), vendored so the editor works offline. |
| `build_library.py` | Regenerates `library.js` from the markdown. |
| `build_single_file.py` | Regenerates `mermaid-editor.html` from the sources. |
| `build_artifact.py` | Emits a body-only copy for hosting elsewhere. |

After editing any source file, rebuild the standalone copy:

```bash
python3 build_single_file.py
```

To upgrade Mermaid, replace the vendored library and rebuild:

```bash
curl -L -o vendor/mermaid.min.js https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js
python3 build_single_file.py
```
