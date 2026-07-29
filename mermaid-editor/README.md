# Mermaid Flowchart Editor

A free, self-contained replacement for the paid Mermaid Live Editor. Write
[Mermaid](https://mermaid.js.org) diagram code on the left, see the rendered
diagram on the right, and save as many diagrams as you like — no account, no
server, no internet connection required.

## How to use it

**Just open `index.html` in a browser** (double-click it, or right-click →
Open With → your browser). Everything runs locally.

If you prefer serving it over HTTP:

```bash
cd mermaid-editor
python3 -m http.server 8000
# then visit http://localhost:8000
```

You can also publish it with GitHub Pages (repo Settings → Pages → deploy from
branch) and use it from any device at
`https://<user>.github.io/<repo>/mermaid-editor/`.

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
- **Examples** — starter templates for flowcharts, sequence, state, class,
  and Gantt diagrams.
- **Zoom** — toolbar buttons or `Ctrl` + scroll wheel over the preview;
  drag the divider to resize the panes. Light and dark mode follow your
  system setting.

## Where are my diagrams stored?

In the browser's localStorage, keyed to wherever the page is opened from
(file path or URL). They are private to your machine and browser profile.
Because clearing browser data would delete them, use the **.mmd** export
button to keep file backups of anything important — you can re-import those
files any time.

## Updating Mermaid

The Mermaid library is vendored at `vendor/mermaid.min.js` (v11.16.0,
[MIT-licensed](https://github.com/mermaid-js/mermaid/blob/develop/LICENSE))
so the editor works offline. To upgrade, replace it with a newer build:

```bash
curl -L -o vendor/mermaid.min.js https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js
```
