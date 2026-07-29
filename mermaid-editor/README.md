# Mermaid Flowchart Editor

A free, self-hosted editor for [mermaid-js](https://github.com/mermaid-js/mermaid) diagrams —
write Mermaid on the left, see the diagram on the right, save your work and export it.
No account, no subscription, no server: it is three static files, and everything runs in
your browser.

## Running it

**Quickest — just open it.** Double-click `index.html`, or drag it into a browser tab.
It needs an internet connection the first time so it can pull Mermaid from a CDN.

**As a local server** (nicer, and required if your browser blocks modules on `file://`):

```sh
cd mermaid-editor
python3 -m http.server 8000
# then visit http://localhost:8000
```

**Fully offline.** Download a local copy of the Mermaid library once:

```sh
cd mermaid-editor
./download-offline-copy.sh
```

That writes `vendor/mermaid.min.js` (~3.5 MB). The editor checks for it before trying any
CDN, so from then on it works with no connection at all. `vendor/` is git-ignored.

**On the web.** The folder is plain static files, so it can be served by GitHub Pages or any
static host. In the repository settings, set Pages to deploy from a branch and point it at
this folder, or copy the folder's contents into the published directory.

## What it does

- **Live preview** as you type, debounced, with parse errors shown in a panel and the failing
  line highlighted in the gutter.
- **Save diagrams** in the browser, with a searchable sidebar; rename, duplicate and delete
  from the list. Unsaved work is kept as a draft, so a reload does not lose it.
- **Export** to PNG (2× resolution), SVG, or `.mmd` source; copy the SVG markup or the code
  to the clipboard.
- **Shareable links** that pack the whole diagram into the URL (deflate-compressed), so you can
  send a diagram to someone without either of you having an account.
- **Import** by opening a `.mmd` file or dropping one onto the page.
- **Backup and restore** every saved diagram as a single JSON file.
- **Templates** for flowcharts, sequence, class, state, ER, Gantt, pie, mind map and git graphs.
- **Themes**: five Mermaid diagram themes, plus a light/dark interface.
- **Pan and zoom** the preview — scroll to zoom, drag to pan, `⤢` to fit.

### Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl`/`⌘` + `S` | Save the current diagram |
| `Ctrl`/`⌘` + `Enter` | Re-render now |
| `Ctrl`/`⌘` + `B` | Show/hide the saved diagram list |
| `Tab` / `Shift`+`Tab` | Indent / dedent the selected lines |

## Where your diagrams are stored

In this browser's `localStorage`, under the key `mermaid-editor.v1`. Nothing is uploaded
anywhere. The practical consequence: diagrams are tied to one browser on one machine, and
clearing site data deletes them.

So treat the browser as a scratchpad, not an archive. For anything that matters, use
**More → Back up all diagrams** for a JSON file, or export the `.mmd` source and commit it
next to the work it belongs to — `.mmd` files are plain text and diff cleanly in git.

## Notes on rendering

Labels are rendered as SVG `<text>` rather than embedded HTML (`htmlLabels: false`). This is
deliberate: browsers refuse to rasterise `foreignObject` content when an SVG is drawn to a
canvas, so HTML labels would silently disappear from PNG exports. The trade-off is that some
HTML-in-label tricks are unavailable; `<br/>` line breaks still work.

Mermaid runs with `securityLevel: 'strict'`, which sanitises HTML in diagram text.

### A `>` inside a label disappears

This catches people out on decision nodes. Mermaid strips a literal `>` from label text —
quoting the label does not help, and it happens at every security level, so it is Mermaid's
behaviour rather than a setting here. Write it as the HTML entity `&gt;`:

```
G{Confidence > 0.8?}      %% renders as "Confidence 0.8?"
G{Confidence &gt; 0.8?}   %% renders as "Confidence > 0.8?"
```

`<`, `&` and `%` come through literally and need no escaping. `\n` in a label gives a line break.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure |
| `styles.css` | Interface styling, light and dark |
| `app.js` | Editor, storage, rendering, export |
| `download-offline-copy.sh` | Fetches `vendor/mermaid.min.js` for offline use |

Mermaid itself is MIT-licensed and is loaded, not bundled — this folder contains no Mermaid
source code.
