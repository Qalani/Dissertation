#!/usr/bin/env sh
# Fetch a local copy of the Mermaid library so the editor works with no internet.
#
# The editor looks for ./vendor/mermaid.min.js first and only falls back to a CDN
# if that file is missing, so running this once is all it takes to go offline.
#
#   ./download-offline-copy.sh          # latest Mermaid 11.x
#   ./download-offline-copy.sh 11.16.0  # a specific version

set -eu

VERSION="${1:-11}"
DIR="$(cd "$(dirname "$0")" && pwd)/vendor"
URL="https://cdn.jsdelivr.net/npm/mermaid@${VERSION}/dist/mermaid.min.js"

mkdir -p "$DIR"
echo "Downloading $URL"
curl -fsSL "$URL" -o "$DIR/mermaid.min.js"

echo "Saved $(du -h "$DIR/mermaid.min.js" | cut -f1) to $DIR/mermaid.min.js"
echo "The editor will now use the local copy and needs no internet connection."
