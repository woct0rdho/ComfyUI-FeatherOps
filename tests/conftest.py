import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = str(Path.home() / "ComfyUI")
for entry in (str(ROOT), COMFY):
    if entry not in sys.path:
        sys.path.insert(0, entry)
