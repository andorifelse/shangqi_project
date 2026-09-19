"""Run the same core packages without ROS, for development on non-ROS hosts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for package in (ROOT / "avm_sim_ws" / "src").iterdir():
    if package.is_dir():
        sys.path.insert(0, str(package))

if __name__ == "__main__":
    from scene_editor.app import main
    main()

