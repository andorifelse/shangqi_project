"""Standalone MVP editor and asynchronous sensor preview using shared core packages."""
from pathlib import Path
import argparse
import time
import viser
from gs_renderer.gaussian import GaussianScene
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from .editor import Editor
from .preview import Preview
from .files import SceneFiles
from scene_manager.persistence import load_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--backend", choices=("cpu","gsplat"), default="cpu")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[2] / "avm_bringup" / "example_assets"
    generate_assets(fixtures)
    scene = load_scene(args.scene) if args.scene else default_scene(fixtures)
    if args.ply:
        scene.gaussian_file = str(args.ply.resolve())
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="fixed", control_width="large", show_share_button=False)
    server.scene.add_grid("/grid", plane="xy", width=14, height=14)
    store = SceneStore(scene)
    editor = Editor(server, store)
    files = SceneFiles(server, store, editor.status, Path.cwd() / "outputs" / "scene.yaml")
    preview = None if args.no_render else Preview(server, store, args.backend)
    @server.on_client_connect
    def on_connect(client):
        client.camera.position = (10, -12, 10)
        client.camera.look_at = (0, 0, 0)
        client.camera.up_direction = (0, 0, 1)
    print(f"AVM viewer ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            files.tick()
            editor.tick()
            if preview:
                preview.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        pass
    finally:
        if preview:
            preview.close()
        server.stop()

