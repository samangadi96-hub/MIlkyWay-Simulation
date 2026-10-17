import math
from pathlib import Path
import numpy as np
import moderngl
import glm
from PIL import Image

from camera import Camera
from lod import LODManager
from galaxy import GalacticBulge, GalacticBar
from galactic_disk import GalacticDisk
from nebula import NebulaManager


def render_scene(ctx, fbo, width, height, disk, bar, bulge, test_cases, out_dir, prefix=""):
    aspect_ratio = width / height
    lod_mgr = LODManager()

    for tc in test_cases:
        cam = Camera()
        cam.distance = tc["dist"]
        cam.target_distance = tc["dist"]
        cam.inclination = tc["incl"]
        cam.target_inclination = tc["incl"]
        cam.azimuth = tc["azim"]
        cam.target_azimuth = tc["azim"]
        cam.update(0.016)

        lod_state = lod_mgr.get_state(cam.absolute_distance)
        star_vis = 1.0 - lod_state["near_weight"]
        if cam.absolute_distance < 11.0:
            def smoothstep_py(e0, e1, x):
                t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
                return t * t * (3.0 - 2.0 * t)
            base_at_11 = 1.0 - lod_mgr.get_state(11.0)["near_weight"]
            d = cam.absolute_distance
            if d < 4.0:
                t = smoothstep_py(2.0, 4.0, d)
                star_vis = t * 0.25
            elif d < 8.0:
                t = smoothstep_py(4.0, 8.0, d)
                star_vis = 0.25 * (1.0 - t) + 0.55 * t
            else:
                t = smoothstep_py(8.0, 11.0, d)
                star_vis = 0.55 * (1.0 - t) + base_at_11 * t

        fbo.use()
        ctx.clear(0.01, 0.01, 0.015, 1.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        ctx.blend_func = (ctx.SRC_ALPHA, ctx.ONE)

        disk.render(cam, aspect_ratio, star_vis, tc["rot"])
        bar.render(cam, aspect_ratio, star_vis, tc["rot"])
        bulge.render(cam, aspect_ratio, star_vis, tc["rot"])

        raw = fbo.read(components=3)
        img = Image.frombytes("RGB", (width, height), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        filename = f"{prefix}{tc['name']}.png"
        save_path = out_dir / filename
        img.save(save_path)
        print(f"Saved: {filename} (dist={tc['dist']:.1f}, incl={math.degrees(tc['incl']):.1f} deg)")


def run_tests():
    print("Initializing standalone ModernGL context...")
    ctx = moderngl.create_context(standalone=True)
    base_dir = Path(__file__).parent

    width, height = 1280, 720
    color_tex = ctx.texture((width, height), 4)
    depth_tex = ctx.depth_texture((width, height))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)

    out_dir = base_dir / "test_renders"
    out_dir.mkdir(exist_ok=True)

    test_cases = [
        {"name": "face_on_dist40", "incl": math.radians(15.0), "azim": 0.0, "dist": 40.0, "rot": 0.0},
        {"name": "oblique_dist35", "incl": math.radians(65.0), "azim": 0.0, "dist": 35.0, "rot": 0.0},
        {"name": "edge_on_dist35", "incl": math.radians(86.0), "azim": 0.0, "dist": 35.0, "rot": 0.0},
        {"name": "near_arm_dist12", "incl": math.radians(50.0), "azim": math.radians(45.0), "dist": 12.0, "rot": 0.0},
        {"name": "far_galaxy_dist80", "incl": math.radians(45.0), "azim": 0.0, "dist": 80.0, "rot": 0.0},
    ]

    print("\n--- 1. Testing Realistic Stellar Population Rendering ---")
    nebula_mgr = NebulaManager(diagnostic_mode=False)
    disk = GalacticDisk(ctx, base_dir, nebula_stars=nebula_mgr.get_stars_array(), diagnostic_mode=False)
    bar = GalacticBar(ctx, base_dir)
    bar.diagnostic_mode = False
    bulge = GalacticBulge(ctx, base_dir)
    bulge.diagnostic_mode = False

    print(f"Disk stars: {disk.num_stars:,}")
    print(f"Bar stars: {bar.num_stars:,}")
    print(f"Bulge stars: {bulge.num_stars:,}")
    print(f"Total stars: {disk.num_stars + bar.num_stars + bulge.num_stars:,}")

    render_scene(ctx, fbo, width, height, disk, bar, bulge, test_cases, out_dir, prefix="pop_")

    print("\n--- 2. Testing Diagnostic Population Classes (O=Magenta, B=Cyan, A=White, F=Yellow-Green, G=Yellow, K=Orange, M=Red) ---")
    diag_nebula = NebulaManager(diagnostic_mode=True)
    diag_disk = GalacticDisk(ctx, base_dir, nebula_stars=diag_nebula.get_stars_array(), diagnostic_mode=True)
    diag_bar = GalacticBar(ctx, base_dir, diagnostic_mode=True)
    diag_bulge = GalacticBulge(ctx, base_dir, diagnostic_mode=True)

    diag_test_cases = [
        {"name": "diagnostic_face_on", "incl": math.radians(15.0), "azim": 0.0, "dist": 40.0, "rot": 0.0},
        {"name": "diagnostic_oblique", "incl": math.radians(65.0), "azim": 0.0, "dist": 35.0, "rot": 0.0},
    ]
    render_scene(ctx, fbo, width, height, diag_disk, diag_bar, diag_bulge, diag_test_cases, out_dir, prefix="")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    run_tests()
