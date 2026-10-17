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


def render_scene(ctx, fbo, scene_tex, post_fbo, post_prog, extract_prog, blur_prog, quad_vao, width, height, disk, bar, bulge, test_cases, out_dir, prefix="", g_map=None, bloom_fbos=None):
    aspect_ratio = width / height
    lod_mgr = LODManager()

    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4
    
    b_extract_fbo, b_half_temp_fbo, b_half_blur_fbo, b_down_fbo, b_quarter_temp_fbo, b_quarter_blur_fbo = bloom_fbos["fbos"]
    b_half_tex, b_half_temp, b_quarter_tex, b_quarter_temp = bloom_fbos["textures"]

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

        # 1. Render scene into HDR FBO
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        ctx.blend_func = (ctx.SRC_ALPHA, ctx.ONE)

        sim_t = tc.get("sim_time", 0.0)
        disk.render(cam, aspect_ratio, star_vis, tc["rot"], sim_time=sim_t)
        bar.render(cam, aspect_ratio, star_vis, tc["rot"], sim_time=sim_t)
        bulge.render(cam, aspect_ratio, star_vis, tc["rot"], sim_time=sim_t)

        # 2. Extract bright highlights (half res)
        b_extract_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.disable(ctx.BLEND)
        scene_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.45
        extract_prog["u_knee"].value = 0.30
        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # 3. Tight Bloom Blur (half res)
        # Horizontal
        b_half_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_half_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / hw, 0.0)
        blur_prog["u_radius"].value = 2.0
        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # Vertical
        b_half_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_half_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / hh)
        blur_prog["u_radius"].value = 2.0
        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # 4. Broad Bloom Blur (quarter res)
        b_down_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_half_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.0
        extract_prog["u_knee"].value = 0.01
        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # Horizontal
        b_quarter_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_quarter_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / qw, 0.0)
        blur_prog["u_radius"].value = 3.5
        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # Vertical
        b_quarter_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_quarter_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / qh)
        blur_prog["u_radius"].value = 3.5
        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # 5. Post-processing pass: bloom + ACES filmic tonemapping
        post_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.disable(ctx.BLEND)
        scene_tex.use(location=0)
        b_half_tex.use(location=1)
        b_quarter_tex.use(location=2)

        post_prog["u_scene"].value = 0
        post_prog["u_bloomTight"].value = 1
        post_prog["u_bloomBroad"].value = 2
        post_prog["u_exposure"].value = 1.10
        post_prog["u_bloomTightIntensity"].value = 0.65
        post_prog["u_bloomBroadIntensity"].value = 0.45
        post_prog["u_dustProtection"].value = 0.85
        if "u_debugPost" in post_prog:
            post_prog["u_debugPost"].value = 0

        quad_vao.render(mode=ctx.TRIANGLE_STRIP)

        # Scientific map overlay pass
        if g_map is not None:
            g_map.render(cam, aspect_ratio, tc["rot"])

        raw = post_fbo.read(components=3)
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
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4

    scene_tex = ctx.texture((width, height), 4, dtype='f2')
    scene_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    depth_tex = ctx.depth_texture((width, height))
    depth_tex.compare_func = ''
    fbo = ctx.framebuffer(color_attachments=[scene_tex], depth_attachment=depth_tex)

    # Post FBO
    post_tex = ctx.texture((width, height), 4)
    post_fbo = ctx.framebuffer(color_attachments=[post_tex])

    # Multi-scale Bloom textures and FBOs (RGBA16F)
    bloom_half_tex = ctx.texture((hw, hh), 4, dtype='f2')
    bloom_half_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    bloom_half_temp = ctx.texture((hw, hh), 4, dtype='f2')
    bloom_half_temp.filter = (ctx.LINEAR, ctx.LINEAR)
    bloom_extract_fbo = ctx.framebuffer(color_attachments=[bloom_half_tex])
    bloom_half_temp_fbo = ctx.framebuffer(color_attachments=[bloom_half_temp])
    bloom_half_blur_fbo = ctx.framebuffer(color_attachments=[bloom_half_tex])

    bloom_quarter_tex = ctx.texture((qw, qh), 4, dtype='f2')
    bloom_quarter_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    bloom_quarter_temp = ctx.texture((qw, qh), 4, dtype='f2')
    bloom_quarter_temp.filter = (ctx.LINEAR, ctx.LINEAR)
    bloom_down_fbo = ctx.framebuffer(color_attachments=[bloom_quarter_tex])
    bloom_quarter_temp_fbo = ctx.framebuffer(color_attachments=[bloom_quarter_temp])
    bloom_quarter_blur_fbo = ctx.framebuffer(color_attachments=[bloom_quarter_tex])

    bloom_fbos = {
        "fbos": (bloom_extract_fbo, bloom_half_temp_fbo, bloom_half_blur_fbo, bloom_down_fbo, bloom_quarter_temp_fbo, bloom_quarter_blur_fbo),
        "textures": (bloom_half_tex, bloom_half_temp, bloom_quarter_tex, bloom_quarter_temp),
    }

    with open(base_dir / "shaders" / "post_vert.glsl", encoding="utf-8") as f:
        p_vert = f.read()
    with open(base_dir / "shaders" / "post_frag.glsl", encoding="utf-8") as f:
        p_frag = f.read()
    with open(base_dir / "shaders" / "bloom_extract_frag.glsl", encoding="utf-8") as f:
        extract_frag = f.read()
    with open(base_dir / "shaders" / "bloom_blur_frag.glsl", encoding="utf-8") as f:
        blur_frag = f.read()

    post_prog = ctx.program(vertex_shader=p_vert, fragment_shader=p_frag)
    extract_prog = ctx.program(vertex_shader=p_vert, fragment_shader=extract_frag)
    blur_prog = ctx.program(vertex_shader=p_vert, fragment_shader=blur_frag)

    quad_data = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype="f4")
    quad_buf = ctx.buffer(quad_data)
    quad_vao = ctx.vertex_array(post_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])

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

    render_scene(ctx, fbo, scene_tex, post_fbo, post_prog, extract_prog, blur_prog, quad_vao, width, height, disk, bar, bulge, test_cases, out_dir, prefix="pop_", bloom_fbos=bloom_fbos)

    print("\n--- 2. Testing Diagnostic Population Classes (O=Magenta, B=Cyan, A=White, F=Yellow-Green, G=Yellow, K=Orange, M=Red) ---")
    diag_nebula = NebulaManager(diagnostic_mode=True)
    diag_disk = GalacticDisk(ctx, base_dir, nebula_stars=diag_nebula.get_stars_array(), diagnostic_mode=True)
    diag_bar = GalacticBar(ctx, base_dir, diagnostic_mode=True)
    diag_bulge = GalacticBulge(ctx, base_dir, diagnostic_mode=True)

    diag_test_cases = [
        {"name": "diagnostic_face_on", "incl": math.radians(15.0), "azim": 0.0, "dist": 40.0, "rot": 0.0},
        {"name": "diagnostic_oblique", "incl": math.radians(65.0), "azim": 0.0, "dist": 35.0, "rot": 0.0},
    ]
    render_scene(ctx, fbo, scene_tex, post_fbo, post_prog, extract_prog, blur_prog, quad_vao, width, height, diag_disk, diag_bar, diag_bulge, diag_test_cases, out_dir, prefix="", bloom_fbos=bloom_fbos)

    print("\n--- 3. Testing Scientific Face-on Galactic Map Overlay (Key M) ---")
    from galactic_map import GalacticMap
    g_map = GalacticMap(ctx, base_dir)
    g_map.enabled = True

    map_test_cases = [
        {"name": "scientific_map_face_on", "incl": math.radians(2.0), "azim": 0.0, "dist": 42.0, "rot": 0.0},
        {"name": "scientific_map_rotated", "incl": math.radians(2.0), "azim": 0.0, "dist": 42.0, "rot": math.radians(35.0)},
        {"name": "scientific_map_full_galaxy", "incl": math.radians(2.0), "azim": 0.0, "dist": 110.0, "rot": 0.0},
    ]
    render_scene(ctx, fbo, scene_tex, post_fbo, post_prog, extract_prog, blur_prog, quad_vao, width, height, disk, bar, bulge, map_test_cases, out_dir, prefix="", g_map=g_map, bloom_fbos=bloom_fbos)

    print("\n--- 4. Testing Differential Galactic Rotation Over Time ---")
    # Verify rotation curve analytically
    r_test = [1.0, 3.5, 7.5, 15.6, 25.0, 35.0, 45.0]
    print(f"Rotation curve params: v_flat={disk.v_flat:.2f}, r_core={disk.r_core:.2f}, Omega_p={disk.pattern_speed:.4f} rad/s")
    print(f"{'Radius':<8} | {'v(r)':<8} | {'omega(r)':<10} | {'T_orb (s)':<10} | {'omega - Omega_p':<16}")
    print("-" * 65)
    for r in r_test:
        vr = disk.v_flat * r / math.sqrt(r * r + disk.r_core * disk.r_core)
        w = disk.v_flat / math.sqrt(r * r + disk.r_core * disk.r_core)
        t_orb = 2 * math.pi / w
        diff = w - disk.pattern_speed
        print(f"{r:<8.1f} | {vr:<8.4f} | {w:<10.4f} | {t_orb:<10.1f} | {diff:<+16.4f}")

    # Differential rotation render tests
    diff_test_cases = [
        {"name": "diff_rotation_t0", "incl": math.radians(20.0), "azim": 0.0, "dist": 40.0, "rot": 0.0, "sim_time": 0.0},
        {"name": "diff_rotation_t15", "incl": math.radians(20.0), "azim": 0.0, "dist": 40.0, "rot": disk.pattern_speed * 15.0, "sim_time": 15.0},
        {"name": "diff_rotation_t30", "incl": math.radians(20.0), "azim": 0.0, "dist": 40.0, "rot": disk.pattern_speed * 30.0, "sim_time": 30.0},
        {"name": "diff_rotation_t60", "incl": math.radians(20.0), "azim": 0.0, "dist": 40.0, "rot": disk.pattern_speed * 60.0, "sim_time": 60.0},
        {"name": "diff_oblique_t30", "incl": math.radians(60.0), "azim": 0.0, "dist": 35.0, "rot": disk.pattern_speed * 30.0, "sim_time": 30.0},
    ]
    render_scene(ctx, fbo, scene_tex, post_fbo, post_prog, extract_prog, blur_prog, quad_vao, width, height, disk, bar, bulge, diff_test_cases, out_dir, prefix="", bloom_fbos=bloom_fbos)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    run_tests()
