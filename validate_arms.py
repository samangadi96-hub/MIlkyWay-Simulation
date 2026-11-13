import math
from pathlib import Path
import numpy as np
import moderngl
import glm
from PIL import Image, ImageDraw

from camera import Camera
from galaxy import GalacticBulge, GalacticBar
from galactic_disk import GalacticDisk
from nebula import NebulaManager


def run_validation():
    print("Running Full Astronomical Scene Validation at t = 0, 15, 30, 60, 120, 300...")
    ctx = moderngl.create_context(standalone=True)
    base_dir = Path(__file__).parent

    width, height = 1280, 720
    aspect = width / height
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4

    out_dir = base_dir / "test_renders"
    out_dir.mkdir(exist_ok=True)

    # 1. HDR Scene Textures
    scene_tex = ctx.texture((width, height), 4, dtype="f2")
    scene_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    depth_tex = ctx.depth_texture((width, height))
    depth_tex.compare_func = ""
    fbo = ctx.framebuffer(color_attachments=[scene_tex], depth_attachment=depth_tex)
    depth_fbo = ctx.framebuffer(depth_attachment=depth_tex)

    # 2. Dust (RG16F)
    dust_tex = ctx.texture((qw, qh), 2, dtype="f2")
    dust_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    dust_fbo = ctx.framebuffer(color_attachments=[dust_tex])

    # 3. Nebula (RGBA16F + R16F depth)
    nebula_col_tex = ctx.texture((qw, qh), 4, dtype="f2")
    nebula_col_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_depth_tex = ctx.texture((qw, qh), 1, dtype="f2")
    nebula_depth_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_fbo = ctx.framebuffer(color_attachments=[nebula_col_tex, nebula_depth_tex])

    # 4. Composite (RGBA16F)
    comp_tex = ctx.texture((width, height), 4, dtype="f2")
    comp_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    comp_fbo = ctx.framebuffer(color_attachments=[comp_tex])

    # 5. Bloom Pyramid
    b_half_tex = ctx.texture((hw, hh), 4, dtype="f2")
    b_half_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    b_half_temp = ctx.texture((hw, hh), 4, dtype="f2")
    b_half_temp.filter = (ctx.LINEAR, ctx.LINEAR)
    bloom_extract_fbo = ctx.framebuffer(color_attachments=[b_half_tex])
    bloom_half_temp_fbo = ctx.framebuffer(color_attachments=[b_half_temp])
    bloom_half_blur_fbo = ctx.framebuffer(color_attachments=[b_half_tex])

    b_quarter_tex = ctx.texture((qw, qh), 4, dtype="f2")
    b_quarter_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    b_quarter_temp = ctx.texture((qw, qh), 4, dtype="f2")
    b_quarter_temp.filter = (ctx.LINEAR, ctx.LINEAR)
    bloom_down_fbo = ctx.framebuffer(color_attachments=[b_quarter_tex])
    bloom_quarter_temp_fbo = ctx.framebuffer(color_attachments=[b_quarter_temp])
    bloom_quarter_blur_fbo = ctx.framebuffer(color_attachments=[b_quarter_tex])

    # 6. Final Output FBO
    out_tex = ctx.texture((width, height), 4)
    out_fbo = ctx.framebuffer(color_attachments=[out_tex])

    # Fullscreen quads
    quad_data = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype="f4")
    quad_buf = ctx.buffer(quad_data)
    main_quad_data = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    main_quad_buf = ctx.buffer(main_quad_data)

    # Load Shaders
    with open(base_dir / "shaders" / "dust_vol_vert.glsl", encoding="utf-8") as f:
        d_vert = f.read()
    with open(base_dir / "shaders" / "dust_vol_frag.glsl", encoding="utf-8") as f:
        d_frag = f.read()
    with open(base_dir / "shaders" / "nebula_vol_vert.glsl", encoding="utf-8") as f:
        n_vert = f.read()
    with open(base_dir / "shaders" / "nebula_vol_frag.glsl", encoding="utf-8") as f:
        n_frag = f.read()
    with open(base_dir / "shaders" / "dust_comp_vert.glsl", encoding="utf-8") as f:
        c_vert = f.read()
    with open(base_dir / "shaders" / "dust_comp_frag.glsl", encoding="utf-8") as f:
        c_frag = f.read()
    with open(base_dir / "shaders" / "post_vert.glsl", encoding="utf-8") as f:
        p_vert = f.read()
    with open(base_dir / "shaders" / "bloom_extract_frag.glsl", encoding="utf-8") as f:
        e_frag = f.read()
    with open(base_dir / "shaders" / "bloom_blur_frag.glsl", encoding="utf-8") as f:
        b_frag = f.read()
    with open(base_dir / "shaders" / "post_frag.glsl", encoding="utf-8") as f:
        p_frag = f.read()

    dust_prog = ctx.program(vertex_shader=d_vert, fragment_shader=d_frag)
    nebula_prog = ctx.program(vertex_shader=n_vert, fragment_shader=n_frag)
    comp_prog = ctx.program(vertex_shader=c_vert, fragment_shader=c_frag)
    extract_prog = ctx.program(vertex_shader=p_vert, fragment_shader=e_frag)
    blur_prog = ctx.program(vertex_shader=p_vert, fragment_shader=b_frag)
    post_prog = ctx.program(vertex_shader=p_vert, fragment_shader=p_frag)

    dust_vao = ctx.vertex_array(dust_prog, [(main_quad_buf, "2f", "in_position")])
    nebula_vao = ctx.vertex_array(nebula_prog, [(main_quad_buf, "2f", "in_position")])
    comp_vao = ctx.vertex_array(comp_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    extract_vao = ctx.vertex_array(extract_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    blur_vao = ctx.vertex_array(blur_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    post_vao = ctx.vertex_array(post_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])

    # Upload nebulae uniforms
    nebula_mgr = NebulaManager(diagnostic_mode=False)
    neb_pos, neb_col = nebula_mgr.get_shader_uniforms()
    if "u_numNebulae" in nebula_prog:
        nebula_prog["u_numNebulae"].value = nebula_mgr.num_nebulae
    pos_data = np.zeros(150 * 4, dtype="f4")
    col_data = np.zeros(150 * 4, dtype="f4")
    pos_data[:len(neb_pos)] = neb_pos
    col_data[:len(neb_col)] = neb_col
    if "u_nebulae_pos" in nebula_prog:
        nebula_prog["u_nebulae_pos"].write(pos_data.tobytes())
    if "u_nebulae_col" in nebula_prog:
        nebula_prog["u_nebulae_col"].write(col_data.tobytes())

    disk = GalacticDisk(ctx, base_dir, nebula_stars=nebula_mgr.get_stars_array(), diagnostic_mode=False)
    bar = GalacticBar(ctx, base_dir)
    bulge = GalacticBulge(ctx, base_dir)

    def render_frame(cam, sim_time, filename):
        rot = disk.pattern_speed * sim_time
        inv_model = glm.rotate(glm.mat4(1.0), -rot, glm.vec3(0.0, 1.0, 0.0))
        dist = cam.distance
        vp = cam.get_projection_matrix(aspect) * cam.get_view_matrix()

        # PASS 1A: Scene Depth
        depth_fbo.use()
        ctx.clear(depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        disk.render(cam, aspect, 1.0, rot, sim_time=sim_time)
        bar.render(cam, aspect, 1.0, rot, sim_time=sim_time)
        bulge.render(cam, aspect, 1.0, rot, sim_time=sim_time)

        # PASS 1B: Additive Color Accumulation
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.blend_func = (ctx.SRC_ALPHA, ctx.ONE)
        disk.render(cam, aspect, 1.0, rot, sim_time=sim_time)
        bar.render(cam, aspect, 1.0, rot, sim_time=sim_time)
        bulge.render(cam, aspect, 1.0, rot, sim_time=sim_time)

        # PASS 2: Dust Volume Pass (quarter res)
        dust_weight = 0.95
        dust_fbo.use()
        ctx.clear(1.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        if "u_camPos" in dust_prog: dust_prog["u_camPos"].value = cam.get_position()
        if "u_invVP" in dust_prog: dust_prog["u_invVP"].write(glm.inverse(vp))
        if "u_dustWeight" in dust_prog: dust_prog["u_dustWeight"].value = dust_weight
        if "u_galaxyModel" in dust_prog: dust_prog["u_galaxyModel"].write(inv_model)
        if "u_dustOpacity" in dust_prog: dust_prog["u_dustOpacity"].value = 1.2
        if "u_dustWidth" in dust_prog: dust_prog["u_dustWidth"].value = 0.25
        if "u_dustArmOffset" in dust_prog: dust_prog["u_dustArmOffset"].value = -0.12
        if "u_dustNoiseScale" in dust_prog: dust_prog["u_dustNoiseScale"].value = 0.8
        if "u_dustDensity" in dust_prog: dust_prog["u_dustDensity"].value = 2.5
        if "u_dustHeightScale" in dust_prog: dust_prog["u_dustHeightScale"].value = 1.0
        if "u_dustLaneWidth" in dust_prog: dust_prog["u_dustLaneWidth"].value = 0.06
        if "u_dustDiffuseWidth" in dust_prog: dust_prog["u_dustDiffuseWidth"].value = 0.18
        if "u_camDist" in dust_prog: dust_prog["u_camDist"].value = dist
        dust_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 3: Nebula Volume Pass (quarter res)
        neb_weight = 1.0
        nebula_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        if "u_camPos" in nebula_prog: nebula_prog["u_camPos"].value = cam.get_position()
        if "u_invVP" in nebula_prog: nebula_prog["u_invVP"].write(glm.inverse(vp))
        if "u_nebulaWeight" in nebula_prog: nebula_prog["u_nebulaWeight"].value = neb_weight
        if "u_galaxyModel" in nebula_prog: nebula_prog["u_galaxyModel"].write(inv_model)
        if "u_camDist" in nebula_prog: nebula_prog["u_camDist"].value = dist
        nebula_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 4: Composite Stars + Dust + Nebulae
        comp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        scene_tex.use(location=0)
        dust_tex.use(location=1)
        nebula_col_tex.use(location=2)
        nebula_depth_tex.use(location=3)
        depth_tex.use(location=4)
        comp_prog["u_scene"].value = 0
        comp_prog["u_dust"].value = 1
        comp_prog["u_nebulaColor"].value = 2
        comp_prog["u_nebulaDepth"].value = 3
        comp_prog["u_sceneDepth"].value = 4
        comp_prog["u_debugDust"].value = False
        comp_prog["u_debugNebula"].value = False
        comp_prog["u_nearPlane"].value = 0.01
        comp_prog["u_farPlane"].value = 10000.0
        if "u_camPos" in comp_prog: comp_prog["u_camPos"].value = cam.get_position()
        if "u_camForward" in comp_prog: comp_prog["u_camForward"].value = cam.get_forward()
        if "u_invVP" in comp_prog: comp_prog["u_invVP"].write(glm.inverse(vp))
        comp_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 5: Bloom Extract
        bloom_extract_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.85
        extract_prog["u_knee"].value = 0.25
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 6: Tight Blur
        bloom_half_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_half_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / hw, 0.0)
        blur_prog["u_radius"].value = 2.0
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)

        bloom_half_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_half_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / hh)
        blur_prog["u_radius"].value = 2.0
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 7: Broad Blur
        bloom_down_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_half_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.0
        extract_prog["u_knee"].value = 0.01
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)

        bloom_quarter_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_quarter_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / qw, 0.0)
        blur_prog["u_radius"].value = 3.5
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)

        bloom_quarter_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        b_quarter_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / qh)
        blur_prog["u_radius"].value = 3.5
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 8: Post Processing + ACES Tone Mapping
        out_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        b_half_tex.use(location=1)
        b_quarter_tex.use(location=2)
        post_prog["u_scene"].value = 0
        post_prog["u_bloomTight"].value = 1
        post_prog["u_bloomBroad"].value = 2
        post_prog["u_exposure"].value = 1.05
        post_prog["u_bloomTightIntensity"].value = 0.35
        post_prog["u_bloomBroadIntensity"].value = 0.18
        post_prog["u_dustProtection"].value = 0.88
        if "u_debugMode" in post_prog:
            post_prog["u_debugMode"].value = 0
        post_vao.render(mode=ctx.TRIANGLE_STRIP)

        raw = out_fbo.read(components=3)
        img = Image.frombytes("RGB", (width, height), raw).transpose(Image.FLIP_TOP_BOTTOM)
        img.save(out_dir / filename)
        print(f"Saved: {filename}")
        return img

    # Test cameras:
    cam_face = Camera()
    cam_face.distance = cam_face.target_distance = 42.0
    cam_face.inclination = cam_face.target_inclination = math.radians(20.0)
    cam_face.azimuth = cam_face.target_azimuth = 0.0
    cam_face.update(0.016)

    cam_oblique = Camera()
    cam_oblique.distance = cam_oblique.target_distance = 36.0
    cam_oblique.inclination = cam_oblique.target_inclination = math.radians(62.0)
    cam_oblique.azimuth = cam_oblique.target_azimuth = math.radians(25.0)
    cam_oblique.update(0.016)

    cam_edge = Camera()
    cam_edge.distance = cam_edge.target_distance = 34.0
    cam_edge.inclination = cam_edge.target_inclination = math.radians(84.0)
    cam_edge.azimuth = cam_edge.target_azimuth = math.radians(10.0)
    cam_edge.update(0.016)

    times = [0.0, 15.0, 30.0, 60.0, 120.0, 300.0]
    face_images = []
    for t in times:
        img = render_frame(cam_face, t, f"val_face_on_t{int(t)}.png")
        face_images.append(img)

    # Oblique & Edge-on checks
    for t in [0.0, 30.0, 60.0, 300.0]:
        render_frame(cam_oblique, t, f"val_oblique_t{int(t)}.png")
        render_frame(cam_edge, t, f"val_edge_on_t{int(t)}.png")

    # Build Side-by-Side Comparison Panel for Face-On at all 6 requested timestamps
    panel_w = 640
    panel_h = 360
    grid_img = Image.new("RGB", (panel_w * 3, panel_h * 2), color=(10, 10, 15))
    draw = ImageDraw.Draw(grid_img)

    labels = ["Sim Time: 0", "Sim Time: 15", "Sim Time: 30", "Sim Time: 60", "Sim Time: 120", "Sim Time: 300"]
    for i, (img, label) in enumerate(zip(face_images, labels)):
        thumb = img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        row = i // 3
        col = i % 3
        x = col * panel_w
        y = row * panel_h
        grid_img.paste(thumb, (x, y))
        draw.rectangle([x + 10, y + 10, x + 160, y + 36], fill=(0, 0, 0))
        draw.text((x + 16, y + 15), label, fill=(255, 230, 150))

    grid_img.save(out_dir / "val_face_on_comparison_grid.png")
    print("Saved: val_face_on_comparison_grid.png")
    print("Validation finished successfully!")


if __name__ == "__main__":
    run_validation()
