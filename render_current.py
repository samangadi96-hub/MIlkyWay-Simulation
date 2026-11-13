import math
from pathlib import Path
import numpy as np
import moderngl
import glm
from PIL import Image

from camera import Camera
from galaxy import GalacticBulge, GalacticBar
from galactic_disk import GalacticDisk
from nebula import NebulaManager


def render_all(prefix="current"):
    ctx = moderngl.create_context(standalone=True)
    base_dir = Path(__file__).parent
    out_dir = base_dir / "test_renders"
    out_dir.mkdir(exist_ok=True)

    width, height = 1280, 720
    aspect = width / height
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4

    scene_tex = ctx.texture((width, height), 4, dtype="f2")
    scene_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    depth_tex = ctx.depth_texture((width, height))
    depth_tex.compare_func = ""
    fbo = ctx.framebuffer(color_attachments=[scene_tex], depth_attachment=depth_tex)
    depth_fbo = ctx.framebuffer(depth_attachment=depth_tex)

    dust_tex = ctx.texture((qw, qh), 2, dtype="f2")
    dust_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    dust_fbo = ctx.framebuffer(color_attachments=[dust_tex])

    nebula_col_tex = ctx.texture((qw, qh), 4, dtype="f2")
    nebula_col_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_depth_tex = ctx.texture((qw, qh), 1, dtype="f2")
    nebula_depth_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_fbo = ctx.framebuffer(color_attachments=[nebula_col_tex, nebula_depth_tex])

    comp_tex = ctx.texture((width, height), 4, dtype="f2")
    comp_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    comp_fbo = ctx.framebuffer(color_attachments=[comp_tex])

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

    out_tex = ctx.texture((width, height), 4, dtype="f2")
    out_fbo = ctx.framebuffer(color_attachments=[out_tex])

    # Quads
    quad2 = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype="f4")
    quad_buf = ctx.buffer(quad2)

    quad_strip = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    quad_strip_buf = ctx.buffer(quad_strip)

    # Shaders
    with open(base_dir / "shaders" / "post_vert.glsl") as f: p_vert = f.read()
    with open(base_dir / "shaders" / "bloom_extract_frag.glsl") as f: extract_frag = f.read()
    with open(base_dir / "shaders" / "bloom_blur_frag.glsl") as f: blur_frag = f.read()
    with open(base_dir / "shaders" / "post_frag.glsl") as f: p_frag = f.read()
    with open(base_dir / "shaders" / "dust_vol_vert.glsl") as f: d_vert = f.read()
    with open(base_dir / "shaders" / "dust_vol_frag.glsl") as f: d_frag = f.read()
    with open(base_dir / "shaders" / "dust_comp_vert.glsl") as f: c_vert = f.read()
    with open(base_dir / "shaders" / "dust_comp_frag.glsl") as f: c_frag = f.read()
    with open(base_dir / "shaders" / "nebula_vol_vert.glsl") as f: n_vert = f.read()
    with open(base_dir / "shaders" / "nebula_vol_frag.glsl") as f: n_frag = f.read()

    post_prog = ctx.program(vertex_shader=p_vert, fragment_shader=p_frag)
    extract_prog = ctx.program(vertex_shader=p_vert, fragment_shader=extract_frag)
    blur_prog = ctx.program(vertex_shader=p_vert, fragment_shader=blur_frag)
    dust_prog = ctx.program(vertex_shader=d_vert, fragment_shader=d_frag)
    comp_prog = ctx.program(vertex_shader=c_vert, fragment_shader=c_frag)
    nebula_prog = ctx.program(vertex_shader=n_vert, fragment_shader=n_frag)

    post_vao = ctx.vertex_array(post_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    extract_vao = ctx.vertex_array(extract_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    blur_vao = ctx.vertex_array(blur_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    comp_vao = ctx.vertex_array(comp_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    dust_vao = ctx.vertex_array(dust_prog, [(quad_strip_buf, "2f", "in_position")])
    nebula_vao = ctx.vertex_array(nebula_prog, [(quad_strip_buf, "2f", "in_position")])

    nebula_mgr = NebulaManager()
    neb_pos, neb_col = nebula_mgr.get_shader_uniforms()
    if "u_numNebulae" in nebula_prog:
        nebula_prog["u_numNebulae"].value = nebula_mgr.num_nebulae
    pos_data = np.zeros(150 * 4, dtype="f4")
    col_data = np.zeros(150 * 4, dtype="f4")
    pos_data[:len(neb_pos)] = neb_pos
    col_data[:len(neb_col)] = neb_col
    if "u_nebulae_pos" in nebula_prog: nebula_prog["u_nebulae_pos"].write(pos_data.tobytes())
    if "u_nebulae_col" in nebula_prog: nebula_prog["u_nebulae_col"].write(col_data.tobytes())

    print("Loading Bulge, Bar, Disk...")
    bulge = GalacticBulge(ctx, base_dir)
    bar = GalacticBar(ctx, base_dir)
    disk = GalacticDisk(ctx, base_dir, nebula_stars=nebula_mgr.get_stars_array())

    def render_frame(cam, sim_t, filename):
        dist = cam.absolute_distance
        vp = cam.get_projection_matrix(aspect) * cam.get_view_matrix()

        # PASS 1A: Scene Depth
        depth_fbo.use()
        ctx.clear(depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        disk.render(cam, aspect, 1.0, 0.0, sim_time=sim_t)
        bar.render(cam, aspect, 1.0, 0.0, sim_time=sim_t)
        bulge.render(cam, aspect, 1.0, 0.0, sim_time=sim_t)

        # PASS 1B: Stellar Accumulation
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE
        disk.render(cam, aspect, 1.0, 0.0, sim_time=sim_t)
        bar.render(cam, aspect, 1.0, 0.0, sim_time=sim_t)
        bulge.render(cam, aspect, 1.0, 0.0, sim_time=sim_t)

        # PASS 1.5: Dust
        dust_weight = 1.0
        dust_fbo.use()
        ctx.clear(1.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        if "u_camPos" in dust_prog: dust_prog["u_camPos"].value = cam.get_position()
        if "u_invVP" in dust_prog: dust_prog["u_invVP"].write(glm.inverse(vp))
        if "u_dustWeight" in dust_prog: dust_prog["u_dustWeight"].value = dust_weight
        if "u_galaxyModel" in dust_prog: dust_prog["u_galaxyModel"].write(glm.mat4(1.0))
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

        # PASS 1.6: Nebula
        nebula_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        if "u_camPos" in nebula_prog: nebula_prog["u_camPos"].value = cam.get_position()
        if "u_invVP" in nebula_prog: nebula_prog["u_invVP"].write(glm.inverse(vp))
        if "u_nebulaWeight" in nebula_prog: nebula_prog["u_nebulaWeight"].value = 1.0
        if "u_galaxyModel" in nebula_prog: nebula_prog["u_galaxyModel"].write(glm.mat4(1.0))
        if "u_camDist" in nebula_prog: nebula_prog["u_camDist"].value = dist
        nebula_vao.render(mode=ctx.TRIANGLE_STRIP)

        # PASS 1.75: Composite
        comp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
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

        # PASS 2: Bloom Extract
        bloom_extract_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.85
        extract_prog["u_knee"].value = 0.25
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)

        # Tight Blur
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

        # Broad Blur
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

        # Post / ACES
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
        if "u_debugMode" in post_prog: post_prog["u_debugMode"].value = 0
        if "u_debugPost" in post_prog: post_prog["u_debugPost"].value = 0
        post_vao.render(mode=ctx.TRIANGLE_STRIP)

        raw = out_fbo.read(components=3)
        img = Image.frombytes("RGB", (width, height), raw).transpose(Image.FLIP_TOP_BOTTOM)
        img.save(out_dir / filename)
        print(f"Saved: {filename}")
        return img

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

    cam_close = Camera()
    cam_close.distance = cam_close.target_distance = 12.0
    cam_close.inclination = cam_close.target_inclination = math.radians(30.0)
    cam_close.azimuth = cam_close.target_azimuth = math.radians(15.0)
    cam_close.update(0.016)

    render_frame(cam_face, 0.0, f"{prefix}_face_on.png")
    render_frame(cam_oblique, 0.0, f"{prefix}_oblique.png")
    render_frame(cam_edge, 0.0, f"{prefix}_edge_on.png")
    render_frame(cam_close, 0.0, f"{prefix}_close.png")
    print(f"Done rendering {prefix} set!")


if __name__ == "__main__":
    render_all("after_v2")
