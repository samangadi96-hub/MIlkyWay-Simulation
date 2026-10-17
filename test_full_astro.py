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

def test_full():
    print("Testing Complete Scene Pipeline (Stars + Dust + Nebulae + Bloom + ACES)...")
    ctx = moderngl.create_context(standalone=True)
    base_dir = Path(__file__).parent
    
    width, height = 1280, 720
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4
    
    # Scene HDR (RGBA16F)
    scene_tex = ctx.texture((width, height), 4, dtype='f2')
    scene_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    depth_tex = ctx.depth_texture((width, height))
    depth_tex.compare_func = ''
    fbo = ctx.framebuffer(color_attachments=[scene_tex], depth_attachment=depth_tex)
    
    # Dust (RG16F)
    dust_tex = ctx.texture((qw, qh), 2, dtype='f2')
    dust_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    dust_fbo = ctx.framebuffer(color_attachments=[dust_tex])
    
    # Nebula (RGBA16F + R16F depth)
    nebula_col_tex = ctx.texture((qw, qh), 4, dtype='f2')
    nebula_col_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_depth_tex = ctx.texture((qw, qh), 1, dtype='f2')
    nebula_depth_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_fbo = ctx.framebuffer(color_attachments=[nebula_col_tex, nebula_depth_tex])
    
    # Comp HDR (RGBA16F)
    comp_tex = ctx.texture((width, height), 4, dtype='f2')
    comp_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    comp_fbo = ctx.framebuffer(color_attachments=[comp_tex])
    
    # Bloom pyramid (RGBA16F)
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
    
    # Final Output FBO
    out_tex = ctx.texture((width, height), 4)
    out_fbo = ctx.framebuffer(color_attachments=[out_tex])
    
    # Shaders
    quad_data = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype="f4")
    quad_buf = ctx.buffer(quad_data)
    main_quad_data = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    main_quad_buf = ctx.buffer(main_quad_data)
    
    with open(base_dir / "shaders" / "dust_vol_vert.glsl", encoding="utf-8") as f:
        d_vert = f.read()
    with open(base_dir / "shaders" / "dust_vol_frag.glsl", encoding="utf-8") as f:
        d_frag = f.read()
    dust_prog = ctx.program(vertex_shader=d_vert, fragment_shader=d_frag)
    dust_vao = ctx.vertex_array(dust_prog, [(main_quad_buf, "2f", "in_position")])
    
    with open(base_dir / "shaders" / "nebula_vol_vert.glsl", encoding="utf-8") as f:
        n_vert = f.read()
    with open(base_dir / "shaders" / "nebula_vol_frag.glsl", encoding="utf-8") as f:
        n_frag = f.read()
    nebula_prog = ctx.program(vertex_shader=n_vert, fragment_shader=n_frag)
    nebula_vao = ctx.vertex_array(nebula_prog, [(main_quad_buf, "2f", "in_position")])
    
    with open(base_dir / "shaders" / "dust_comp_vert.glsl", encoding="utf-8") as f:
        c_vert = f.read()
    with open(base_dir / "shaders" / "dust_comp_frag.glsl", encoding="utf-8") as f:
        c_frag = f.read()
    comp_prog = ctx.program(vertex_shader=c_vert, fragment_shader=c_frag)
    comp_vao = ctx.vertex_array(comp_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    
    with open(base_dir / "shaders" / "post_vert.glsl", encoding="utf-8") as f:
        p_vert = f.read()
    with open(base_dir / "shaders" / "bloom_extract_frag.glsl", encoding="utf-8") as f:
        e_frag = f.read()
    with open(base_dir / "shaders" / "bloom_blur_frag.glsl", encoding="utf-8") as f:
        b_frag = f.read()
    with open(base_dir / "shaders" / "post_frag.glsl", encoding="utf-8") as f:
        p_frag = f.read()
    extract_prog = ctx.program(vertex_shader=p_vert, fragment_shader=e_frag)
    blur_prog = ctx.program(vertex_shader=p_vert, fragment_shader=b_frag)
    post_prog = ctx.program(vertex_shader=p_vert, fragment_shader=p_frag)
    
    extract_vao = ctx.vertex_array(extract_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    blur_vao = ctx.vertex_array(blur_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    post_vao = ctx.vertex_array(post_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    
    nebula_mgr = NebulaManager(diagnostic_mode=False)
    disk = GalacticDisk(ctx, base_dir, nebula_stars=nebula_mgr.get_stars_array(), diagnostic_mode=False)
    bar = GalacticBar(ctx, base_dir)
    bulge = GalacticBulge(ctx, base_dir)
    
    # Nebula uniforms
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
            
    out_dir = base_dir / "test_renders"
    out_dir.mkdir(exist_ok=True)
    
    test_views = [
        {"name": "astro_full_inclined_m31", "incl": math.radians(64.0), "azim": math.radians(28.0), "dist": 38.0},
        {"name": "astro_full_face_on_ngc1365", "incl": math.radians(15.0), "azim": 0.0, "dist": 40.0},
        {"name": "astro_full_milky_way_edge_on", "incl": math.radians(84.0), "azim": math.radians(12.0), "dist": 32.0},
        {"name": "astro_full_core_bar_detail", "incl": math.radians(35.0), "azim": math.radians(45.0), "dist": 16.0},
    ]
    
    aspect = width / height
    
    for tv in test_views:
        cam = Camera()
        cam.distance = tv["dist"]
        cam.target_distance = tv["dist"]
        cam.inclination = tv["incl"]
        cam.target_inclination = tv["incl"]
        cam.azimuth = tv["azim"]
        cam.target_azimuth = tv["azim"]
        cam.update(0.016)
        
        dist = cam.absolute_distance
        
        # 1. Scene Stars Pass (HDR RGBA16F)
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        ctx.blend_func = (ctx.SRC_ALPHA, ctx.ONE)
        
        disk.render(cam, aspect, 1.0, 0.0, sim_time=0.0)
        bar.render(cam, aspect, 1.0, 0.0, sim_time=0.0)
        bulge.render(cam, aspect, 1.0, 0.0, sim_time=0.0)
        
        # 2. Dust Volume Pass (quarter res)
        dust_weight = 1.0 if dist > 30.0 else max(0.0, (dist - 8.0) / 22.0)
        dust_fbo.use()
        ctx.clear(1.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        vp = cam.get_projection_matrix(aspect) * cam.get_view_matrix()
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
        
        # 3. Nebula Volume Pass (quarter res)
        neb_weight = 1.0 if dist > 30.0 else max(0.0, (dist - 8.0) / 22.0)
        nebula_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        if "u_camPos" in nebula_prog: nebula_prog["u_camPos"].value = cam.get_position()
        if "u_invVP" in nebula_prog: nebula_prog["u_invVP"].write(glm.inverse(vp))
        if "u_nebulaWeight" in nebula_prog: nebula_prog["u_nebulaWeight"].value = neb_weight
        if "u_galaxyModel" in nebula_prog: nebula_prog["u_galaxyModel"].write(glm.mat4(1.0))
        if "u_camDist" in nebula_prog: nebula_prog["u_camDist"].value = dist
        nebula_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 4. Composite Stars + Dust + Nebulae into comp_fbo
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
        comp_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 5. Bloom Extraction (half res)
        bloom_extract_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.28
        extract_prog["u_knee"].value = 0.22
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 6. Tight Bloom Blur
        bloom_half_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_half_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / hw, 0.0)
        blur_prog["u_radius"].value = 2.0
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        bloom_half_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_half_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / hh)
        blur_prog["u_radius"].value = 2.0
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 7. Broad Bloom Blur
        bloom_down_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_half_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.0
        extract_prog["u_knee"].value = 0.01
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        bloom_quarter_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_quarter_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / qw, 0.0)
        blur_prog["u_radius"].value = 3.5
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        bloom_quarter_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_quarter_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / qh)
        blur_prog["u_radius"].value = 3.5
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 8. Post-processing + ACES filmic tonemapping
        out_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        bloom_half_tex.use(location=1)
        bloom_quarter_tex.use(location=2)
        post_prog["u_scene"].value = 0
        post_prog["u_bloomTight"].value = 1
        post_prog["u_bloomBroad"].value = 2
        post_prog["u_exposure"].value = 1.15
        post_prog["u_bloomTightIntensity"].value = 0.60
        post_prog["u_bloomBroadIntensity"].value = 0.40
        post_prog["u_dustProtection"].value = 0.88
        post_prog["u_debugPost"].value = 0
        post_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        raw = out_fbo.read(components=3)
        img = Image.frombytes("RGB", (width, height), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        fn = f"{tv['name']}.png"
        img.save(out_dir / fn)
        print(f"Saved: {fn}")

if __name__ == "__main__":
    test_full()
