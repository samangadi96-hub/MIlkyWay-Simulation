import math
from pathlib import Path
import numpy as np
import moderngl
import glm
from PIL import Image

from blackhole import BlackHole
from camera import Camera
from lod import LODManager
from renderer import Renderer
from galaxy import GalacticBulge, GalacticBar
from galactic_disk import GalacticDisk
from nebula import NebulaManager
from far_galaxy import FarGalaxy

def test_full():
    print("Testing Complete Scene Pipeline (Stars + Far LOD + Black Hole + Dust + Nebulae + Bloom + ACES)...")
    ctx = moderngl.create_context(standalone=True)
    base_dir = Path(__file__).parent
    
    width, height = 1280, 720
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4
    
    # 1. Scene HDR (RGBA16F) and Depth Texture
    scene_tex = ctx.texture((width, height), 4, dtype='f2')
    scene_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    depth_tex = ctx.depth_texture((width, height))
    depth_tex.compare_func = ''
    fbo = ctx.framebuffer(color_attachments=[scene_tex], depth_attachment=depth_tex)
    depth_fbo = ctx.framebuffer(depth_attachment=depth_tex)
    
    # 2. Dust (RG16F)
    dust_tex = ctx.texture((qw, qh), 2, dtype='f2')
    dust_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    dust_fbo = ctx.framebuffer(color_attachments=[dust_tex])
    
    # 3. Nebula (RGBA16F + R16F depth)
    nebula_col_tex = ctx.texture((qw, qh), 4, dtype='f2')
    nebula_col_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_depth_tex = ctx.texture((qw, qh), 1, dtype='f2')
    nebula_depth_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    nebula_fbo = ctx.framebuffer(color_attachments=[nebula_col_tex, nebula_depth_tex])
    
    # 4. Comp HDR (RGBA16F)
    comp_tex = ctx.texture((width, height), 4, dtype='f2')
    comp_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    comp_fbo = ctx.framebuffer(color_attachments=[comp_tex])
    
    # 5. Bloom pyramid (RGBA16F)
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
    
    # 6. Final Output FBO
    out_tex = ctx.texture((width, height), 4)
    out_fbo = ctx.framebuffer(color_attachments=[out_tex])
    
    # Screen Quads
    quad_data = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype="f4")
    quad_buf = ctx.buffer(quad_data)
    main_quad_data = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    main_quad_buf = ctx.buffer(main_quad_data)
    
    # Black Hole Shaders
    with open(base_dir / "shaders" / "vertex.glsl", encoding="utf-8") as f:
        near_vert = f.read()
    with open(base_dir / "shaders" / "fragment.glsl", encoding="utf-8") as f:
        near_frag = f.read()
    near_prog = ctx.program(vertex_shader=near_vert, fragment_shader=near_frag)
    near_vao = ctx.vertex_array(near_prog, [(main_quad_buf, "2f", "in_position")])
    
    with open(base_dir / "shaders" / "medium_blackhole_vert.glsl", encoding="utf-8") as f:
        med_vert = f.read()
    with open(base_dir / "shaders" / "medium_blackhole_frag.glsl", encoding="utf-8") as f:
        med_frag = f.read()
    med_prog = ctx.program(vertex_shader=med_vert, fragment_shader=med_frag)
    med_vao = ctx.vertex_array(med_prog, [(main_quad_buf, "2f", "in_position")])
    
    with open(base_dir / "shaders" / "billboard_vert.glsl", encoding="utf-8") as f:
        far_vert = f.read()
    with open(base_dir / "shaders" / "billboard_frag.glsl", encoding="utf-8") as f:
        far_frag = f.read()
    billboard_prog = ctx.program(vertex_shader=far_vert, fragment_shader=far_frag)
    billboard_vao = ctx.vertex_array(billboard_prog, [(main_quad_buf, "2f", "in_position")])
    
    bh_renderer = Renderer(ctx, near_prog, near_vao, med_prog, med_vao, billboard_prog, billboard_vao)
    
    # Volume Shaders
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
    far_galaxy = FarGalaxy(ctx, base_dir, nebula_mgr=nebula_mgr, disk=disk, bar=bar, bulge=bulge)
    lod_mgr = LODManager()
    bh = BlackHole()
    
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
        # Preserved Black Hole views (Distances 3, 6, 9)
        {"name": "bh_preserve_dist3", "incl": math.radians(20.0), "azim": 0.0, "dist": 3.0},
        {"name": "bh_preserve_dist6", "incl": math.radians(30.0), "azim": math.radians(35.0), "dist": 6.0},
        {"name": "bh_preserve_dist9", "incl": math.radians(40.0), "azim": math.radians(20.0), "dist": 9.0},
        # Medium-distance structure (Distances 16, 28)
        {"name": "med_core_detail_dist16", "incl": math.radians(35.0), "azim": math.radians(45.0), "dist": 16.0},
        {"name": "med_inclined_spiral_dist28", "incl": math.radians(60.0), "azim": math.radians(25.0), "dist": 28.0},
        # LOD Blending Transition (Distance 45)
        {"name": "lod_smooth_blend_dist45", "incl": math.radians(52.0), "azim": math.radians(30.0), "dist": 45.0},
        # Far-distance galaxy LOD appearance (Face-on, edge-on, oblique)
        {"name": "far_galaxy_face_on_dist90", "incl": math.radians(1.0), "azim": 0.0, "dist": 90.0},
        {"name": "far_galaxy_face_on_ref_a", "incl": math.radians(1.0), "azim": 0.0, "rot": 0.0, "dist": 90.0},
        {"name": "far_galaxy_edge_on_dist65", "incl": math.radians(88.0), "azim": math.radians(10.0), "dist": 65.0},
        {"name": "far_galaxy_oblique_dist75", "incl": math.radians(64.0), "azim": math.radians(35.0), "dist": 75.0},
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
        lod_state = lod_mgr.get_state(dist)
        
        star_visibility = lod_state["star_visibility"]
        bulge_visibility = lod_state["bulge_visibility"]
        galaxy_far_weight = lod_state["galaxy_far_weight"]
        
        cam_pos = cam.get_position()
        cam_fwd = cam.get_forward()
        cam_right = cam.get_right()
        cam_up = cam.get_up()
        bh_cam_pos = (cam_pos[0] * 12.0, cam_pos[1] * 12.0, cam_pos[2] * 12.0)
        
        # 1A. Scene Depth Pass (Writes true nearest depth for scene occlusion)
        depth_fbo.use()
        ctx.clear(depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        
        rot = tv.get("rot", 0.0)
        if star_visibility > 0.001:
            disk.render(cam, aspect, 1.0, rot, sim_time=0.0)
            bar.render(cam, aspect, 1.0, rot, sim_time=0.0)
            bulge.render(cam, aspect, 1.0, rot, sim_time=0.0)
        if galaxy_far_weight > 0.001:
            far_galaxy.render_depth(cam, aspect, rot)
            
        # 1B. Color Accumulation Pass (Pure Additive: NO stars discarded by depth test)
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (ctx.SRC_ALPHA, ctx.ONE)
        
        if star_visibility > 0.001:
            disk.render(cam, aspect, star_visibility, rot, sim_time=0.0)
            bar.render(cam, aspect, star_visibility, rot, sim_time=0.0)
            bulge.render(cam, aspect, bulge_visibility, rot, sim_time=0.0)
        if galaxy_far_weight > 0.001:
            far_galaxy.render(cam, aspect, galaxy_far_weight, rot)
            
        # 1C. Black Hole Render (Near / Medium / Far Billboard)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.blend_func = (ctx.ONE, ctx.SRC_ALPHA)
        
        # Set black hole uniforms
        for prog in [near_prog, med_prog, billboard_prog]:
            if "aspectRatio" in prog: prog["aspectRatio"].value = aspect
            if "u_camDistance" in prog: prog["u_camDistance"].value = dist
            if "u_cameraPosition" in prog: prog["u_cameraPosition"].value = bh_cam_pos
            if "u_cameraForward" in prog: prog["u_cameraForward"].value = cam_fwd
            if "u_cameraRight" in prog: prog["u_cameraRight"].value = cam_right
            if "u_cameraUp" in prog: prog["u_cameraUp"].value = cam_up
            if "u_time" in prog: prog["u_time"].value = 0.0
            if "u_diskThickness" in prog: prog["u_diskThickness"].value = lod_state["disk_thickness"]
            if "u_glowIntensity" in prog: prog["u_glowIntensity"].value = lod_state["glow_intensity"]
            if "u_diskOuterRadius" in prog: prog["u_diskOuterRadius"].value = lod_state["disk_outer_radius"]
            if "u_innerDisk" in prog: prog["u_innerDisk"].value = bh.inner_disk_radius
            if "u_outerDisk" in prog: prog["u_outerDisk"].value = bh.outer_disk_radius
            if "u_volumetricDetail" in prog: prog["u_volumetricDetail"].value = lod_state["volumetric_detail"]
            if "u_proceduralStarWeight" in prog: prog["u_proceduralStarWeight"].value = lod_state["near_weight"]
            if "u_lodDetail" in prog: prog["u_lodDetail"].value = lod_state["ray_march_detail"]
            if "u_diskSquish" in prog: prog["u_diskSquish"].value = cam.disk_squish
            if "u_dopplerStrength" in prog: prog["u_dopplerStrength"].value = lod_state["doppler_strength"]
            if "u_innerRingStrength" in prog: prog["u_innerRingStrength"].value = lod_state["inner_ring_strength"]
            
        bh_renderer.render(lod_state)
        ctx.depth_mask = True
        
        # 2. Dust Volume Pass (quarter res)
        dust_weight = 0.0
        if dist > 8.0:
            if dist < 15.0:
                t = max(0.0, min(1.0, (dist - 8.0) / 7.0))
                dust_weight = t * t * (3.0 - 2.0 * t) * 0.4
            elif dist < 30.0:
                t = max(0.0, min(1.0, (dist - 15.0) / 15.0))
                dust_weight = 0.4 + t * t * (3.0 - 2.0 * t) * 0.5
            else:
                t = max(0.0, min(1.0, (dist - 30.0) / 20.0))
                dust_weight = 0.9 + t * t * (3.0 - 2.0 * t) * 0.1
        dust_weight *= (1.0 - galaxy_far_weight)
                
        dust_fbo.use()
        ctx.clear(1.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        vp = cam.get_projection_matrix(aspect) * cam.get_view_matrix()
        if dust_weight > 0.001:
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
        neb_weight = 0.0
        if dist > 8.0:
            if dist < 12.0:
                t = max(0.0, min(1.0, (dist - 8.0) / 4.0))
                neb_weight = t * t * (3.0 - 2.0 * t) * 0.3
            elif dist < 30.0:
                t = max(0.0, min(1.0, (dist - 12.0) / 18.0))
                neb_weight = 0.3 + t * t * (3.0 - 2.0 * t) * 0.7
            else:
                neb_weight = 1.0
        neb_weight *= (1.0 - galaxy_far_weight)
                
        nebula_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.BLEND)
        ctx.disable(ctx.DEPTH_TEST)
        if neb_weight > 0.001:
            if "u_camPos" in nebula_prog: nebula_prog["u_camPos"].value = cam.get_position()
            if "u_invVP" in nebula_prog: nebula_prog["u_invVP"].write(glm.inverse(vp))
            if "u_nebulaWeight" in nebula_prog: nebula_prog["u_nebulaWeight"].value = neb_weight
            if "u_galaxyModel" in nebula_prog: nebula_prog["u_galaxyModel"].write(glm.mat4(1.0))
            if "u_camDist" in nebula_prog: nebula_prog["u_camDist"].value = dist
            nebula_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 4. Composite Stars + Dust + Nebulae into comp_fbo
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
        if "u_camPos" in comp_prog:
            comp_prog["u_camPos"].value = cam.get_position()
        if "u_camForward" in comp_prog:
            comp_prog["u_camForward"].value = cam.get_forward()
        if "u_invVP" in comp_prog:
            comp_prog["u_invVP"].write(glm.inverse(vp))
        comp_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 5. Bloom Extraction (half res)
        bloom_extract_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.85
        extract_prog["u_knee"].value = 0.25
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
        
        # 7. Broad Bloom Blur (quarter res)
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
        
        # 8. Final Post-processing + Tone Mapping (ACES HDR)
        out_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        bloom_half_tex.use(location=1)
        bloom_quarter_tex.use(location=2)
        
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
        img = Image.frombytes("RGB", (width, height), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        filename = f"{tv['name']}.png"
        img.save(out_dir / filename)
        print(f"Saved: {filename} (dist={dist:.1f}, star_vis={star_visibility:.2f}, far_lod={galaxy_far_weight:.2f})")

if __name__ == "__main__":
    test_full()
