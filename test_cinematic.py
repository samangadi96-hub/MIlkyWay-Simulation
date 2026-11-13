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

def test_cinematic_pipeline():
    print("Testing Cinematic Astrophotography Pipeline in ModernGL...")
    ctx = moderngl.create_context(standalone=True)
    base_dir = Path(__file__).parent
    
    width, height = 1280, 720
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4
    
    # 1. HDR Scene Framebuffer (RGBA16F)
    scene_tex = ctx.texture((width, height), 4, dtype='f2')
    scene_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    depth_tex = ctx.depth_texture((width, height))
    depth_tex.compare_func = ''
    fbo = ctx.framebuffer(color_attachments=[scene_tex], depth_attachment=depth_tex)
    
    # 2. Composited Scene Framebuffer (RGBA16F)
    comp_tex = ctx.texture((width, height), 4, dtype='f2')
    comp_tex.filter = (ctx.LINEAR, ctx.LINEAR)
    comp_fbo = ctx.framebuffer(color_attachments=[comp_tex])
    
    # 3. Bloom Framebuffers (RGBA16F)
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
    
    # Final Output FBO (for saving to disk)
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
    
    # Load shaders
    with open(base_dir / "shaders" / "post_vert.glsl", encoding="utf-8") as f:
        p_vert = f.read()
    with open(base_dir / "shaders" / "bloom_extract_frag.glsl", encoding="utf-8") as f:
        extract_frag = f.read()
    with open(base_dir / "shaders" / "bloom_blur_frag.glsl", encoding="utf-8") as f:
        blur_frag = f.read()
    with open(base_dir / "shaders" / "post_frag.glsl", encoding="utf-8") as f:
        p_frag = f.read()
        
    extract_prog = ctx.program(vertex_shader=p_vert, fragment_shader=extract_frag)
    blur_prog = ctx.program(vertex_shader=p_vert, fragment_shader=blur_frag)
    post_prog = ctx.program(vertex_shader=p_vert, fragment_shader=p_frag)
    
    extract_vao = ctx.vertex_array(extract_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    blur_vao = ctx.vertex_array(blur_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    post_vao = ctx.vertex_array(post_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
    
    out_dir = base_dir / "test_renders"
    out_dir.mkdir(exist_ok=True)
    
    nebula_mgr = NebulaManager(diagnostic_mode=False)
    disk = GalacticDisk(ctx, base_dir, nebula_stars=nebula_mgr.get_stars_array(), diagnostic_mode=False)
    bar = GalacticBar(ctx, base_dir)
    bulge = GalacticBulge(ctx, base_dir)
    
    print(f"Total stars: {disk.num_stars + bar.num_stars + bulge.num_stars:,}")
    
    test_views = [
        {"name": "astro_inclined_dist38", "incl": math.radians(62.0), "azim": math.radians(25.0), "dist": 38.0},
        {"name": "astro_face_on_dist42", "incl": math.radians(18.0), "azim": 0.0, "dist": 42.0},
        {"name": "astro_milky_way_edge_on", "incl": math.radians(82.0), "azim": math.radians(10.0), "dist": 34.0},
        {"name": "astro_core_zoom_dist14", "incl": math.radians(35.0), "azim": math.radians(45.0), "dist": 14.0},
    ]
    
    aspect_ratio = width / height
    
    for tv in test_views:
        cam = Camera()
        cam.distance = tv["dist"]
        cam.target_distance = tv["dist"]
        cam.inclination = tv["incl"]
        cam.target_inclination = tv["incl"]
        cam.azimuth = tv["azim"]
        cam.target_azimuth = tv["azim"]
        cam.update(0.016)
        
        # 1. Render Scene into HDR FBO
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        ctx.depth_mask = True
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (ctx.SRC_ALPHA, ctx.ONE)
        
        disk.render(cam, aspect_ratio, 1.0, 0.0, sim_time=0.0)
        bar.render(cam, aspect_ratio, 1.0, 0.0, sim_time=0.0)
        bulge.render(cam, aspect_ratio, 1.0, 0.0, sim_time=0.0)
        
        # Pass through to comp_fbo (simulating composite pass)
        comp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.disable(ctx.BLEND)
        scene_tex.use(location=0)
        
        # Simple copy to comp_fbo
        # In actual main.py, dust and nebulae are composited here
        quad_vao_copy = ctx.vertex_array(extract_prog, [(quad_buf, "2f 2f", "in_vert", "in_texcoord")])
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.0
        extract_prog["u_knee"].value = 0.01
        quad_vao_copy.render(mode=ctx.TRIANGLE_STRIP)
        
        # 2. Bloom Extraction Pass (half res)
        bloom_extract_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.45
        extract_prog["u_knee"].value = 0.30
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 3. Tight Bloom Blur (half res)
        # Horizontal
        bloom_half_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_half_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / hw, 0.0)
        blur_prog["u_radius"].value = 2.0
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # Vertical
        bloom_half_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_half_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / hh)
        blur_prog["u_radius"].value = 2.0
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 4. Broad Bloom Blur (quarter res)
        # Downsample tight bloom to quarter res
        bloom_down_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_half_tex.use(location=0)
        extract_prog["u_image"].value = 0
        extract_prog["u_threshold"].value = 0.0
        extract_prog["u_knee"].value = 0.01
        extract_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # Horizontal
        bloom_quarter_temp_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_quarter_tex.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (1.0 / qw, 0.0)
        blur_prog["u_radius"].value = 3.5
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # Vertical
        bloom_quarter_blur_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        bloom_quarter_temp.use(location=0)
        blur_prog["u_image"].value = 0
        blur_prog["u_direction"].value = (0.0, 1.0 / qh)
        blur_prog["u_radius"].value = 3.5
        blur_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        # 5. Final Post-processing + Tone Mapping Pass
        out_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp_tex.use(location=0)
        bloom_half_tex.use(location=1)
        bloom_quarter_tex.use(location=2)
        
        post_prog["u_scene"].value = 0
        post_prog["u_bloomTight"].value = 1
        post_prog["u_bloomBroad"].value = 2
        post_prog["u_exposure"].value = 1.10
        post_prog["u_bloomTightIntensity"].value = 0.65
        post_prog["u_bloomBroadIntensity"].value = 0.45
        post_prog["u_dustProtection"].value = 0.85
        
        post_vao.render(mode=ctx.TRIANGLE_STRIP)
        
        raw = out_fbo.read(components=3)
        img = Image.frombytes("RGB", (width, height), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        filename = f"{tv['name']}.png"
        img.save(out_dir / filename)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    test_cinematic_pipeline()
