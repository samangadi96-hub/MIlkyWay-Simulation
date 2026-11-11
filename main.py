import math
from pathlib import Path

import moderngl_window as mglw
import numpy as np

from blackhole import BlackHole
BLACK_HOLE_LOCAL_SCALE = BlackHole.BLACK_HOLE_LOCAL_SCALE
from camera import Camera
import glm
from lod import LODManager
from renderer import Renderer
from galaxy import GalacticBulge, GalacticBar
from galactic_disk import GalacticDisk
from nebula import NebulaManager
from galactic_map import GalacticMap
from far_galaxy import FarGalaxy
from ism import ISMVolumeManager
from s_stars import SStarCluster

class MilkyWaySimulation(mglw.WindowConfig):

    gl_version = (3, 3)
    title = "Milky Way Simulation"

    window_size = (1280, 720)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ==========================================================
        # Scene Objects
        # ==========================================================

        self.blackhole   = BlackHole()
        self.camera      = Camera()
        self.lod_manager = LODManager()
        self.bulge       = GalacticBulge(self.ctx, Path(__file__).parent)
        self.bar         = GalacticBar(self.ctx, Path(__file__).parent)
        self.nebula_mgr  = NebulaManager()
        self.galactic_disk = GalacticDisk(self.ctx, Path(__file__).parent, nebula_stars=self.nebula_mgr.get_stars_array())
        self.galactic_map  = GalacticMap(self.ctx, Path(__file__).parent)
        self.far_galaxy    = FarGalaxy(self.ctx, Path(__file__).parent, nebula_mgr=self.nebula_mgr, disk=self.galactic_disk, bar=self.bar, bulge=self.bulge)
        self.ism           = ISMVolumeManager(self.ctx, nebula_mgr=self.nebula_mgr)
        self.s_stars       = SStarCluster(self.ctx, Path(__file__).parent, bh_mass=self.blackhole.mass)
        
        # ==========================================================
        # Differential Galactic Rotation & Simulation Clock
        # ==========================================================
        self.sim_time = 0.0
        self.s_star_years = 0.0
        self.s_star_time_scale = 2.0  # S-cluster years per real second at 1x speed
        self.sim_speed = 1.0
        self.sim_paused = False
        self.v_flat = self.galactic_disk.v_flat
        self.r_core = self.galactic_disk.r_core
        self.pattern_speed = self.galactic_disk.pattern_speed
        self.galaxy_rotation_angle = 0.0

        # Keyboard state flags
        self.move_up    = False
        self.move_down  = False
        self.move_left  = False
        self.move_right = False
        self._zoom_in   = False
        self._zoom_out  = False
        self.debug_dust = False
        self.debug_nebula = False
        self.debug_post_mode = 0

        print()
        print('  Milky Way Simulation - Controls')
        print('  --------------------------------')
        print('  Left Drag        -> orbit galaxy')
        print('  Middle Drag      -> pan camera')
        print('  Mouse Scroll     -> zoom in / out')
        print('  W/S / Up/Down    -> tilt camera')
        print('  A/D / Left/Right -> rotate camera')
        print('  Q / E            -> zoom in / out')
        print('  R                -> reset camera')
        print('  SPACE            -> pause / resume orbital simulation')
        print('  [ / ]            -> decrease / increase simulation speed')
        print('  Backspace / 0    -> reset simulation clock')
        print('  T                -> toggle dust debug')
        print('  N                -> toggle nebula debug')
        print('  B                -> cycle post-processing debug views')
        print('  M                -> toggle scientific galactic map overlay')
        print('  O                -> toggle Sgr A* S-star cluster & orbits')
        print('  Y                -> toggle S-star identification labels & legend')
        print('  --------------------------------')
        print()

        BASE_DIR = Path(__file__).parent

        # ==========================================================
        # Load Main Shaders
        # ==========================================================

        with open(BASE_DIR / "shaders" / "vertex.glsl", encoding="utf-8") as f:
            vertex_shader = f.read()

        with open(BASE_DIR / "shaders" / "fragment.glsl", encoding="utf-8") as f:
            fragment_shader = f.read()

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        # ==========================================================
        # Fullscreen Quad
        # ==========================================================

        quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype="f4")

        self.main_quad_buffer = self.ctx.buffer(quad)

        self.main_quad_vao = self.ctx.vertex_array(
            self.program,
            [
                (self.main_quad_buffer, "2f", "in_position")
            ],
        )

        # ==========================================================
        # Medium Black Hole Shader
        # ==========================================================
        with open(BASE_DIR / "shaders" / "medium_blackhole_vert.glsl", encoding="utf-8") as f:
            medium_vert = f.read()
        with open(BASE_DIR / "shaders" / "medium_blackhole_frag.glsl", encoding="utf-8") as f:
            medium_frag = f.read()
        self.medium_program = self.ctx.program(
            vertex_shader=medium_vert,
            fragment_shader=medium_frag,
        )
        self.medium_vao = self.ctx.vertex_array(
            self.medium_program,
            [(self.main_quad_buffer, "2f", "in_position")]
        )

        # ==========================================================
        # Billboard Shader
        # ==========================================================
        with open(BASE_DIR / "shaders" / "billboard_vert.glsl", encoding="utf-8") as f:
            billboard_vert = f.read()
        with open(BASE_DIR / "shaders" / "billboard_frag.glsl", encoding="utf-8") as f:
            billboard_frag = f.read()
        self.billboard_program = self.ctx.program(
            vertex_shader=billboard_vert,
            fragment_shader=billboard_frag,
        )
        self.billboard_vao = self.ctx.vertex_array(
            self.billboard_program,
            [(self.main_quad_buffer, "2f", "in_position")]
        )
        

        # ==========================================================
        # Create Framebuffer
        # ==========================================================

        self.create_framebuffer(*self.wnd.size)

        # ==========================================================
        # Load Post Processing Shaders
        # ==========================================================

        with open(BASE_DIR / "shaders" / "post_vert.glsl", encoding="utf-8") as f:
            post_vert = f.read()

        with open(BASE_DIR / "shaders" / "post_frag.glsl", encoding="utf-8") as f:
            post_frag = f.read()

        self.post_program = self.ctx.program(
            vertex_shader=post_vert,
            fragment_shader=post_frag,
        )

        quad2 = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype="f4")

        self.quad_buffer = self.ctx.buffer(quad2)

        self.quad_vao = self.ctx.vertex_array(
            self.post_program,
            [
                (self.quad_buffer, "2f 2f", "in_vert", "in_texcoord")
            ],
        )

        with open(BASE_DIR / "shaders" / "bloom_extract_frag.glsl", encoding="utf-8") as f:
            extract_frag = f.read()

        with open(BASE_DIR / "shaders" / "bloom_blur_frag.glsl", encoding="utf-8") as f:
            blur_frag = f.read()

        self.extract_program = self.ctx.program(
            vertex_shader=post_vert,
            fragment_shader=extract_frag,
        )
        self.blur_program = self.ctx.program(
            vertex_shader=post_vert,
            fragment_shader=blur_frag,
        )

        self.extract_vao = self.ctx.vertex_array(
            self.extract_program,
            [(self.quad_buffer, "2f 2f", "in_vert", "in_texcoord")]
        )
        self.blur_vao = self.ctx.vertex_array(
            self.blur_program,
            [(self.quad_buffer, "2f 2f", "in_vert", "in_texcoord")]
        )

        # ==========================================================
        # Galaxy 3D-Star Radiance Resolve Shader
        # ==========================================================
        with open(BASE_DIR / "shaders" / "galaxy_resolve_vert.glsl", encoding="utf-8") as f:
            galaxy_resolve_vert = f.read()

        with open(BASE_DIR / "shaders" / "galaxy_resolve_frag.glsl", encoding="utf-8") as f:
            galaxy_resolve_frag = f.read()

        self.galaxy_resolve_program = self.ctx.program(
            vertex_shader=galaxy_resolve_vert,
            fragment_shader=galaxy_resolve_frag,
        )
        self.galaxy_resolve_vao = self.ctx.vertex_array(
            self.galaxy_resolve_program,
            [(self.quad_buffer, "2f 2f", "in_vert", "in_texcoord")]
        )

        # ==========================================================
        # Dust Volume Shader (quarter-res)
        # ==========================================================
        with open(BASE_DIR / "shaders" / "dust_vol_vert.glsl", encoding="utf-8") as f:
            vol_vert = f.read()
        with open(BASE_DIR / "shaders" / "dust_vol_frag.glsl", encoding="utf-8") as f:
            vol_frag = f.read()
        self.dust_program = self.ctx.program(
            vertex_shader=vol_vert, fragment_shader=vol_frag,
        )
        self.dust_vao = self.ctx.vertex_array(
            self.dust_program, [(self.main_quad_buffer, "2f", "in_position")]
        )

        # Composite shader
        with open(BASE_DIR / "shaders" / "dust_comp_vert.glsl", encoding="utf-8") as f:
            comp_vert = f.read()
        with open(BASE_DIR / "shaders" / "dust_comp_frag.glsl", encoding="utf-8") as f:
            comp_frag = f.read()
        self.comp_program = self.ctx.program(
            vertex_shader=comp_vert, fragment_shader=comp_frag,
        )
        self.comp_vao = self.ctx.vertex_array(
            self.comp_program, [(self.quad_buffer, "2f 2f", "in_vert", "in_texcoord")]
        )

        # ==========================================================
        # Nebula Volume Shader (quarter-res)
        # ==========================================================
        with open(BASE_DIR / "shaders" / "nebula_vol_vert.glsl", encoding="utf-8") as f:
            neb_vert = f.read()
        with open(BASE_DIR / "shaders" / "nebula_vol_frag.glsl", encoding="utf-8") as f:
            neb_frag = f.read()
        self.nebula_program = self.ctx.program(
            vertex_shader=neb_vert, fragment_shader=neb_frag,
        )
        self.nebula_vao = self.ctx.vertex_array(
            self.nebula_program, [(self.main_quad_buffer, "2f", "in_position")]
        )
        
        # Pass static nebula data to unified ISM & standalone nebula shaders
        neb_pos, neb_col = self.nebula_mgr.get_shader_uniforms()
        pos_data = np.zeros(150 * 4, dtype="f4")
        col_data = np.zeros(150 * 4, dtype="f4")
        pos_data[:len(neb_pos)] = neb_pos
        col_data[:len(neb_col)] = neb_col
        for neb_prog in (self.dust_program, self.nebula_program):
            if "u_numNebulae" in neb_prog:
                neb_prog["u_numNebulae"].value = self.nebula_mgr.num_nebulae
            if "u_nebulae_pos" in neb_prog:
                neb_prog["u_nebulae_pos"].write(pos_data.tobytes())
            if "u_nebulae_col" in neb_prog:
                neb_prog["u_nebulae_col"].write(col_data.tobytes())


        self.renderer = Renderer(
                    self.ctx,
                    self.program,
                    self.main_quad_vao,
                    self.medium_program,
                    self.medium_vao,
                    self.billboard_program,
                    self.billboard_vao
                )
        
        # ==========================================================
        # Rendering Settings
        # ==========================================================

        self.ctx.enable(self.ctx.BLEND)

        self.ctx.blend_func = (
            self.ctx.SRC_ALPHA,
            self.ctx.ONE_MINUS_SRC_ALPHA,
        )

    # ==============================================================
    # Framebuffer
    # ==============================================================

    def create_framebuffer(self, width, height):

        # Full-resolution HDR scene framebuffer (RGBA16F for high dynamic range star radiance)
        self.scene_tex = self.ctx.texture((width, height), 4, dtype='f2')
        self.scene_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        # Use depth TEXTURE (not renderbuffer) so it can be sampled in shaders
        self.depth_tex = self.ctx.depth_texture((width, height))
        # Disable comparison mode so we read raw depth values, not shadow-map 0/1
        self.depth_tex.compare_func = ''

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_tex],
            depth_attachment=self.depth_tex
        )
        self.depth_fbo = self.ctx.framebuffer(
            depth_attachment=self.depth_tex
        )

        # Dedicated HDR accumulation buffer for 3D stars prior to soft resolve
        self.star_tex = self.ctx.texture((width, height), 4, dtype='f2')
        self.star_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.star_fbo = self.ctx.framebuffer(color_attachments=[self.star_tex])

        # Scaled 3/8-resolution unified 3D ISM MRT FBO (480x270 at 1280x720: 2.25x sharper than 1/4-res, fast on iGPU)
        iw, ih = max((width * 3) // 8, 1), max((height * 3) // 8, 1)
        qw, qh = max(width // 4, 1), max(height // 4, 1)
        self.dust_tex = self.ctx.texture((iw, ih), 4, dtype='f2')  # RGBA16F: RGB=CCM T, A=front depth
        self.dust_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.dust_fbo = self.ctx.framebuffer(color_attachments=[self.dust_tex])

        self.ism_bins_tex = self.ctx.texture((iw, ih), 4, dtype='f2')  # RGBA16F: 4-bin cumulative optical depth
        self.ism_bins_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.ism_bins_fbo = self.ctx.framebuffer(color_attachments=[self.ism_bins_tex])

        self.nebula_col_tex = self.ctx.texture((iw, ih), 4, dtype='f2')  # RGBA16F: RGB=Emission, A=Transmittance
        self.nebula_col_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.nebula_depth_tex = self.ctx.texture((iw, ih), 1, dtype='f2')  # R16F: Front depth
        self.nebula_depth_tex.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.nebula_fbo = self.ctx.framebuffer(color_attachments=[self.nebula_col_tex, self.nebula_depth_tex])
        self.nebula_col_fbo = self.ctx.framebuffer(color_attachments=[self.nebula_col_tex])
        self.nebula_depth_fbo = self.ctx.framebuffer(color_attachments=[self.nebula_depth_tex])
        self.ism_fbo = self.ctx.framebuffer(
            color_attachments=[self.dust_tex, self.ism_bins_tex, self.nebula_col_tex, self.nebula_depth_tex]
        )

        # Intermediate composited scene FBO (full res RGBA16F)
        self.comp_tex = self.ctx.texture((width, height), 4, dtype='f2')
        self.comp_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.comp_fbo = self.ctx.framebuffer(color_attachments=[self.comp_tex])

        # Multi-scale Bloom Framebuffers (RGBA16F)
        # Half-resolution for tight optical glow around bright stars and cores
        hw, hh = max(width // 2, 1), max(height // 2, 1)
        self.bloom_half_tex = self.ctx.texture((hw, hh), 4, dtype='f2')
        self.bloom_half_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.bloom_half_temp = self.ctx.texture((hw, hh), 4, dtype='f2')
        self.bloom_half_temp.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.bloom_extract_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_half_tex])
        self.bloom_half_temp_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_half_temp])
        self.bloom_half_blur_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_half_tex])

        # Quarter-resolution for broad diffuse galactic halo
        self.bloom_quarter_tex = self.ctx.texture((qw, qh), 4, dtype='f2')
        self.bloom_quarter_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.bloom_quarter_temp = self.ctx.texture((qw, qh), 4, dtype='f2')
        self.bloom_quarter_temp.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.bloom_down_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_quarter_tex])
        self.bloom_quarter_temp_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_quarter_temp])
        self.bloom_quarter_blur_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_quarter_tex])

    # ==============================================================
    # Window Resize
    # ==============================================================

    def on_resize(self, width: int, height: int):

        self.scene_tex.release()
        self.depth_fbo.release()
        self.depth_tex.release()
        self.fbo.release()
        self.star_tex.release()
        self.star_fbo.release()
        self.dust_tex.release()
        self.dust_fbo.release()
        self.ism_bins_tex.release()
        self.ism_bins_fbo.release()
        self.nebula_col_tex.release()
        self.nebula_depth_tex.release()
        self.nebula_col_fbo.release()
        self.nebula_depth_fbo.release()
        self.nebula_fbo.release()
        self.ism_fbo.release()
        self.comp_tex.release()
        self.comp_fbo.release()

        self.bloom_half_tex.release()
        self.bloom_half_temp.release()
        self.bloom_extract_fbo.release()
        self.bloom_half_temp_fbo.release()
        self.bloom_half_blur_fbo.release()

        self.bloom_quarter_tex.release()
        self.bloom_quarter_temp.release()
        self.bloom_down_fbo.release()
        self.bloom_quarter_temp_fbo.release()
        self.bloom_quarter_blur_fbo.release()

        self.create_framebuffer(width, height)

        self.blackhole.aspect_ratio = width / height

    # ==============================================================
    # Mouse Controls
    # ==============================================================

    def on_mouse_press_event(self, x: int, y: int, button: int):
        self.camera.on_mouse_press(x, y, button)

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.camera.on_mouse_release(x, y, button)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        self.camera.on_mouse_drag(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.camera.on_mouse_scroll(0, 0, x_offset, y_offset)

    # ==============================================================
    # Keyboard Controls
    # ==============================================================

    def on_key_event(self, key, action, modifiers):

        if action == self.wnd.keys.ACTION_PRESS:

            if key in (self.wnd.keys.W, self.wnd.keys.UP):
                self.move_up = True

            elif key in (self.wnd.keys.S, self.wnd.keys.DOWN):
                self.move_down = True

            elif key in (self.wnd.keys.A, self.wnd.keys.LEFT):
                self.move_left = True

            elif key in (self.wnd.keys.D, self.wnd.keys.RIGHT):
                self.move_right = True

            elif key == self.wnd.keys.Q:
                self._zoom_in = True

            elif key == self.wnd.keys.E:
                self._zoom_out = True

            elif key == self.wnd.keys.R:
                self.camera.reset()

            elif key == self.wnd.keys.T:
                self.debug_dust = not self.debug_dust

            elif key == self.wnd.keys.N:
                self.debug_nebula = not self.debug_nebula

            elif key == self.wnd.keys.B:
                self.debug_post_mode = (self.debug_post_mode + 1) % 5
                mode_names = [
                    "Normal (Cinematic Astrophotography)",
                    "Tight Bloom Buffer",
                    "Broad Bloom Buffer",
                    "Attenuated Combined Bloom",
                    "Raw Scene (No Bloom)"
                ]
                print(f"\n[Post-Processing] Mode: {mode_names[self.debug_post_mode]}")

            elif key == self.wnd.keys.M:
                self.galactic_map.enabled = not self.galactic_map.enabled
                status = "ON" if self.galactic_map.enabled else "OFF"
                print(f"\n[Map] Scientific Galactic Map Overlay: {status}")

            elif key == self.wnd.keys.O:
                self.s_stars.show_cluster = not self.s_stars.show_cluster
                status = "ON" if self.s_stars.show_cluster else "OFF"
                print(f"\n[Sgr A*] S-Star Cluster & Orbits: {status}")

            elif key == self.wnd.keys.Y:
                self.s_stars.show_labels = not self.s_stars.show_labels
                status = "ON" if self.s_stars.show_labels else "OFF"
                print(f"\n[Sgr A*] S-Star Identification Overlay: {status}")

            elif key == self.wnd.keys.SPACE:
                self.sim_paused = not self.sim_paused
                status = "PAUSED" if self.sim_paused else "RUNNING"
                print(f"\n[Sim] Differential Galactic Rotation: {status} (Speed: {self.sim_speed:.2f}x)")

            elif key == self.wnd.keys.LEFT_BRACKET:
                self.sim_speed = max(0.125, self.sim_speed / 2.0)
                print(f"\n[Sim] Simulation Speed Multiplier: {self.sim_speed:.2f}x")

            elif key == self.wnd.keys.RIGHT_BRACKET:
                self.sim_speed = min(16.0, self.sim_speed * 2.0)
                print(f"\n[Sim] Simulation Speed Multiplier: {self.sim_speed:.2f}x")

            elif key in (self.wnd.keys.BACKSPACE, self.wnd.keys.NUMBER_0):
                self.sim_time = 0.0
                self.s_star_years = 0.0
                self.galaxy_rotation_angle = 0.0
                print(f"\n[Sim] Simulation Clock Reset to t=0.0s")

        elif action == self.wnd.keys.ACTION_RELEASE:

            if key in (self.wnd.keys.W, self.wnd.keys.UP):
                self.move_up = False

            elif key in (self.wnd.keys.S, self.wnd.keys.DOWN):
                self.move_down = False

            elif key in (self.wnd.keys.A, self.wnd.keys.LEFT):
                self.move_left = False

            elif key in (self.wnd.keys.D, self.wnd.keys.RIGHT):
                self.move_right = False

            elif key == self.wnd.keys.Q:
                self._zoom_in = False

            elif key == self.wnd.keys.E:
                self._zoom_out = False
   # ==============================================================
    # Render Loop
    # ==============================================================

    def on_render(self, time, frametime):

        bh = self.blackhole

        if self.move_up:
            self.camera.look_up()

        if self.move_down:
            self.camera.look_down()

        if self.move_left:
            self.camera.rotate_left(frametime)

        if self.move_right:
            self.camera.rotate_right(frametime)

        if self._zoom_in:
            self.camera.zoom_in(frametime)

        if self._zoom_out:
            self.camera.zoom_out(frametime)

        self.camera.update(frametime)

        # ----------------------------------------------------------
        # Camera data for black-hole shaders (computed once per frame)
        # ----------------------------------------------------------
        cam_pos   = self.camera.get_position()
        cam_fwd   = self.camera.get_forward()
        cam_right = self.camera.get_right()
        cam_up    = self.camera.get_up()

        # Convert camera position to black-hole local space
        bh_cam_pos = (
            cam_pos[0] * BLACK_HOLE_LOCAL_SCALE,
            cam_pos[1] * BLACK_HOLE_LOCAL_SCALE,
            cam_pos[2] * BLACK_HOLE_LOCAL_SCALE,
        )


        # ----------------------------------------------------------
        # LOD
        # ----------------------------------------------------------

        dist = self.camera.absolute_distance
        lod_state = self.lod_manager.get_state(
            dist
        )


        fps = 1.0 / max(frametime, 0.0001)

        # Single elapsed simulation clock with pause/resume and speed control
        if not self.sim_paused:
            self.sim_time += frametime * self.sim_speed
            self.s_star_years += frametime * self.s_star_time_scale * self.sim_speed

        self.s_stars.update(self.s_star_years, cam_pos)

        # Reverse apparent galaxy rotation from anticlockwise to clockwise:
        # signed_pattern_angle = -pattern_speed * sim_time
        signed_pattern_angle = -self.pattern_speed * self.sim_time
        self.galaxy_rotation_angle = math.fmod(signed_pattern_angle, 2.0 * math.pi)
        if self.galaxy_rotation_angle > 0.0:
            self.galaxy_rotation_angle -= 2.0 * math.pi

        sim_status = "PAUSED" if self.sim_paused else f"{self.sim_speed:.2f}x"
        print(
            f"Stars: {self.bulge.num_stars + self.bar.num_stars + self.galactic_disk.num_stars} "
            f"Sim: {self.sim_time:.1f}s [{sim_status}] "
            f"Bulge vis: {lod_state['bulge_visibility']:.2f} "
            f"Pattern Rot: {math.degrees(self.galaxy_rotation_angle):.1f}° "
            f"Dist: {self.camera.absolute_distance:.1f} "
            f"FPS: {fps:.1f}      ",
            end="\r",
            flush=True
        )
        
        # ----------------------------------------------------------
        # PASS 1
        # Render Scene
        # ----------------------------------------------------------

        # Calculate black hole screen position (0,0,0) in NDC and compute view-projection matrix
        vp = self.camera.get_projection_matrix(bh.aspect_ratio) * self.camera.get_view_matrix()
        vp_matrix = vp
        bh_clip = vp * glm.vec4(0.0, 0.0, 0.0, 1.0)
        if bh_clip.w > 0.0:
            bh_screen_pos = (bh_clip.x / bh_clip.w, bh_clip.y / bh_clip.w)
        else:
            bh_screen_pos = (100.0, 100.0)

        # NEAR Shader Uniforms
        if "u_time" in self.program:
            self.program["u_time"].value = time
        if "aspectRatio" in self.program:
            self.program["aspectRatio"].value = bh.aspect_ratio
        if "u_eventHorizon" in self.program:
            self.program["u_eventHorizon"].value = bh.event_horizon_radius
        if "u_photonSphere" in self.program:
            self.program["u_photonSphere"].value = bh.photon_sphere_radius
        if "u_innerDisk" in self.program:
            self.program["u_innerDisk"].value = bh.inner_disk_radius
        if "u_outerDisk" in self.program:
            self.program["u_outerDisk"].value = bh.outer_disk_radius
        if "u_camDistance" in self.program:
            self.program["u_camDistance"].value = self.camera.absolute_distance

        # Real camera basis (replaces synthetic u_azimuth/u_diskSquish camera)
        if "u_cameraPosition" in self.program:
            self.program["u_cameraPosition"].value = bh_cam_pos
        if "u_cameraForward" in self.program:
            self.program["u_cameraForward"].value = cam_fwd
        if "u_cameraRight" in self.program:
            self.program["u_cameraRight"].value = cam_right
        if "u_cameraUp" in self.program:
            self.program["u_cameraUp"].value = cam_up

        # Continuous LOD Parameters for NEAR
        if "u_lodDetail" in self.program:
            self.program["u_lodDetail"].value = lod_state["ray_march_detail"]
        if "u_volumetricDetail" in self.program:
            self.program["u_volumetricDetail"].value = lod_state["volumetric_detail"]
        if "u_diskThickness" in self.program:
            self.program["u_diskThickness"].value = lod_state["disk_thickness"]
        if "u_proceduralStarWeight" in self.program:
            self.program["u_proceduralStarWeight"].value = lod_state["near_weight"]
        if "u_bhScreenPos" in self.program:
            self.program["u_bhScreenPos"].value = bh_screen_pos

        # Medium LOD Uniforms
        if "u_camDistance" in self.medium_program:
            self.medium_program["u_camDistance"].value = self.camera.absolute_distance
        if "aspectRatio" in self.medium_program:
            self.medium_program["aspectRatio"].value = bh.aspect_ratio
        if "u_diskSquish" in self.medium_program:
            self.medium_program["u_diskSquish"].value = self.camera.disk_squish
        if "u_time" in self.medium_program:
            self.medium_program["u_time"].value = time
        if "u_diskThickness" in self.medium_program:
            self.medium_program["u_diskThickness"].value = lod_state["disk_thickness"]
        if "u_glowIntensity" in self.medium_program:
            self.medium_program["u_glowIntensity"].value = lod_state["glow_intensity"]
        if "u_innerDisk" in self.medium_program:
            self.medium_program["u_innerDisk"].value = bh.inner_disk_radius
        if "u_outerDisk" in self.medium_program:
            self.medium_program["u_outerDisk"].value = bh.outer_disk_radius
        if "u_innerRingStrength" in self.medium_program:
            self.medium_program["u_innerRingStrength"].value = lod_state["inner_ring_strength"]
        if "u_dopplerStrength" in self.medium_program:
            self.medium_program["u_dopplerStrength"].value = lod_state["doppler_strength"]
        if "u_diskOuterRadius" in self.medium_program:
            self.medium_program["u_diskOuterRadius"].value = lod_state["disk_outer_radius"]
        if "u_volumetricDetail" in self.medium_program:
            self.medium_program["u_volumetricDetail"].value = lod_state["volumetric_detail"]
        if "u_proceduralStarWeight" in self.medium_program:
            self.medium_program["u_proceduralStarWeight"].value = lod_state["near_weight"]
        if "u_bhScreenPos" in self.medium_program:
            self.medium_program["u_bhScreenPos"].value = bh_screen_pos

        # Real camera basis for medium shader
        if "u_cameraPosition" in self.medium_program:
            self.medium_program["u_cameraPosition"].value = bh_cam_pos
        if "u_cameraForward" in self.medium_program:
            self.medium_program["u_cameraForward"].value = cam_fwd
        if "u_cameraRight" in self.medium_program:
            self.medium_program["u_cameraRight"].value = cam_right
        if "u_cameraUp" in self.medium_program:
            self.medium_program["u_cameraUp"].value = cam_up

        # Billboard LOD Uniforms
        if "u_camDistance" in self.billboard_program:
            self.billboard_program["u_camDistance"].value = self.camera.absolute_distance
        if "u_time" in self.billboard_program:
            self.billboard_program["u_time"].value = time
        if "aspectRatio" in self.billboard_program:
            self.billboard_program["aspectRatio"].value = bh.aspect_ratio
        if "u_diskSquish" in self.billboard_program:
            self.billboard_program["u_diskSquish"].value = self.camera.disk_squish
        if "u_diskThickness" in self.billboard_program:
            self.billboard_program["u_diskThickness"].value = lod_state["disk_thickness"]
        if "u_glowIntensity" in self.billboard_program:
            self.billboard_program["u_glowIntensity"].value = lod_state["glow_intensity"]
        if "u_innerDisk" in self.billboard_program:
            self.billboard_program["u_innerDisk"].value = bh.inner_disk_radius
        if "u_outerDisk" in self.billboard_program:
            self.billboard_program["u_outerDisk"].value = bh.outer_disk_radius
        if "u_innerRingStrength" in self.billboard_program:
            self.billboard_program["u_innerRingStrength"].value = lod_state["inner_ring_strength"]
        if "u_dopplerStrength" in self.billboard_program:
            self.billboard_program["u_dopplerStrength"].value = lod_state["doppler_strength"]
        if "u_diskOuterRadius" in self.billboard_program:
            self.billboard_program["u_diskOuterRadius"].value = lod_state["disk_outer_radius"]
        if "u_volumetricDetail" in self.billboard_program:
            self.billboard_program["u_volumetricDetail"].value = lod_state["volumetric_detail"]
        if "u_proceduralStarWeight" in self.billboard_program:
            self.billboard_program["u_proceduralStarWeight"].value = lod_state["near_weight"]
        if "u_bhScreenPos" in self.billboard_program:
            self.billboard_program["u_bhScreenPos"].value = bh_screen_pos

        # Real camera basis for billboard shader
        if "u_cameraPosition" in self.billboard_program:
            self.billboard_program["u_cameraPosition"].value = bh_cam_pos
        if "u_cameraForward" in self.billboard_program:
            self.billboard_program["u_cameraForward"].value = cam_fwd
        if "u_cameraRight" in self.billboard_program:
            self.billboard_program["u_cameraRight"].value = cam_right
        if "u_cameraUp" in self.billboard_program:
            self.billboard_program["u_cameraUp"].value = cam_up

        # ----------------------------------------------------------
        # Component and LOD visibility
        # ----------------------------------------------------------
        star_visibility = lod_state["star_visibility"]
        bulge_visibility = lod_state["bulge_visibility"]
        galaxy_far_weight = lod_state["galaxy_far_weight"]

        # ----------------------------------------------------------
        # PASS 1A: Scene Depth Pass (for Dust and Nebulae Occlusion)
        # Separates depth information from additive stellar accumulation
        # so overlapping stars never depend on draw order.
        # ----------------------------------------------------------
        self.depth_fbo.use()
        self.ctx.clear(depth=1.0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.depth_mask = True
        self.ctx.disable(self.ctx.BLEND)

        if star_visibility > 0.001:
            self.galactic_disk.render(
                self.camera, 
                bh.aspect_ratio, 
                star_visibility, 
                self.galaxy_rotation_angle,
                sim_time=self.sim_time,
                depth_pass=True
            )
            self.bar.render(
                self.camera, 
                bh.aspect_ratio, 
                star_visibility, 
                self.galaxy_rotation_angle,
                sim_time=self.sim_time,
                depth_pass=True
            )
            self.bulge.render(
                self.camera, 
                bh.aspect_ratio, 
                bulge_visibility, 
                self.galaxy_rotation_angle,
                sim_time=self.sim_time,
                depth_pass=True
            )

        if galaxy_far_weight > 0.001:
            self.far_galaxy.render_depth(
                self.camera,
                bh.aspect_ratio,
                self.galaxy_rotation_angle
            )

        # ----------------------------------------------------------
        # PASS 1B: Stellar Accumulation + Soft Resolve, then Far Galaxy LOD
        # Additive blending with depth testing disabled: NO star is ever
        # discarded, and overlapping stars never depend on draw order.
        # ----------------------------------------------------------
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0) # Color clear; depth preserved from Pass 1A
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.ctx.depth_mask = False

        if star_visibility > 0.001:
            # Accumulate 3D stars into dedicated HDR buffer and apply gentle soft-shoulder
            # resolve before black-hole & far-galaxy compositing to prevent nuclear burnout.
            self.star_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.ctx.enable(self.ctx.BLEND)
            self.ctx.blend_func = (self.ctx.SRC_ALPHA, self.ctx.ONE)

            self.galactic_disk.render(
                self.camera, 
                bh.aspect_ratio, 
                star_visibility, 
                self.galaxy_rotation_angle,
                sim_time=self.sim_time,
                depth_pass=False
            )
            self.bar.render(
                self.camera, 
                bh.aspect_ratio, 
                star_visibility, 
                self.galaxy_rotation_angle,
                sim_time=self.sim_time,
                depth_pass=False
            )
            self.bulge.render(
                self.camera, 
                bh.aspect_ratio, 
                bulge_visibility, 
                self.galaxy_rotation_angle,
                sim_time=self.sim_time,
                depth_pass=False
            )

            self.fbo.use()
            self.ctx.disable(self.ctx.BLEND)
            self.star_tex.use(location=0)
            self.galaxy_resolve_program["u_galaxyTex"].value = 0
            self.galaxy_resolve_program["u_knee"].value = 0.95
            self.galaxy_resolve_program["u_cap"].value = 1.30
            self.galaxy_resolve_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # Far Galaxy LOD representation on the galactic plane
        if galaxy_far_weight > 0.001:
            self.fbo.use()
            self.ctx.enable(self.ctx.BLEND)
            self.far_galaxy.render(
                self.camera,
                bh.aspect_ratio,
                galaxy_far_weight,
                self.galaxy_rotation_angle
            )

        # ----------------------------------------------------------
        # Post-process passes (Black hole raymarcher + depth-split S-stars)
        # ----------------------------------------------------------
        # Far-side S-stars & orbits behind Sgr A* (occluded/attenuated by BH shadow & disk)
        self.fbo.use()
        self.s_stars.render_side(
            self.camera,
            bh.aspect_ratio,
            self.wnd.size,
            side_filter=-1,
        )

        # Composite mode: StarColor * Transmittance + GasColor * 1.0
        self.fbo.use()
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = (self.ctx.ONE, self.ctx.SRC_ALPHA)
        
        self.renderer.render(lod_state)

        # Near-side S-stars & orbits transiting in front of Sgr A*
        self.s_stars.render_side(
            self.camera,
            bh.aspect_ratio,
            self.wnd.size,
            side_filter=1,
        )
        
        # Restore depth mask after BH rendering
        self.ctx.depth_mask = True

        # ----------------------------------------------------------
        # PASS 1.5: Unified 3D Interstellar Medium (Dust + Molecular Clouds + Coupled Nebulae)
        # ----------------------------------------------------------
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

        nebula_weight = 0.0
        if dist > 8.0:
            if dist < 12.0:
                t = max(0.0, min(1.0, (dist - 8.0) / 4.0))
                nebula_weight = t * t * (3.0 - 2.0 * t) * 0.3
            elif dist < 30.0:
                t = max(0.0, min(1.0, (dist - 12.0) / 18.0))
                nebula_weight = 0.3 + t * t * (3.0 - 2.0 * t) * 0.7
            else:
                nebula_weight = 1.0
        nebula_weight *= (1.0 - galaxy_far_weight)

        self.ctx.disable(self.ctx.BLEND)
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.dust_fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ism_bins_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.nebula_col_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.nebula_depth_fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        vp = self.camera.get_projection_matrix(bh.aspect_ratio) * self.camera.get_view_matrix()
        inv_model = glm.rotate(glm.mat4(1.0), -self.galaxy_rotation_angle, glm.vec3(0.0, 1.0, 0.0))

        if dust_weight > 0.001 or nebula_weight > 0.001:
            self.ism_fbo.use()
            self.ism.bind_textures(self.dust_program, base_unit=0)
            if "u_camPos" in self.dust_program:
                self.dust_program["u_camPos"].value = self.camera.get_position()
            if "u_invVP" in self.dust_program:
                self.dust_program["u_invVP"].write(glm.inverse(vp))
            if "u_dustWeight" in self.dust_program:
                self.dust_program["u_dustWeight"].value = dust_weight
            if "u_nebulaWeight" in self.dust_program:
                self.dust_program["u_nebulaWeight"].value = nebula_weight
            if "u_unifiedISM" in self.dust_program:
                self.dust_program["u_unifiedISM"].value = 1
            if "u_galaxyModel" in self.dust_program:
                self.dust_program["u_galaxyModel"].write(inv_model)

            # Dust tunable params
            if "u_dustOpacity" in self.dust_program: self.dust_program["u_dustOpacity"].value = 1.2
            if "u_dustWidth" in self.dust_program: self.dust_program["u_dustWidth"].value = 0.25
            if "u_dustArmOffset" in self.dust_program: self.dust_program["u_dustArmOffset"].value = -0.12
            if "u_dustNoiseScale" in self.dust_program: self.dust_program["u_dustNoiseScale"].value = 0.8
            if "u_dustDensity" in self.dust_program: self.dust_program["u_dustDensity"].value = 2.5
            if "u_dustHeightScale" in self.dust_program: self.dust_program["u_dustHeightScale"].value = 1.0
            if "u_dustLaneWidth" in self.dust_program: self.dust_program["u_dustLaneWidth"].value = 0.06
            if "u_dustDiffuseWidth" in self.dust_program: self.dust_program["u_dustDiffuseWidth"].value = 0.18
            if "u_camDist" in self.dust_program: self.dust_program["u_camDist"].value = dist

            self.dust_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # ----------------------------------------------------------
        # PASS 1.75: Composite dust & nebulae onto scene
        # ----------------------------------------------------------
        self.comp_fbo.use()
        ctx_cam_pos = self.camera.get_position()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(self.ctx.BLEND)

        self.scene_tex.use(location=0)
        self.dust_tex.use(location=1)
        self.nebula_col_tex.use(location=2)
        self.nebula_depth_tex.use(location=3)
        self.scene_depth_tex = self.depth_tex
        self.scene_depth_tex.use(location=4)
        self.ism_bins_tex.use(location=5)

        self.comp_program["u_scene"].value = 0
        self.comp_program["u_dust"].value = 1
        self.comp_program["u_nebulaColor"].value = 2
        self.comp_program["u_nebulaDepth"].value = 3
        self.comp_program["u_sceneDepth"].value = 4
        if "u_ismBins" in self.comp_program:
            self.comp_program["u_ismBins"].value = 5
        if "u_useUnifiedISM" in self.comp_program:
            self.comp_program["u_useUnifiedISM"].value = 1
        if "u_galaxyModel" in self.comp_program:
            self.comp_program["u_galaxyModel"].write(inv_model)
        if "u_debugDust" in self.comp_program:
            self.comp_program["u_debugDust"].value = self.debug_dust
        if "u_debugNebula" in self.comp_program:
            self.comp_program["u_debugNebula"].value = self.debug_nebula
        if "u_nearPlane" in self.comp_program:
            self.comp_program["u_nearPlane"].value = 0.01
        if "u_farPlane" in self.comp_program:
            self.comp_program["u_farPlane"].value = 10000.0
        if "u_camPos" in self.comp_program:
            self.comp_program["u_camPos"].value = ctx_cam_pos
        if "u_camForward" in self.comp_program:
            self.comp_program["u_camForward"].value = self.camera.get_forward()
        if "u_invVP" in self.comp_program:
            self.comp_program["u_invVP"].write(glm.inverse(vp))
        self.comp_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # ----------------------------------------------------------
        # PASS 2
        # Astrophotography Post Processing & Multi-Scale Bloom
        # ----------------------------------------------------------
        w, h = self.wnd.size
        hw, hh = max(w // 2, 1), max(h // 2, 1)
        qw, qh = max(w // 4, 1), max(h // 4, 1)

        # 2.1: Extract bright highlights into bloom_half_tex
        self.bloom_extract_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.ctx.disable(self.ctx.BLEND)
        self.comp_tex.use(location=0)
        self.extract_program["u_image"].value = 0
        self.extract_program["u_threshold"].value = 0.85
        self.extract_program["u_knee"].value = 0.25
        self.extract_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # 2.2: Tight Bloom Blur (half res)
        # Horizontal
        self.bloom_half_temp_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_half_tex.use(location=0)
        self.blur_program["u_image"].value = 0
        self.blur_program["u_direction"].value = (1.0 / hw, 0.0)
        self.blur_program["u_radius"].value = 2.0
        self.blur_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # Vertical
        self.bloom_half_blur_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_half_temp.use(location=0)
        self.blur_program["u_image"].value = 0
        self.blur_program["u_direction"].value = (0.0, 1.0 / hh)
        self.blur_program["u_radius"].value = 2.0
        self.blur_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # 2.3: Broad Bloom Blur (quarter res)
        # Downsample tight bloom into quarter res
        self.bloom_down_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_half_tex.use(location=0)
        self.extract_program["u_image"].value = 0
        self.extract_program["u_threshold"].value = 0.0
        self.extract_program["u_knee"].value = 0.01
        self.extract_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # Horizontal
        self.bloom_quarter_temp_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_quarter_tex.use(location=0)
        self.blur_program["u_image"].value = 0
        self.blur_program["u_direction"].value = (1.0 / qw, 0.0)
        self.blur_program["u_radius"].value = 3.5
        self.blur_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # Vertical
        self.bloom_quarter_blur_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.bloom_quarter_temp.use(location=0)
        self.blur_program["u_image"].value = 0
        self.blur_program["u_direction"].value = (0.0, 1.0 / qh)
        self.blur_program["u_radius"].value = 3.5
        self.blur_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # 2.4: Final Screen Pass: HDR Composite + Multi-scale Bloom + Filmic ACES Tone Mapping
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.ctx.disable(self.ctx.BLEND)

        self.comp_tex.use(location=0)
        self.bloom_half_tex.use(location=1)
        self.bloom_quarter_tex.use(location=2)

        self.post_program["u_scene"].value = 0
        self.post_program["u_bloomTight"].value = 1
        self.post_program["u_bloomBroad"].value = 2
        self.post_program["u_exposure"].value = 1.05
        self.post_program["u_bloomTightIntensity"].value = 0.35
        self.post_program["u_bloomBroadIntensity"].value = 0.18
        self.post_program["u_dustProtection"].value = 0.88
        if "u_debugPost" in self.post_program:
            self.post_program["u_debugPost"].value = self.debug_post_mode

        self.quad_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # ----------------------------------------------------------
        # PASS 3: Scientific Face-on Galactic Map Overlay (Key M)
        # ----------------------------------------------------------
        if self.galactic_map.enabled:
            self.galactic_map.render(
                self.camera,
                bh.aspect_ratio,
                self.galaxy_rotation_angle
            )

        # ----------------------------------------------------------
        # PASS 4: Sagittarius A* S-Star Identification Overlay (Key Y)
        # ----------------------------------------------------------
        self.s_stars.render_overlay(
            self.camera,
            bh.aspect_ratio,
            self.wnd.size,
        )


if __name__ == "__main__":
    mglw.run_window_config(MilkyWaySimulation)