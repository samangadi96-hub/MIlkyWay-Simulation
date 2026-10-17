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
        
        self.galaxy_rotation_angle = 0.0



        # Keyboard state flags
        self.move_up    = False
        self.move_down  = False
        self.move_left  = False
        self.move_right = False
        self._zoom_in   = False
        self._zoom_out  = False
        self._zoom_out  = False
        self.debug_dust = False
        self.debug_nebula = False

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
        print('  T                -> toggle dust debug')
        print('  N                -> toggle nebula debug')
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
        
        # Pass static nebula data to shader
        neb_pos, neb_col = self.nebula_mgr.get_shader_uniforms()
        if "u_numNebulae" in self.nebula_program:
            self.nebula_program["u_numNebulae"].value = self.nebula_mgr.num_nebulae
        for i in range(150):
            if f"u_nebulae_pos[{i}]" in self.nebula_program:
                self.nebula_program[f"u_nebulae_pos[{i}]"].value = tuple(neb_pos[i*4:i*4+4])
            if f"u_nebulae_col[{i}]" in self.nebula_program:
                self.nebula_program[f"u_nebulae_col[{i}]"].value = tuple(neb_col[i*4:i*4+4])


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

        self.scene_tex = self.ctx.texture((width, height), 4)
        # Use depth TEXTURE (not renderbuffer) so it can be sampled in shaders
        self.depth_tex = self.ctx.depth_texture((width, height))
        # Disable comparison mode so we read raw depth values, not shadow-map 0/1
        self.depth_tex.compare_func = ''

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_tex],
            depth_attachment=self.depth_tex
        )

        # Quarter-resolution FBO for dust pass
        # 2 components (RG16F): R = transmittance, G = front dust depth
        qw, qh = max(width // 4, 1), max(height // 4, 1)
        self.dust_tex = self.ctx.texture((qw, qh), 2, dtype='f2')
        self.dust_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.dust_fbo = self.ctx.framebuffer(color_attachments=[self.dust_tex])

        # Quarter-resolution FBO for nebula pass
        self.nebula_col_tex = self.ctx.texture((qw, qh), 4, dtype='f2') # RGBA16F: RGB=Emission, A=Transmittance
        self.nebula_col_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.nebula_depth_tex = self.ctx.texture((qw, qh), 1, dtype='f2') # R16F: Front depth
        self.nebula_depth_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.nebula_fbo = self.ctx.framebuffer(color_attachments=[self.nebula_col_tex, self.nebula_depth_tex])

        # Intermediate composited scene FBO (full res)
        self.comp_tex = self.ctx.texture((width, height), 4)
        self.comp_fbo = self.ctx.framebuffer(color_attachments=[self.comp_tex])

    # ==============================================================
    # Window Resize
    # ==============================================================

    def on_resize(self, width: int, height: int):

        self.scene_tex.release()
        self.fbo.release()
        self.dust_tex.release()
        self.dust_fbo.release()
        self.nebula_col_tex.release()
        self.nebula_depth_tex.release()
        self.nebula_fbo.release()
        self.comp_tex.release()
        self.comp_fbo.release()

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

        lod_state = self.lod_manager.get_state(
            self.camera.absolute_distance
        )


        fps = 1.0 / max(frametime, 0.0001)
        
        self.galaxy_rotation_angle += frametime * 0.05
        
        print(
            f"Stars: {self.bulge.num_stars + self.bar.num_stars + self.galactic_disk.num_stars} "
            f"Bulge vis: {lod_state['bulge_visibility']:.2f} "
            f"Galaxy Rot: {self.galaxy_rotation_angle:.2f} "
            f"Dist: {self.camera.absolute_distance:.1f} "
            f"FPS: {fps:.1f}      ",
            end="\r",
            flush=True
        )
        
        # ----------------------------------------------------------
        # PASS 1
        # Render Scene
        # ----------------------------------------------------------

        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        
        # Enable depth test AND depth writes for stars.
        # This writes the nearest star's depth per pixel, which the dust
        # composite shader uses for foreground/background occlusion.
        # Additive blending still works correctly for color output.
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.depth_mask = True
        
        # Use standard additive blending for 3D stars
        self.ctx.blend_func = (self.ctx.SRC_ALPHA, self.ctx.ONE)

        # Calculate black hole screen position (0,0,0) in NDC
        vp_matrix = self.camera.get_projection_matrix(bh.aspect_ratio) * self.camera.get_view_matrix()
        bh_clip = vp_matrix * glm.vec4(0.0, 0.0, 0.0, 1.0)
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

        # Render Galactic Disk first, then Bulge
        # Fade out 3D stars when very close to avoid stacking over procedural stars
        star_visibility = 1.0 
        base_visibility = 1.0 - lod_state["near_weight"]
        star_visibility = base_visibility
        
        # LOCAL FIX: Restore 3D moving stars in the near region
        # Ensures they don't abruptly appear or disappear by blending from 2.0 to 11.0
        dist = self.camera.absolute_distance
        if dist < 11.0:
            def smoothstep_py(e0, e1, x):
                t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
                return t * t * (3.0 - 2.0 * t)
                
            # Get the exact base visibility that will be active at distance 11.0
            # to ensure a mathematically perfect, seamless merge.
            base_at_11 = 1.0 - self.lod_manager.get_state(11.0)["near_weight"]
            
            if dist < 4.0:
                # Ramp up: 0.0 at dist 2.0 -> ~0.125 at dist 3.0 -> 0.25 at dist 4.0
                t = smoothstep_py(2.0, 4.0, dist)
                local_vis = t * 0.25
            elif dist < 8.0:
                # Moderate to Strong: 0.25 at dist 4.0 -> ~0.40 at dist 6.0 -> 0.55 at dist 8.0
                t = smoothstep_py(4.0, 8.0, dist)
                local_vis = 0.25 * (1.0 - t) + 0.55 * t
            else:
                # Merge up to normal transition: 0.55 at dist 8.0 -> base_at_11 at dist 11.0
                t = smoothstep_py(8.0, 11.0, dist)
                local_vis = 0.55 * (1.0 - t) + base_at_11 * t
                
            star_visibility = local_vis
        
        self.galactic_disk.render(
            self.camera, 
            bh.aspect_ratio, 
            star_visibility, 
            self.galaxy_rotation_angle
        )
        self.bar.render(
            self.camera, 
            bh.aspect_ratio, 
            star_visibility, 
            self.galaxy_rotation_angle
        )
        self.bulge.render(
            self.camera, 
            bh.aspect_ratio, 
            star_visibility, 
            self.galaxy_rotation_angle
        )

        # ----------------------------------------------------------
        # Post-process passes (Black hole raymarcher)
        # ----------------------------------------------------------
        # Disable depth test for BH — it renders as fullscreen quads.
        # depth_mask=False prevents BH quads from overwriting star depths
        # that the dust compositor needs.
        self.ctx.disable(self.ctx.DEPTH_TEST)
        self.ctx.depth_mask = False
        
        # Composite mode: StarColor * Transmittance + GasColor * 1.0
        self.ctx.blend_func = (self.ctx.ONE, self.ctx.SRC_ALPHA)
        
        self.renderer.render(lod_state)
        
        # Restore depth mask after BH rendering
        self.ctx.depth_mask = True

        # ----------------------------------------------------------
        # PASS 1.5: Galaxy Dust (quarter-res)
        # ----------------------------------------------------------
        dust_weight = 0.0
        if dist > 8.0:
            # Smoother, wider transition curve
            # ~8-12: gradual introduction
            # ~12-30: clearly visible
            # ~30-50: strongest
            # 50+: sustained
            if dist < 15.0:
                t = max(0.0, min(1.0, (dist - 8.0) / 7.0))
                dust_weight = t * t * (3.0 - 2.0 * t) * 0.4
            elif dist < 30.0:
                t = max(0.0, min(1.0, (dist - 15.0) / 15.0))
                dust_weight = 0.4 + t * t * (3.0 - 2.0 * t) * 0.5
            else:
                t = max(0.0, min(1.0, (dist - 30.0) / 20.0))
                dust_weight = 0.9 + t * t * (3.0 - 2.0 * t) * 0.1
        
        self.dust_fbo.use()
        self.ctx.clear(1.0, 0.0, 0.0, 1.0) # R=1.0 (full transmittance), G=0.0 (no dust depth)
        self.ctx.disable(self.ctx.BLEND)
        self.ctx.disable(self.ctx.DEPTH_TEST)
        
        if dust_weight > 0.001:
            vp = self.camera.get_projection_matrix(bh.aspect_ratio) * self.camera.get_view_matrix()
            if "u_camPos" in self.dust_program:
                self.dust_program["u_camPos"].value = self.camera.get_position()
            if "u_invVP" in self.dust_program:
                self.dust_program["u_invVP"].write(glm.inverse(vp))
            if "u_dustWeight" in self.dust_program:
                self.dust_program["u_dustWeight"].value = dust_weight
            if "u_galaxyModel" in self.dust_program:
                # To rotate dust WITH galaxy, we pass inverse galaxy rotation so camera effectively rotates
                inv_model = glm.rotate(glm.mat4(1.0), -self.galaxy_rotation_angle, glm.vec3(0.0, 1.0, 0.0))
                self.dust_program["u_galaxyModel"].write(inv_model)
            
            # Dust tunable params
            if "u_dustOpacity" in self.dust_program: self.dust_program["u_dustOpacity"].value = 1.2
            if "u_dustWidth" in self.dust_program: self.dust_program["u_dustWidth"].value = 0.25
            if "u_dustArmOffset" in self.dust_program: self.dust_program["u_dustArmOffset"].value = -0.12
            if "u_dustNoiseScale" in self.dust_program: self.dust_program["u_dustNoiseScale"].value = 0.8
            if "u_dustDensity" in self.dust_program: self.dust_program["u_dustDensity"].value = 2.5
            
            # New upgraded dust params
            if "u_dustHeightScale" in self.dust_program: self.dust_program["u_dustHeightScale"].value = 1.0
            if "u_dustLaneWidth" in self.dust_program: self.dust_program["u_dustLaneWidth"].value = 0.06
            if "u_dustDiffuseWidth" in self.dust_program: self.dust_program["u_dustDiffuseWidth"].value = 0.18
            if "u_camDist" in self.dust_program: self.dust_program["u_camDist"].value = dist
            
            self.dust_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # ----------------------------------------------------------
        # PASS 1.6: Galaxy Nebulae (quarter-res)
        # ----------------------------------------------------------
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
                
        self.nebula_fbo.use()
        # Clear: A=1.0 (transmittance), Depth=0.0
        self.ctx.clear(0.0, 0.0, 0.0, 1.0) 
        self.ctx.disable(self.ctx.BLEND)
        self.ctx.disable(self.ctx.DEPTH_TEST)
        
        if nebula_weight > 0.001:
            vp = self.camera.get_projection_matrix(bh.aspect_ratio) * self.camera.get_view_matrix()
            if "u_camPos" in self.nebula_program:
                self.nebula_program["u_camPos"].value = self.camera.get_position()
            if "u_invVP" in self.nebula_program:
                self.nebula_program["u_invVP"].write(glm.inverse(vp))
            if "u_nebulaWeight" in self.nebula_program:
                self.nebula_program["u_nebulaWeight"].value = nebula_weight
            if "u_galaxyModel" in self.nebula_program:
                inv_model = glm.rotate(glm.mat4(1.0), -self.galaxy_rotation_angle, glm.vec3(0.0, 1.0, 0.0))
                self.nebula_program["u_galaxyModel"].write(inv_model)
            if "u_camDist" in self.nebula_program:
                self.nebula_program["u_camDist"].value = dist
                
            self.nebula_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # ----------------------------------------------------------
        # PASS 1.75: Composite dust & nebulae onto scene
        # ----------------------------------------------------------
        self.comp_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(self.ctx.BLEND)
        
        self.scene_tex.use(location=0)
        self.dust_tex.use(location=1)
        self.scene_depth_tex = self.depth_tex
        self.scene_depth_tex.use(location=4)
        self.nebula_col_tex.use(location=2)
        self.nebula_depth_tex.use(location=3)
        
        self.comp_program["u_scene"].value = 0
        self.comp_program["u_dust"].value = 1
        self.comp_program["u_nebulaColor"].value = 2
        self.comp_program["u_nebulaDepth"].value = 3
        self.comp_program["u_sceneDepth"].value = 4
        if "u_debugDust" in self.comp_program:
            self.comp_program["u_debugDust"].value = self.debug_dust
        if "u_debugNebula" in self.comp_program:
            self.comp_program["u_debugNebula"].value = self.debug_nebula
        if "u_nearPlane" in self.comp_program:
            self.comp_program["u_nearPlane"].value = 0.01
        if "u_farPlane" in self.comp_program:
            self.comp_program["u_farPlane"].value = 10000.0
        self.comp_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # ----------------------------------------------------------
        # PASS 2
        # Post Processing
        # ----------------------------------------------------------

        self.ctx.screen.use()

        self.ctx.clear(0.02, 0.02, 0.03, 1.0)
        self.ctx.disable(self.ctx.DEPTH_TEST)

        self.comp_tex.use(location=0)

        self.post_program["scene"].value = 0

        self.quad_vao.render(mode=self.ctx.TRIANGLE_STRIP)


if __name__ == "__main__":
    mglw.run_window_config(MilkyWaySimulation)