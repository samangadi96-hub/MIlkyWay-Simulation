import numpy as np
import moderngl
import glm
from galactic_bar import GalacticBar


class GalacticBulge:
    """
    Central nuclear bulge of the Milky Way simulation.
    
    Concentrated spheroidal/ellipsoidal core surrounding Sagittarius A*,
    providing high central stellar density that blends seamlessly into
    the surrounding 3D GalacticBar without washing it out into a circular haze.
    """

    def __init__(self, ctx, base_dir, num_stars=250000, seed=1337, diagnostic_mode=False):
        self.ctx = ctx
        self.num_stars = num_stars
        self.seed = seed
        self.diagnostic_mode = diagnostic_mode
        
        # Load shaders
        with open(base_dir / "shaders" / "bulge_vert.glsl", encoding="utf-8") as f:
            vert_code = f.read()
        with open(base_dir / "shaders" / "bulge_frag.glsl", encoding="utf-8") as f:
            frag_code = f.read()
            
        self.program = self.ctx.program(
            vertex_shader=vert_code,
            fragment_shader=frag_code,
        )
        
        rng = np.random.default_rng(self.seed)
        
        # ==========================================================
        # Generate Central Nuclear Bulge Stars
        # ==========================================================
        # 65% exponential central core + 35% inner spheroidal halo
        core_count = int(self.num_stars * 0.65)
        halo_count = self.num_stars - core_count
        
        # Core: dense steep exponential falloff around Sgr A*
        r_core = rng.exponential(0.32, core_count)
        theta_core = rng.uniform(0.0, 2.0 * np.pi, core_count)
        phi_core = np.arccos(rng.uniform(-1.0, 1.0, core_count))
        
        x_core = r_core * np.sin(phi_core) * np.cos(theta_core)
        z_core = r_core * np.sin(phi_core) * np.sin(theta_core)
        y_core = r_core * np.cos(phi_core) * 0.35
        
        # Halo: compact spheroidal nucleus (r <= 1.0)
        x_halo = rng.normal(0.0, 0.45, halo_count)
        z_halo = rng.normal(0.0, 0.45, halo_count)
        y_halo = rng.normal(0.0, 0.28, halo_count)
        
        x = np.concatenate((x_core, x_halo))
        z = np.concatenate((z_core, z_halo))
        y = np.concatenate((y_core, y_halo))
        
        # Maintain clean event horizon / accretion disk clearing (r >= 0.25)
        r_3d = np.sqrt(x * x + y * y + z * z)
        too_close = r_3d < 0.25
        if np.any(too_close):
            scale_out = 0.25 / np.maximum(r_3d[too_close], 1e-4)
            x[too_close] *= scale_out
            y[too_close] *= scale_out
            z[too_close] *= scale_out
            
        positions = np.column_stack((x, y, z)).astype('f4')
        
        # ==========================================================
        # Astrophysical Stellar Population Generation
        # ==========================================================
        # Older, metal-rich population II & old disk stars:
        # Predominantly K and M stars with G subgiants/dwarfs and red clump giants.
        from stellar_population import generate_stellar_population
        colors, sizes, brightness = generate_stellar_population(
            self.num_stars,
            population_type='bulge',
            rng=rng,
            diagnostic_mode=getattr(self, 'diagnostic_mode', False)
        )
        
        star_data = np.column_stack((positions, colors, sizes, brightness)).astype('f4')
        
        self.vbo = self.ctx.buffer(star_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 1f 1f', 'in_position', 'in_color', 'in_size', 'in_brightness')]
        )
        
    def render(self, camera, aspect_ratio, visibility, rotation_angle, sim_time=0.0):
        if visibility <= 0.001:
            return
            
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        
        if "u_visibility" in self.program:
            self.program["u_visibility"].value = visibility
            
        if "u_view" in self.program:
            self.program["u_view"].write(camera.get_view_matrix())
        if "u_projection" in self.program:
            self.program["u_projection"].write(camera.get_projection_matrix(aspect_ratio))
        if "u_model" in self.program:
            model = glm.rotate(glm.mat4(1.0), rotation_angle, glm.vec3(0.0, 1.0, 0.0))
            self.program["u_model"].write(model)
            
        self.vao.render(moderngl.POINTS)

