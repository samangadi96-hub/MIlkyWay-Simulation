import numpy as np
import moderngl
import glm

class GalacticDisk:
    def __init__(self, ctx, base_dir, nebula_stars=None, seed=42, diagnostic_mode=False):
        self.ctx = ctx
        self.num_stars = 2000000
        self.seed = seed
        self.diagnostic_mode = diagnostic_mode
        
        # Load shaders (reusing bulge shaders for now, but they will have u_galaxyAngle)
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
        # Spiral Galaxy Parameters
        # ==========================================================
        ARM_COUNT = 4
        SPIRAL_TIGHTNESS = 2.5      # 'b' in theta = b * log(r/r0)
        ARM_WIDTH = 0.35       # Base angular width
        ARM_STRENGTH = 0.85     # Max proportion of stars in arms
        INNER_RADIUS = 2.0
        OUTER_RADIUS = 45.0
        BULGE_RADIUS = 6.0          # Where arms fully emerge
        
        DISK_SCALE = 8.0
        
        # 1. Generate Radii
        # Use Gamma distribution (shape=2) to get probability P(r) ~ r * exp(-r / DISK_SCALE)
        # This accounts for increasing circumference area at larger radii.
        radius = rng.gamma(shape=2.0, scale=DISK_SCALE, size=self.num_stars)
        
        # Truncate via rejection sampling instead of clipping to avoid artificial rings
        invalid = (radius < INNER_RADIUS) | (radius > OUTER_RADIUS)
        while np.any(invalid):
            radius[invalid] = rng.gamma(shape=2.0, scale=DISK_SCALE, size=np.sum(invalid))
            invalid = (radius < INNER_RADIUS) | (radius > OUTER_RADIUS)
        
        # 2. Determine Arm vs Inter-arm Population (Density Waves)
        # Arms emerge smoothly from the bulge
        def smoothstep(edge0, edge1, x):
            t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
            
        arm_strength = smoothstep(INNER_RADIUS, BULGE_RADIUS, radius)
        
        # Probability of a star belonging to an arm
        p_arm = ARM_STRENGTH * arm_strength
        is_arm = rng.random(self.num_stars) < p_arm
        
        # 3. Calculate Angles
        theta = np.zeros(self.num_stars)
        
        # Inter-arm stars: uniformly distributed in angle
        theta[~is_arm] = rng.uniform(0.0, 2.0 * np.pi, np.sum(~is_arm))
        
        # Arm stars: distributed along logarithmic spirals
        num_arm_stars = np.sum(is_arm)
        
        # Slightly vary density between arms to make it less perfect
        arm_probs = np.array([0.27, 0.23, 0.26, 0.24])
        arm_indices = rng.choice(ARM_COUNT, p=arm_probs, size=num_arm_stars)
        arm_base_angles = arm_indices * (2 * np.pi / ARM_COUNT)
        
        # Base logarithmic spiral angle: theta = theta0 + b * log(r / r0)
        ideal_theta = arm_base_angles + SPIRAL_TIGHTNESS * np.log(radius[is_arm] / INNER_RADIUS)
        
        # Slightly vary width between arms
        arm_widths = np.array([1.0, 1.1, 0.9, 1.05]) * ARM_WIDTH
        
        # Angular spread: tighter near the core, slightly wider at the edges
        spread = rng.normal(0.0, arm_widths[arm_indices], num_arm_stars)
        
        theta[is_arm] = ideal_theta + spread
        
        # 4. 3D Structure
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        
        # Thickness: very thin compared to radius, slightly thicker in the middle
        thickness_scale = 0.2 + 0.3 * np.exp(-radius / 10.0)
        y = rng.normal(0.0, 1.0, self.num_stars) * thickness_scale
        
        positions = np.column_stack((x, y, z)).astype('f4')
        
        # ==========================================================
        # 5. Continuous Arm-Proximity & Astrophysical Stellar Population
        # ==========================================================
        # Calculate arm-proximity factor A in [0, 1] for continuous population mixing:
        # Near arm spine, young hot OB fraction is elevated.
        # Inter-arm regions retain a rich, balanced mix of older & intermediate stars.
        arm_factor = np.zeros(self.num_stars, dtype=np.float64)
        if num_arm_stars > 0:
            norm_spread = np.abs(spread) / arm_widths[arm_indices]
            arm_factor[is_arm] = np.exp(-0.5 * (norm_spread ** 2)) * arm_strength[is_arm]
            
        from stellar_population import generate_stellar_population
        colors, sizes, brightness = generate_stellar_population(
            self.num_stars,
            population_type='disk',
            rng=rng,
            arm_factor=arm_factor,
            radius=radius,
            diagnostic_mode=self.diagnostic_mode
        )
        
        star_data = np.column_stack((positions, colors, sizes, brightness)).astype('f4')
        
        if nebula_stars is not None and len(nebula_stars) > 0:
            star_data = np.vstack((star_data, nebula_stars))
            self.num_stars += len(nebula_stars)
        
        self.vbo = self.ctx.buffer(star_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 1f 1f', 'in_position', 'in_color', 'in_size', 'in_brightness')]
        )
        
    def render(self, camera, aspect_ratio, visibility, rotation_angle):
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
