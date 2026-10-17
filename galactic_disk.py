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
        ARM_WIDTH = 0.35            # Base angular width
        ARM_STRENGTH = 0.80         # Proportion of stars in arms
        INNER_ARM_RADIUS = 3.5      # Where arms emerge smoothly around bar ends
        OUTER_RADIUS = 45.0
        
        DISK_SCALE = 8.0
        
        # In-plane bar orientation in polar coordinates (matches GalacticBar 28 deg tilt)
        BAR_ANGLE_POLAR = np.pi / 2.0 - 0.488  # ~ 1.0828 rad (62.0 degrees)
        
        def smoothstep_np(e0, e1, x):
            t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        
        # 1. Determine Arm vs Inter-arm Membership
        is_arm = rng.random(self.num_stars) < ARM_STRENGTH
        num_arm_stars = np.sum(is_arm)
        num_inter_stars = self.num_stars - num_arm_stars
        
        # 2. Sample Radii for Arms and Inter-arm Populations
        radius = np.zeros(self.num_stars, dtype=np.float64)
        
        # Arm stars: emerge smoothly from inner bar region (r >= 3.5) and sweep outwards
        r_arm = rng.gamma(shape=2.0, scale=DISK_SCALE, size=num_arm_stars)
        inv_arm = (r_arm < INNER_ARM_RADIUS) | (r_arm > OUTER_RADIUS)
        while np.any(inv_arm):
            r_arm[inv_arm] = rng.gamma(shape=2.0, scale=DISK_SCALE, size=np.sum(inv_arm))
            inv_arm = (r_arm < INNER_ARM_RADIUS) | (r_arm > OUTER_RADIUS)
        radius[is_arm] = r_arm
        
        # Inter-arm stars: smoothly sample throughout the disk
        r_inter = rng.gamma(shape=2.0, scale=DISK_SCALE, size=num_inter_stars)
        inv_inter = (r_inter < 3.2) | (r_inter > OUTER_RADIUS)
        while np.any(inv_inter):
            r_inter[inv_inter] = rng.gamma(shape=2.0, scale=DISK_SCALE, size=np.sum(inv_inter))
            inv_inter = (r_inter < 3.2) | (r_inter > OUTER_RADIUS)
        
        # 3. Calculate Angles
        theta = np.zeros(self.num_stars)
        
        # Arm stars: distributed along logarithmic spirals aligned with bar tips
        # At r = 7.0, Arm 0 matches near bar tip and Arm 2 matches far bar tip
        ARM_BASE_OFFSET = BAR_ANGLE_POLAR - SPIRAL_TIGHTNESS * np.log(7.0 / INNER_ARM_RADIUS)
        
        arm_probs = np.array([0.28, 0.22, 0.28, 0.22])  # Scutum-Centaurus & Perseus dominant
        arm_indices = rng.choice(ARM_COUNT, p=arm_probs, size=num_arm_stars)
        arm_base_angles = arm_indices * (2 * np.pi / ARM_COUNT) + ARM_BASE_OFFSET
        
        # Logarithmic spiral angle: theta = theta0 + b * log(r / INNER_ARM_RADIUS)
        ideal_theta = arm_base_angles + SPIRAL_TIGHTNESS * np.log(radius[is_arm] / INNER_ARM_RADIUS)
        
        arm_widths = np.array([1.0, 1.1, 0.95, 1.05]) * ARM_WIDTH
        spread = rng.normal(0.0, arm_widths[arm_indices], num_arm_stars)
        theta[is_arm] = ideal_theta + spread
        
        # Inter-arm stars: smoothly depleted along the bar minor axis inside r < 7.5
        # (astrophysical inter-bar resonance cavity, completely free of circular step discontinuities)
        theta_inter = rng.uniform(0.0, 2.0 * np.pi, num_inter_stars)
        minor_axis_prox = np.abs(np.sin(theta_inter - BAR_ANGLE_POLAR))  # 1.0 along minor axis, 0.0 along bar major axis
        depletion = np.where(r_inter < 7.5, (1.0 - smoothstep_np(3.0, 7.5, r_inter)) * (0.88 * minor_axis_prox), 0.0)
        rejected = rng.random(num_inter_stars) < depletion
        
        attempts = 0
        while np.any(rejected) and attempts < 10:
            n_rej = np.sum(rejected)
            new_r = rng.gamma(shape=2.0, scale=DISK_SCALE, size=n_rej)
            inv = (new_r < 3.2) | (new_r > OUTER_RADIUS)
            while np.any(inv):
                new_r[inv] = rng.gamma(shape=2.0, scale=DISK_SCALE, size=np.sum(inv))
                inv = (new_r < 3.2) | (new_r > OUTER_RADIUS)
            new_th = rng.uniform(0.0, 2.0 * np.pi, n_rej)
            new_prox = np.abs(np.sin(new_th - BAR_ANGLE_POLAR))
            new_dep = np.where(new_r < 7.5, (1.0 - smoothstep_np(3.0, 7.5, new_r)) * (0.88 * new_prox), 0.0)
            acc = rng.random(n_rej) >= new_dep
            if np.any(acc):
                sub_indices = np.where(rejected)[0][acc]
                r_inter[sub_indices] = new_r[acc]
                theta_inter[sub_indices] = new_th[acc]
            curr_prox = np.abs(np.sin(theta_inter - BAR_ANGLE_POLAR))
            curr_dep = np.where(r_inter < 7.5, (1.0 - smoothstep_np(3.0, 7.5, r_inter)) * (0.88 * curr_prox), 0.0)
            rejected = rng.random(num_inter_stars) < curr_dep
            attempts += 1
            
        radius[~is_arm] = r_inter
        theta[~is_arm] = theta_inter
        
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
            arm_factor[is_arm] = np.exp(-0.5 * (norm_spread ** 2))
            
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
