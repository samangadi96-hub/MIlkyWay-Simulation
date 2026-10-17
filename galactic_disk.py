import math
from pathlib import Path
import numpy as np
import moderngl
import glm


class GalacticDisk:
    """
    Stellar disk of the Milky Way simulation with time-dependent, radius-dependent
    differential galactic rotation and persistent Lin-Shu density wave kinematics.
    
    Individual stars orbit with differential angular velocity:
        omega(r) = v_flat / sqrt(r*r + r_core*r_core)
    approaching flat tangential velocity v_flat at large radii.
    
    The central bar and large-scale 4-arm spiral density wave rotate coherently at
    pattern speed Omega_p (corotation near the bar tips r ~ 7.5).
    Stars drift continuously relative to the spiral pattern without winding up or
    blurring into concentric rings.
    """

    def __init__(self, ctx, base_dir, nebula_stars=None, seed=42, diagnostic_mode=False):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.num_stars = 2000000
        self.seed = seed
        self.diagnostic_mode = diagnostic_mode

        # Differential rotation and spiral density wave parameters
        self.v_flat = 0.85
        self.r_core = 3.2
        self.r_cr = 7.5
        self.pattern_speed = self.v_flat / math.sqrt(self.r_cr ** 2 + self.r_core ** 2)

        self.spiral_tightness = 2.5
        self.inner_arm_radius = 3.5
        self.outer_radius = 45.0
        self.disk_scale = 8.0

        # In-plane bar orientation in polar coordinates (matches GalacticBar 28 deg tilt)
        self.bar_angle_polar = math.pi / 2.0 - 0.488  # ~ 1.0828 rad
        self.arm_base_offset = self.bar_angle_polar - self.spiral_tightness * math.log(7.0 / self.inner_arm_radius)
        self.compression = 0.22

        # Load dedicated disk shaders supporting GPU differential rotation
        with open(self.base_dir / "shaders" / "disk_vert.glsl", encoding="utf-8") as f:
            vert_code = f.read()
        with open(self.base_dir / "shaders" / "bulge_frag.glsl", encoding="utf-8") as f:
            frag_code = f.read()

        self.program = self.ctx.program(
            vertex_shader=vert_code,
            fragment_shader=frag_code,
        )

        rng = np.random.default_rng(self.seed)

        def smoothstep_np(e0, e1, x):
            t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # 1. Sample Radii for Disk Stars (smooth exponential/gamma distribution)
        radius = rng.gamma(shape=2.0, scale=self.disk_scale, size=self.num_stars)
        inv = (radius < self.inner_arm_radius) | (radius > self.outer_radius)
        while np.any(inv):
            radius[inv] = rng.gamma(shape=2.0, scale=self.disk_scale, size=np.sum(inv))
            inv = (radius < self.inner_arm_radius) | (radius > self.outer_radius)

        # 2. Guiding Center Azimuthal Distribution
        # Uniform across [0, 2*pi) with smooth depletion along the bar minor axis inside r < 7.5
        # (astrophysical inter-bar resonance cavity, keeping the 3D bar prominent)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=self.num_stars)
        minor_axis_prox = np.abs(np.sin(theta - self.bar_angle_polar))
        depletion = np.where(radius < 7.5, (1.0 - smoothstep_np(3.0, 7.5, radius)) * (0.88 * minor_axis_prox), 0.0)
        rejected = rng.random(self.num_stars) < depletion

        attempts = 0
        while np.any(rejected) and attempts < 10:
            n_rej = np.sum(rejected)
            new_r = rng.gamma(shape=2.0, scale=self.disk_scale, size=n_rej)
            inv = (new_r < self.inner_arm_radius) | (new_r > self.outer_radius)
            while np.any(inv):
                new_r[inv] = rng.gamma(shape=2.0, scale=self.disk_scale, size=np.sum(inv))
                inv = (new_r < self.inner_arm_radius) | (new_r > self.outer_radius)
            new_th = rng.uniform(0.0, 2.0 * math.pi, size=n_rej)
            new_prox = np.abs(np.sin(new_th - self.bar_angle_polar))
            new_dep = np.where(new_r < 7.5, (1.0 - smoothstep_np(3.0, 7.5, new_r)) * (0.88 * new_prox), 0.0)
            acc = rng.random(n_rej) >= new_dep
            if np.any(acc):
                sub_indices = np.where(rejected)[0][acc]
                radius[sub_indices] = new_r[acc]
                theta[sub_indices] = new_th[acc]
            curr_prox = np.abs(np.sin(theta - self.bar_angle_polar))
            curr_dep = np.where(radius < 7.5, (1.0 - smoothstep_np(3.0, 7.5, radius)) * (0.88 * curr_prox), 0.0)
            rejected = rng.random(self.num_stars) < curr_dep
            attempts += 1

        # 3. 3D Spatial Distribution
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        thickness_scale = 0.2 + 0.3 * np.exp(-radius / 10.0)
        y = rng.normal(0.0, 1.0, self.num_stars) * thickness_scale

        positions = np.column_stack((x, y, z)).astype('f4')

        # 4. Continuous Arm-Proximity & Astrophysical Stellar Population
        arm_spine = self.arm_base_offset + self.spiral_tightness * np.log(radius / self.inner_arm_radius)
        psi = 4.0 * (theta - arm_spine)
        psi_wrap = (psi + math.pi) % (2.0 * math.pi) - math.pi
        arm_factor = 0.5 + 0.5 * np.cos(psi_wrap)

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

        # GPU-side differential rotation and density wave uniforms
        if "u_simTime" in self.program:
            self.program["u_simTime"].value = sim_time
        if "u_vFlat" in self.program:
            self.program["u_vFlat"].value = self.v_flat
        if "u_rCore" in self.program:
            self.program["u_rCore"].value = self.r_core
        if "u_patternSpeed" in self.program:
            self.program["u_patternSpeed"].value = self.pattern_speed
        if "u_armTightness" in self.program:
            self.program["u_armTightness"].value = self.spiral_tightness
        if "u_armOffset" in self.program:
            self.program["u_armOffset"].value = self.arm_base_offset
        if "u_innerArmR" in self.program:
            self.program["u_innerArmR"].value = self.inner_arm_radius
        if "u_compression" in self.program:
            self.program["u_compression"].value = self.compression

        self.vao.render(moderngl.POINTS)
