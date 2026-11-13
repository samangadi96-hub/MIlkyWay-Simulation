import math
from pathlib import Path
import numpy as np
import moderngl
import glm


class GalacticDisk:
    """
    Stellar disk of the Milky Way simulation with time-dependent, radius-dependent
    differential galactic rotation and persistent Lin-Shu density wave kinematics.
    
    Architectural separation of density wave and stellar orbital motion:
    - Exactly FOUR dominant, recognizable major spiral arms (Scutum-Centaurus,
      Sagittarius, Perseus, Norma) maintained stably across all simulation times.
    - Pattern frame rotates coherently at pattern speed Omega_p.
    - Arm-tracing population (young stars, OB associations, clumpy spine) remains
      concentrated in the persistent density wave without indefinite winding.
    - Diffuse older stellar population exhibits genuine differential orbital motion:
          omega(r) = v_flat / sqrt(r*r + r_core*r_core)
      drifting continuously through and between the spiral arms while reinforcing
      the density wave through gentle Lin-Shu epicyclic compression.
    """

    def __init__(self, ctx, base_dir, num_stars=2000000, nebula_stars=None, seed=42, diagnostic_mode=False, far_accumulator=None):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.num_stars = num_stars
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
        self.compression = 0.08  # Gentle, physically consistent Lin-Shu wave compression (sub-caustic)

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

        # ==========================================================
        # 1. Dual-Component Stellar Disk (Structured Arms + Diffuse Disc)
        # ==========================================================
        # Continuous underlying disk + four major spiral-arm overdensities:
        # 48% Arm stars concentrated along the 4 spiral arms (young OB & star-forming tracers)
        # 52% Diffuse background disk stars (continuous disk covering inner galaxy through inter-arm space)
        self.n_arm = int(self.num_stars * 0.48)
        self.n_diffuse = self.num_stars - self.n_arm
        n_arm = self.n_arm
        n_diffuse = self.n_diffuse

        # --- A. Arm Population (Structured, Clumpy, Layered) ---
        # Sample radii proportional to P(R) ~ R * Sigma(R) (Gamma distribution shape=2.0)
        r_arm = rng.gamma(shape=2.0, scale=7.5, size=int(n_arm * 1.35))
        r_arm = r_arm[(r_arm >= 3.2) & (r_arm <= self.outer_radius)]
        fade_prob = smoothstep_np(2.8, 4.2, r_arm) * (1.0 - smoothstep_np(38.0, 46.0, r_arm))
        r_arm = r_arm[rng.random(len(r_arm)) < fade_prob][:n_arm]
        while len(r_arm) < n_arm:
            extra = rng.gamma(shape=2.0, scale=7.5, size=n_arm - len(r_arm))
            extra = extra[(extra >= 3.2) & (extra <= self.outer_radius)]
            fade = smoothstep_np(2.8, 4.2, extra) * (1.0 - smoothstep_np(38.0, 46.0, extra))
            r_arm = np.concatenate((r_arm, extra[rng.random(len(extra)) < fade]))
        r_arm = r_arm[:n_arm]

        # 4 Spiral arms with realistic astrophysical asymmetry:
        # Arm 0 (Scutum-Centaurus, major): 31%
        # Arm 1 (Sagittarius, minor): 20%
        # Arm 2 (Perseus, major): 30%
        # Arm 3 (Norma, minor): 19%
        arm_indices = rng.choice(4, size=n_arm, p=[0.31, 0.20, 0.30, 0.19])
        arm_base_angles = arm_indices * (2.0 * math.pi / 4.0)

        # Logarithmic spiral spine
        arm_spine_arm = self.arm_base_offset + self.spiral_tightness * np.log(r_arm / self.inner_arm_radius)
        ideal_theta_arm = arm_base_angles + arm_spine_arm

        # Along-arm density wave modulation (star-forming knots, clumps & gaps):
        # Controlled amplitude prevents duplicate-track splitting while preserving authentic clumps
        s_norm = (r_arm - self.inner_arm_radius) / (self.outer_radius - self.inner_arm_radius)
        phi_arm = arm_indices * 1.5708
        knot_offset = 0.12 * np.sin(2.0 * math.pi * 3.2 * s_norm + phi_arm) + 0.06 * np.sin(2.0 * math.pi * 6.5 * s_norm)

        # Azimuthal Gaussian width: two-tier astrophysical arm structure
        # Tier 1 (55%): dense star-forming spine, knots and bright clusters
        # Tier 2 (45%): broad stellar envelope connecting smoothly into the disk
        is_dense_spine = rng.random(n_arm) < 0.55
        sigma_theta_dense = 0.11 + 0.08 * (r_arm / self.outer_radius)
        sigma_theta_broad = 0.24 + 0.16 * (r_arm / self.outer_radius)
        sigma_theta = np.where(is_dense_spine, sigma_theta_dense, sigma_theta_broad)

        theta_arm = ideal_theta_arm + rng.normal(0.0, sigma_theta, size=n_arm) + np.where(is_dense_spine, knot_offset, knot_offset * 0.4)

        # Arm thickness: young stars are in a very thin disk
        thickness_arm = 0.12 + 0.14 * (r_arm / self.outer_radius)
        y_arm = rng.normal(0.0, thickness_arm, size=n_arm)

        # --- B. Diffuse Inter-Arm Population (Continuous Underlying Disc) ---
        # Continuous stellar disk covering from inner transition (r ~ 1.2) through the main disk
        # Sampled proportional to polar area element P(R) ~ R * Sigma(R) = R * exp(-R / R_d)
        r_diff = rng.gamma(shape=2.0, scale=6.8, size=int(n_diffuse * 1.35))
        r_diff = r_diff[(r_diff >= 1.2) & (r_diff <= self.outer_radius)]
        fade_diff = smoothstep_np(1.0, 2.5, r_diff) * (1.0 - smoothstep_np(38.0, 46.0, r_diff))
        r_diff = r_diff[rng.random(len(r_diff)) < fade_diff][:n_diffuse]
        while len(r_diff) < n_diffuse:
            extra = rng.gamma(shape=2.0, scale=6.8, size=n_diffuse - len(r_diff))
            extra = extra[(extra >= 1.2) & (extra <= self.outer_radius)]
            fade = smoothstep_np(1.0, 2.5, extra) * (1.0 - smoothstep_np(38.0, 46.0, extra))
            r_diff = np.concatenate((r_diff, extra[rng.random(len(extra)) < fade]))
        r_diff = r_diff[:n_diffuse]

        # Guiding centers are strictly uniform across [0, 2*pi) so differential rotation
        # leaves the underlying background distribution completely stationary and free
        # of winding tracks or caustic spiral rings over all simulation times.
        theta_diff = rng.uniform(0.0, 2.0 * math.pi, size=n_diffuse)

        # Diffuse disk vertical distribution: physically motivated thin-disk (~280 pc) and thick-disk (~800 pc)
        # Using model conversion 1 unit ~ 700 pc: h_thin ~ 0.40, h_thick ~ 1.14
        # Isothermal sech^2(z / z0) equilibrium profile
        is_thick = rng.random(n_diffuse) < 0.25
        h_thin = 0.35 + 0.20 * (r_diff / self.outer_radius)
        h_thick = 0.85 + 0.35 * (r_diff / self.outer_radius)
        scale_h = np.where(is_thick, h_thick, h_thin)

        u_samp = np.clip(rng.uniform(0.005, 0.995, size=n_diffuse), 0.005, 0.995)
        y_diff = scale_h * 0.5 * np.log((1.0 + (2.0 * u_samp - 1.0)) / (1.0 - (2.0 * u_samp - 1.0) + 1e-7))

        # Combine components
        radius = np.concatenate((r_arm, r_diff)).astype('f4')
        theta = np.concatenate((theta_arm, theta_diff)).astype('f4')
        y = np.concatenate((y_arm, y_diff)).astype('f4')

        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        positions = np.column_stack((x, y, z)).astype('f4')

        # 4. Continuous Arm-Proximity & Astrophysical Stellar Population
        r_spine = np.maximum(radius, self.inner_arm_radius)
        arm_spine_total = self.arm_base_offset + self.spiral_tightness * np.log(r_spine / self.inner_arm_radius)
        psi = 4.0 * (theta - arm_spine_total)
        psi_wrap = (psi + math.pi) % (2.0 * math.pi) - math.pi
        arm_factor = 0.5 + 0.5 * np.cos(psi_wrap)
        is_arm_star = np.arange(self.num_stars) < n_arm
        arm_factor = np.where(is_arm_star, np.clip(arm_factor * 1.25, 0.0, 1.0), arm_factor * 0.45)

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

        if far_accumulator is not None:
            far_accumulator.accumulate_disk(positions, colors, brightness, n_arm)

        self.vbo = self.ctx.buffer(star_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 1f 1f', 'in_position', 'in_color', 'in_size', 'in_brightness')]
        )

        # Nebula-associated stars: rotate rigidly at pattern speed with the nebulae
        # so differential shearing never separates young stellar clusters from their parent clouds.
        self.nebula_vao = None
        if nebula_stars is not None and len(nebula_stars) > 0:
            with open(self.base_dir / "shaders" / "bulge_vert.glsl", encoding="utf-8") as f:
                neb_vert_code = f.read()
            self.nebula_program = self.ctx.program(
                vertex_shader=neb_vert_code,
                fragment_shader=frag_code,
            )
            self.nebula_vbo = self.ctx.buffer(np.ascontiguousarray(nebula_stars, dtype='f4').tobytes())
            self.nebula_vao = self.ctx.vertex_array(
                self.nebula_program,
                [(self.nebula_vbo, '3f 3f 1f 1f', 'in_position', 'in_color', 'in_size', 'in_brightness')]
            )
            self.num_stars += len(nebula_stars)

    def render(self, camera, aspect_ratio, visibility, rotation_angle, sim_time=0.0, depth_pass=False):
        if visibility <= 0.001:
            return

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        if depth_pass:
            self.ctx.disable(moderngl.BLEND)
        else:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        if "u_depthPass" in self.program:
            self.program["u_depthPass"].value = 1 if depth_pass else 0
        if "u_visibility" in self.program:
            self.program["u_visibility"].value = visibility

        if "u_view" in self.program:
            self.program["u_view"].write(camera.get_view_matrix())
        if "u_projection" in self.program:
            self.program["u_projection"].write(camera.get_projection_matrix(aspect_ratio))
        model = glm.rotate(glm.mat4(1.0), rotation_angle, glm.vec3(0.0, 1.0, 0.0))
        if "u_model" in self.program:
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
        if "u_barAnglePolar" in self.program:
            self.program["u_barAnglePolar"].value = self.bar_angle_polar

        # 1. Arm-tracer population: coherent density-wave pattern frame
        # Stars maintain persistent 4-arm structure, asymmetry, and clumpy spine without winding
        if "u_driftScale" in self.program:
            self.program["u_driftScale"].value = 0.0
        if "u_compression" in self.program:
            self.program["u_compression"].value = 0.0
        self.vao.render(moderngl.POINTS, vertices=self.n_arm, first=0)

        # 2. Diffuse orbiting population: differential rotation through density wave
        # Stars orbit with radius-dependent angular velocity omega(r), compressing into
        # the wave as they pass through, without shearing the large-scale pattern
        if "u_driftScale" in self.program:
            self.program["u_driftScale"].value = 1.0
        if "u_compression" in self.program:
            self.program["u_compression"].value = self.compression
        self.vao.render(moderngl.POINTS, vertices=self.n_diffuse, first=self.n_arm)

        # Render nebula-associated stars with coherent pattern speed
        if self.nebula_vao is not None:
            if "u_depthPass" in self.nebula_program:
                self.nebula_program["u_depthPass"].value = 1 if depth_pass else 0
            if "u_visibility" in self.nebula_program:
                self.nebula_program["u_visibility"].value = visibility
            if "u_view" in self.nebula_program:
                self.nebula_program["u_view"].write(camera.get_view_matrix())
            if "u_projection" in self.nebula_program:
                self.nebula_program["u_projection"].write(camera.get_projection_matrix(aspect_ratio))
            if "u_model" in self.nebula_program:
                self.nebula_program["u_model"].write(model)
            self.nebula_vao.render(moderngl.POINTS)
