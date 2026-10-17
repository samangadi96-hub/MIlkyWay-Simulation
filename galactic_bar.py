import numpy as np
import moderngl
import glm


class GalacticBar:
    """
    Procedurally generated 3D stellar bar for the Milky Way simulation.
    
    Constructed from an authentic 3D stellar distribution:
    - High-contrast elongated density distribution along the galactic major axis
    - High-density central core blending seamlessly into the nuclear bulge
    - 3D boxy / peanut-shaped vertical structure (X-shape) near the center
    - Distinct spindle profile with elliptical width tapering towards the ends
    - Smooth taper and curvature at the ends transitioning directly into inner spiral arms
    - Deterministic random population of older stars (warm white, gold, subtle orange)
    - Fully compatible with the existing star rendering pipeline and coordinate transforms.
    """

    def __init__(self, ctx, base_dir, num_stars=600000, seed=42, diagnostic_mode=False):
        self.ctx = ctx
        self.num_stars = num_stars
        self.seed = seed
        self.diagnostic_mode = diagnostic_mode

        # Load existing star shaders
        with open(base_dir / "shaders" / "bulge_vert.glsl", encoding="utf-8") as f:
            vert_code = f.read()
        with open(base_dir / "shaders" / "bulge_frag.glsl", encoding="utf-8") as f:
            frag_code = f.read()

        self.program = self.ctx.program(
            vertex_shader=vert_code,
            fragment_shader=frag_code,
        )

        # ==========================================================
        # Procedural Bar Parameters (Milky Way scale)
        # ==========================================================
        # Semi-major axis: reaches r ~ 7.2 where spiral arms fully form
        BAR_LENGTH = 7.2
        
        # In-plane orientation angle: aligns bar ends with inner spiral arms
        # Arms 0 and 2 at r ~ 6.0 - 7.0
        BAR_ANGLE = -0.349  # ~ -20 degrees

        # Transverse in-plane semi-width at center (tight aspect ratio ~ 3.8 : 1)
        BAR_WIDTH_0 = 0.82

        # Vertical thickness base scale
        BAR_HEIGHT_BASE = 0.50

        # Peanut lobe parameters (boxy/peanut bulge)
        PEANUT_PEAK_U = 2.2
        PEANUT_LOBE_H = 0.65

        rng = np.random.default_rng(self.seed)

        # ==========================================================
        # 1. Major Axis (u) Sampling
        # ==========================================================
        # To create a distinctly elongated stellar bar (not washed out into
        # a sphere), we use a dual-component distribution:
        # - 30% dense central bar core (concentrated around center)
        # - 70% extended bar spine (Ferrers density profile with high density
        #   sustained along the length, smoothly tapering at the tips)
        n_core = int(self.num_stars * 0.30)
        n_spine = self.num_stars - n_core

        # Core component
        u_core = rng.normal(0.0, 1.85, size=int(n_core * 1.25))
        u_core = u_core[np.abs(u_core) < BAR_LENGTH][:n_core]
        while len(u_core) < n_core:
            extra = rng.normal(0.0, 1.85, size=n_core - len(u_core))
            u_core = np.concatenate((u_core, extra[np.abs(extra) < BAR_LENGTH]))
        u_core = u_core[:n_core]

        # Spine component: Ferrers profile P(u) ~ (1 - (u/L)^2)^0.65
        u_spine = []
        while len(u_spine) < n_spine:
            cand = rng.uniform(-BAR_LENGTH, BAR_LENGTH, size=(n_spine - len(u_spine)) * 2)
            p = (1.0 - (cand / BAR_LENGTH) ** 2) ** 0.65
            acc = cand[rng.random(len(cand)) < p]
            u_spine.extend(acc)
        u_spine = np.array(u_spine[:n_spine])

        u = np.concatenate((u_core, u_spine))
        u_abs = np.abs(u)

        # ==========================================================
        # 2. Minor In-Plane Axis (v) Sampling
        # ==========================================================
        # Elliptical spindle width: narrow and well-defined, tapering toward the ends
        u_norm = u_abs / BAR_LENGTH
        width_profile = np.sqrt(np.maximum(0.04, 1.0 - 0.78 * (u_norm ** 2)))
        v = rng.normal(0.0, BAR_WIDTH_0 * width_profile, size=self.num_stars)

        # Smooth pitch curvature at the tips: as the bar ends reach the inner arms,
        # they smoothly curve into the logarithmic spiral pitch direction
        def smoothstep_np(e0, e1, x):
            t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        tip_curve = np.sign(u) * smoothstep_np(4.5, BAR_LENGTH, u_abs) * 0.52 * (u_abs - 4.5)
        v += tip_curve

        # ==========================================================
        # 3. Vertical Axis (w / height) - Boxy / Peanut Structure
        # ==========================================================
        # Real galactic bars exhibit a peanut/boxy profile due to vertical
        # buckling resonances, peaking around |u| ~ 1.6 - 2.8 and tapering thin
        # toward the ends to match the surrounding stellar disk.
        h_disk_blend = BAR_HEIGHT_BASE * (1.0 - 0.55 * smoothstep_np(3.2, BAR_LENGTH, u_abs))
        lobe_elevation = PEANUT_LOBE_H * (u_abs / PEANUT_PEAK_U) * np.exp(1.0 - (u_abs / PEANUT_PEAK_U) ** 1.35)

        # 45% of stars in the resonant zone (0.7 < |u| < 4.0) populate the upper and lower peanut lobes
        is_peanut_zone = (u_abs > 0.7) & (u_abs < 4.0)
        in_lobe = is_peanut_zone & (rng.random(self.num_stars) < 0.45)
        lobe_sign = rng.choice([-1.0, 1.0], size=self.num_stars)

        w = np.zeros(self.num_stars, dtype=np.float64)
        w[in_lobe] = lobe_sign[in_lobe] * lobe_elevation[in_lobe] + rng.normal(0.0, 0.24, size=np.sum(in_lobe))

        # The remaining stars form the continuous core and disc plane
        core_boost = 0.18 * np.exp(-0.5 * ((u_abs[~in_lobe] - PEANUT_PEAK_U) / 1.2) ** 2)
        w[~in_lobe] = rng.normal(0.0, h_disk_blend[~in_lobe] + core_boost)

        # ==========================================================
        # 4. Transform from Bar-Local (u, v, w) to Galactic (x, y, z)
        # ==========================================================
        cos_a = np.cos(BAR_ANGLE)
        sin_a = np.sin(BAR_ANGLE)

        x = u * cos_a - v * sin_a
        z = u * sin_a + v * cos_a
        y = w

        # Maintain a clean horizon around Sgr A* (r >= 0.25)
        r_3d = np.sqrt(x * x + y * y + z * z)
        too_close = r_3d < 0.25
        if np.any(too_close):
            scale_out = 0.25 / np.maximum(r_3d[too_close], 1e-4)
            x[too_close] *= scale_out
            y[too_close] *= scale_out
            z[too_close] *= scale_out

        positions = np.column_stack((x, y, z)).astype('f4')

        # ==========================================================
        # 5. Astrophysical Stellar Population Generation
        # ==========================================================
        # Primarily older & intermediate-age stars (K, M, and G dwarfs/giants)
        # Warm golden population that complements the central bulge.
        from stellar_population import generate_stellar_population
        colors, sizes, brightness = generate_stellar_population(
            self.num_stars,
            population_type='bar',
            rng=rng,
            diagnostic_mode=getattr(self, 'diagnostic_mode', False)
        )

        star_data = np.column_stack((positions, colors, sizes, brightness)).astype('f4')

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
