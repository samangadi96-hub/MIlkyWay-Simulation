import math
from pathlib import Path
import numpy as np
import moderngl
import glm


class FarGalaxy:
    """
    Genuine far-distance galaxy Level-of-Detail (LOD) representation.
    
    Derives directly from the underlying 3D galaxy stellar populations:
    - 2.85 million stars from GalacticBulge, GalacticBar, and GalacticDisk
    - Warm golden Population II core and elongated central bar (28 deg tilt)
    - 4 asymmetric spiral arms with celestial blue-white OB star-forming ridges
    - Smooth diffuse older stellar disc with exponential radial falloff
    - Feathery dark dust lanes running along the bar leading edges and inner spiral arms
    - Localized H-II emission complexes and starburst beads matching NebulaManager
    
    Rendered as a layered 3D volumetric slab on the actual galactic plane
    to provide perspective-correct views from face-on, oblique, and edge-on orientations.
    """

    def __init__(self, ctx, base_dir, nebula_mgr=None, disk=None, bar=None, bulge=None, res=1024):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.res = res
        self.half_size = 48.0

        # Load dedicated shaders
        with open(self.base_dir / "shaders" / "far_galaxy_vert.glsl", encoding="utf-8") as f:
            vert_code = f.read()
        with open(self.base_dir / "shaders" / "far_galaxy_frag.glsl", encoding="utf-8") as f:
            frag_code = f.read()

        self.program = self.ctx.program(
            vertex_shader=vert_code,
            fragment_shader=frag_code,
        )

        # ------------------------------------------------------------------
        # 1. Precompute Multichannel Galaxy Representation
        # ------------------------------------------------------------------
        stellar_data, dust_neb_data = self._generate_far_textures(nebula_mgr, disk, bar, bulge)

        self.stellar_tex = self.ctx.texture((self.res, self.res), 4, data=stellar_data.tobytes(), dtype='f2')
        self.stellar_tex.filter = (self.ctx.LINEAR_MIPMAP_LINEAR, self.ctx.LINEAR)
        self.stellar_tex.build_mipmaps()

        self.dust_nebula_tex = self.ctx.texture((self.res, self.res), 4, data=dust_neb_data.tobytes(), dtype='f2')
        self.dust_nebula_tex.filter = (self.ctx.LINEAR_MIPMAP_LINEAR, self.ctx.LINEAR)
        self.dust_nebula_tex.build_mipmaps()

        # ------------------------------------------------------------------
        # 2. Build 3D Slab Geometry (Layered Horizontal Slices)
        # ------------------------------------------------------------------
        # Slices along Y provide physical vertical thickness for oblique/edge-on views
        # while summing smoothly to 1.0 total weight for face-on views.
        hs = self.half_size
        self.layers = [
            {"y":  0.00, "weight": 0.36},
            {"y":  0.28, "weight": 0.22},
            {"y": -0.28, "weight": 0.22},
            {"y":  0.62, "weight": 0.10},
            {"y": -0.62, "weight": 0.10},
        ]

        quad_verts = []
        for l in self.layers:
            y = l["y"]
            # 2 triangles (6 vertices) per layer: (x, y, z, u, v)
            quad_verts.extend([-hs, y, -hs, 0.0, 0.0])
            quad_verts.extend([ hs, y, -hs, 1.0, 0.0])
            quad_verts.extend([-hs, y,  hs, 0.0, 1.0])
            quad_verts.extend([-hs, y,  hs, 0.0, 1.0])
            quad_verts.extend([ hs, y, -hs, 1.0, 0.0])
            quad_verts.extend([ hs, y,  hs, 1.0, 1.0])

        self.vbo = self.ctx.buffer(np.array(quad_verts, dtype='f4').tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 2f', 'in_position', 'in_texcoord')]
        )

    def _generate_far_textures(self, nebula_mgr, disk=None, bar=None, bulge=None):
        """Precomputes stellar radiance and dust/nebula multichannel textures from galaxy stars."""
        res = self.res
        hs = self.half_size

        x_coords = np.linspace(-hs, hs, res, dtype=np.float32)
        z_coords = np.linspace(-hs, hs, res, dtype=np.float32)
        X, Z = np.meshgrid(x_coords, z_coords)
        R = np.sqrt(X * X + Z * Z)
        Theta = np.arctan2(Z, X)

        def fft_gaussian_blur(arr, sigma):
            H, W = arr.shape
            ky = np.fft.fftfreq(H)[:, None]
            kx = np.fft.rfftfreq(W)[None, :]
            filter_fft = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (ky ** 2 + kx ** 2))
            f_arr = np.fft.rfft2(arr)
            return np.fft.irfft2(f_arr * filter_fft).astype(np.float32)

        def smoothstep_np(e0, e1, x):
            t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # ----------------------------------------------------------
        # 1. Read or Generate Stellar Data
        # ----------------------------------------------------------
        if disk is not None and hasattr(disk, 'vbo') and bar is not None and bulge is not None:
            d_disk = np.frombuffer(disk.vbo.read(), dtype=np.float32).reshape(-1, 8)
            d_bar = np.frombuffer(bar.vbo.read(), dtype=np.float32).reshape(-1, 8)
            d_bulge = np.frombuffer(bulge.vbo.read(), dtype=np.float32).reshape(-1, 8)
        else:
            # Deterministic generation matching GalacticDisk, GalacticBar, GalacticBulge
            from galaxy import GalacticBulge, GalacticBar
            from galactic_disk import GalacticDisk
            _b = GalacticBulge(self.ctx, self.base_dir)
            _bar = GalacticBar(self.ctx, self.base_dir)
            _d = GalacticDisk(self.ctx, self.base_dir)
            d_bulge = np.frombuffer(_b.vbo.read(), dtype=np.float32).reshape(-1, 8)
            d_bar = np.frombuffer(_bar.vbo.read(), dtype=np.float32).reshape(-1, 8)
            d_disk = np.frombuffer(_d.vbo.read(), dtype=np.float32).reshape(-1, 8)
            for _obj in [_b, _bar, _d]:
                if hasattr(_obj, 'vbo'): _obj.vbo.release()
                if hasattr(_obj, 'vao'): _obj.vao.release()
                if hasattr(_obj, 'program'): _obj.program.release()

        # Separate disk into arm and diffuse components
        n_arm = getattr(disk, "n_arm", int(len(d_disk) * 0.48))
        x_arm, z_arm = d_disk[:n_arm, 0], d_disk[:n_arm, 2]
        rgb_arm, lum_arm = d_disk[:n_arm, 3:6], d_disk[:n_arm, 7]
        r_arm_stars = np.sqrt(x_arm * x_arm + z_arm * z_arm)
        th_arm_stars = np.arctan2(z_arm, x_arm)

        x_diff, z_diff = d_disk[n_arm:, 0], d_disk[n_arm:, 2]
        rgb_diff, lum_diff = d_disk[n_arm:, 3:6], d_disk[n_arm:, 7]

        # ----------------------------------------------------------
        # Spiral Arm Geometry & Gating (matching Reference A)
        # ----------------------------------------------------------
        ARM_COUNT = 4
        SPIRAL_TIGHTNESS = 2.5
        INNER_ARM_R = 3.5
        BAR_ANGLE = 0.488
        ARM_BASE_OFFSET = (math.pi / 2.0 - 0.488) - SPIRAL_TIGHTNESS * math.log(7.0 / INNER_ARM_R)

        r_clamped = np.maximum(r_arm_stars, INNER_ARM_R)
        arm_spine = ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * np.log(r_clamped / INNER_ARM_R)
        rel_theta = (th_arm_stars - arm_spine) % (2.0 * math.pi)
        arm_idx = np.floor(rel_theta / (2.0 * math.pi / ARM_COUNT) + 0.5).astype(int) % ARM_COUNT

        # Major arms (0 & 2: Scutum-Centaurus & Perseus) launch from bar tips (r ~ 6.0 - 7.8)
        # Minor arms (1 & 3: Sagittarius & Norma) launch further out (r ~ 9.2 - 11.5)
        # This keeps the central region dedicated to the golden bar and core as in Reference A.
        gate_major = smoothstep_np(6.0, 7.8, r_arm_stars)
        gate_minor = smoothstep_np(9.2, 11.5, r_arm_stars)
        is_major = (arm_idx == 0) | (arm_idx == 2)
        arm_gate = np.where(is_major, gate_major, gate_minor)
        lum_arm_gated = lum_arm * arm_gate

        # Histogram splatting into 1024x1024 grid
        H_arm_r, _, _ = np.histogram2d(x_arm, z_arm, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_arm[:, 0] * lum_arm_gated)
        H_arm_g, _, _ = np.histogram2d(x_arm, z_arm, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_arm[:, 1] * lum_arm_gated)
        H_arm_b, _, _ = np.histogram2d(x_arm, z_arm, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_arm[:, 2] * lum_arm_gated)
        H_arm_r, H_arm_g, H_arm_b = H_arm_r.T.astype(np.float32), H_arm_g.T.astype(np.float32), H_arm_b.T.astype(np.float32)

        H_diff_r, _, _ = np.histogram2d(x_diff, z_diff, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_diff[:, 0] * lum_diff)
        H_diff_g, _, _ = np.histogram2d(x_diff, z_diff, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_diff[:, 1] * lum_diff)
        H_diff_b, _, _ = np.histogram2d(x_diff, z_diff, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_diff[:, 2] * lum_diff)
        H_diff_r, H_diff_g, H_diff_b = H_diff_r.T.astype(np.float32), H_diff_g.T.astype(np.float32), H_diff_b.T.astype(np.float32)

        x_b, z_b = d_bar[:, 0], d_bar[:, 2]
        rgb_b, lum_b = d_bar[:, 3:6], d_bar[:, 7]
        H_bar_r, _, _ = np.histogram2d(x_b, z_b, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_b[:, 0] * lum_b)
        H_bar_g, _, _ = np.histogram2d(x_b, z_b, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_b[:, 1] * lum_b)
        H_bar_b, _, _ = np.histogram2d(x_b, z_b, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_b[:, 2] * lum_b)
        H_bar_r, H_bar_g, H_bar_b = H_bar_r.T.astype(np.float32), H_bar_g.T.astype(np.float32), H_bar_b.T.astype(np.float32)

        x_c, z_c = d_bulge[:, 0], d_bulge[:, 2]
        rgb_c, lum_c = d_bulge[:, 3:6], d_bulge[:, 7]
        H_bulge_r, _, _ = np.histogram2d(x_c, z_c, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_c[:, 0] * lum_c)
        H_bulge_g, _, _ = np.histogram2d(x_c, z_c, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_c[:, 1] * lum_c)
        H_bulge_b, _, _ = np.histogram2d(x_c, z_c, bins=res, range=[[-hs, hs], [-hs, hs]], weights=rgb_c[:, 2] * lum_c)
        H_bulge_r, H_bulge_g, H_bulge_b = H_bulge_r.T.astype(np.float32), H_bulge_g.T.astype(np.float32), H_bulge_b.T.astype(np.float32)

        # 1. Multi-scale filtering for arm:
        # Granular star clusters (sigma=1.0) + arm spine (sigma=2.8) + diffuse arm envelope (sigma=6.0)
        arm_sharp_r = fft_gaussian_blur(H_arm_r, 1.0)
        arm_sharp_g = fft_gaussian_blur(H_arm_g, 1.0)
        arm_sharp_b = fft_gaussian_blur(H_arm_b, 1.0)

        arm_med_r = fft_gaussian_blur(H_arm_r, 2.8)
        arm_med_g = fft_gaussian_blur(H_arm_g, 2.8)
        arm_med_b = fft_gaussian_blur(H_arm_b, 2.8)

        arm_broad_r = fft_gaussian_blur(H_arm_r, 6.0)
        arm_broad_g = fft_gaussian_blur(H_arm_g, 6.0)
        arm_broad_b = fft_gaussian_blur(H_arm_b, 6.0)

        arm_r = (arm_sharp_r * 0.40 + arm_med_r * 0.42 + arm_broad_r * 0.18) * 0.42
        arm_g = (arm_sharp_g * 0.40 + arm_med_g * 0.42 + arm_broad_g * 0.18) * 0.42
        arm_b = (arm_sharp_b * 0.40 + arm_med_b * 0.42 + arm_broad_b * 0.18) * 0.42

        # Give arms their signature Reference A celestial blue-white hue
        arm_trans = smoothstep_np(6.0, 9.5, R)
        arm_tint_r = (1.0 - arm_trans) * 1.15 + arm_trans * 0.68
        arm_tint_g = (1.0 - arm_trans) * 1.05 + arm_trans * 0.85
        arm_tint_b = (1.0 - arm_trans) * 0.75 + arm_trans * 1.48
        arm_r *= arm_tint_r
        arm_g *= arm_tint_g
        arm_b *= arm_tint_b

        # 2. Diffuse disc
        diff_r = fft_gaussian_blur(H_diff_r, 5.0) * 0.22
        diff_g = fft_gaussian_blur(H_diff_g, 5.0) * 0.22
        diff_b = fft_gaussian_blur(H_diff_b, 5.0) * 0.26

        # 3. Bar: Crisp elongated spine (sigma=1.2) + subtle body (sigma=2.6)
        # Preserving authentic aspect ratio (~3.5:1) without over-blurring into a circular haze
        bar_sharp_r = fft_gaussian_blur(H_bar_r, 1.2)
        bar_sharp_g = fft_gaussian_blur(H_bar_g, 1.2)
        bar_sharp_b = fft_gaussian_blur(H_bar_b, 1.2)

        bar_smooth_r = fft_gaussian_blur(H_bar_r, 2.6)
        bar_smooth_g = fft_gaussian_blur(H_bar_g, 2.6)
        bar_smooth_b = fft_gaussian_blur(H_bar_b, 2.6)

        bar_raw_r = bar_sharp_r * 0.65 + bar_smooth_r * 0.35
        bar_raw_g = bar_sharp_g * 0.65 + bar_smooth_g * 0.35
        bar_raw_b = bar_sharp_b * 0.65 + bar_smooth_b * 0.35

        # Calibrate bar surface brightness (~7.5) with rich warm gold tint matching Reference A
        bar_lum = np.sqrt(bar_raw_r**2 + bar_raw_g**2 + bar_raw_b**2)
        bar_scale = 1.0 / (1.0 + bar_lum * 0.12)
        bar_r = bar_raw_r * bar_scale * 4.2 * 1.42
        bar_g = bar_raw_g * bar_scale * 4.2 * 1.18
        bar_b = bar_raw_b * bar_scale * 4.2 * 0.60

        # 4. Central Nuclear Bulge: tight compact core peaking around Sgr A*
        bulge_sharp_r = fft_gaussian_blur(H_bulge_r, 1.0)
        bulge_sharp_g = fft_gaussian_blur(H_bulge_g, 1.0)
        bulge_sharp_b = fft_gaussian_blur(H_bulge_b, 1.0)

        bulge_lum = np.sqrt(bulge_sharp_r**2 + bulge_sharp_g**2 + bulge_sharp_b**2)
        bulge_scale = 1.0 / (1.0 + bulge_lum * 0.10)
        bulge_r = bulge_sharp_r * bulge_scale * 1.25 * 1.42
        bulge_g = bulge_sharp_g * bulge_scale * 1.25 * 1.25
        bulge_b = bulge_sharp_b * bulge_scale * 1.25 * 0.88

        # Combine stellar components
        stars_r = arm_r + diff_r + bar_r + bulge_r
        stars_g = arm_g + diff_g + bar_g + bulge_g
        stars_b = arm_b + diff_b + bar_b + bulge_b

        # Outer disk falloff: smooth exponential decay starting at r=35
        outer_mask = np.where(R <= 35.0, 1.0, np.exp(-0.5 * ((R - 35.0) / 4.5)**2))
        stars_r *= outer_mask
        stars_g *= outer_mask
        stars_b *= outer_mask

        stellar_density = np.sqrt(stars_r**2 + stars_g**2 + stars_b**2)
        alpha_map = np.clip(stellar_density * 0.50, 0.0, 1.0) * outer_mask

        stellar_img = np.zeros((res, res, 4), dtype=np.float32)
        stellar_img[:, :, 0] = stars_r
        stellar_img[:, :, 1] = stars_g
        stellar_img[:, :, 2] = stars_b
        stellar_img[:, :, 3] = alpha_map

        # ----------------------------------------------------------
        # 2. Dust Map Generation (Filamentary lanes along inner arms)
        # ----------------------------------------------------------
        noise_warp_x = (
            0.35 * np.sin(X * 0.55 + Z * 0.40) +
            0.18 * np.sin(X * 1.35 - Z * 1.10) +
            0.08 * np.cos(X * 2.85 + Z * 2.45)
        )
        noise_warp_z = (
            0.35 * np.cos(Z * 0.55 - X * 0.40) +
            0.18 * np.cos(Z * 1.35 + X * 1.10) +
            0.08 * np.sin(Z * 2.85 - X * 2.45)
        )

        X_warped = X + noise_warp_x * 0.80
        Z_warped = Z + noise_warp_z * 0.80
        R_warped = np.sqrt(X_warped * X_warped + Z_warped * Z_warped)
        Theta_warped = np.arctan2(Z_warped, X_warped)
        r_arm_w = np.maximum(R_warped, INNER_ARM_R)

        noise_dust = (
            0.40 * np.sin(X * 0.48 + Z * 0.38) * np.cos(Z * 0.58 - X * 0.28) +
            0.30 * np.sin(X * 1.25 - Z * 0.95) * np.sin(Z * 1.35 + X * 0.75) +
            0.18 * np.cos(X * 2.65 + Z * 2.35) * np.sin(Z * 3.05 - X * 2.55)
        )

        arm_dust_total = np.zeros_like(R)
        for i in range(ARM_COUNT):
            is_maj = (i % 2 == 0)
            r_start = 6.6 if is_maj else 10.5
            w_arm = 1.0 if is_maj else 0.60

            arm_base = float(i) * (2.0 * math.pi / ARM_COUNT)
            ideal_theta = arm_base + ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * np.log(r_arm_w / INNER_ARM_R)
            d_theta = (Theta_warped - ideal_theta + math.pi) % (2.0 * math.pi) - math.pi

            d_arm = d_theta - (-0.14)
            linear_dist = np.abs(d_arm) * R_warped

            lane_w = (0.040 + 0.030 * (R / 45.0)) * R
            core_lane = np.exp(-0.5 * (linear_dist / lane_w)**2) * w_arm * 2.5

            d_arm_sub = d_theta - (-0.02)
            sub_lane = np.exp(-0.5 * (np.abs(d_arm_sub) * R_warped / (lane_w * 1.5))**2) * w_arm * 0.65

            s_norm = (R - INNER_ARM_R) / 41.5
            wisp_mod = 0.85 + 0.35 * np.sin(2.0 * math.pi * 5.5 * s_norm + i * 1.57) + 0.25 * noise_dust

            lane_combined = (core_lane + sub_lane) * np.clip(wisp_mod, 0.20, 1.80)
            arm_dust_total += lane_combined * smoothstep_np(r_start - 0.5, r_start + 1.2, R) * (R <= 42.0)

        dust_optical_depth = arm_dust_total * outer_mask

        # ----------------------------------------------------------
        # 3. Nebulae Map (Organic H-II complexes & bar starbursts)
        # ----------------------------------------------------------
        neb_rgb = np.zeros((res, res, 3), dtype=np.float32)

        if nebula_mgr is not None and hasattr(nebula_mgr, 'nebulae_data'):
            for n in nebula_mgr.nebulae_data:
                nx, ny, nz, nr = n['position']
                col = np.array(n['color'][:3], dtype=np.float32) * n['color'][3]
                ix0 = max(0, int((nx - 3.0 * nr + hs) / (2.0 * hs) * res))
                ix1 = min(res, int((nx + 3.0 * nr + hs) / (2.0 * hs) * res) + 1)
                iz0 = max(0, int((nz - 3.0 * nr + hs) / (2.0 * hs) * res))
                iz1 = min(res, int((nz + 3.0 * nr + hs) / (2.0 * hs) * res) + 1)
                if ix1 > ix0 and iz1 > iz0:
                    subX = X[iz0:iz1, ix0:ix1]
                    subZ = Z[iz0:iz1, ix0:ix1]
                    d2 = (subX - nx)**2 + (subZ - nz)**2
                    mask = np.exp(-0.5 * d2 / (nr * 0.40)**2) * 1.8 + np.exp(-0.5 * d2 / (nr * 0.80)**2) * 0.4
                    for c in range(3):
                        neb_rgb[iz0:iz1, ix0:ix1, c] += mask * col[c]

        # Clustered H-II star-forming beads and young OB clusters along spiral arm shock fronts
        rng_neb = np.random.default_rng(888)
        num_beads = 480
        r_beads = rng_neb.uniform(6.8, 39.0, num_beads)
        arm_beads = rng_neb.choice(4, size=num_beads, p=[0.38, 0.12, 0.38, 0.12])
        base_th_beads = arm_beads * (2.0 * math.pi / 4.0)
        th_beads = base_th_beads + ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * np.log(r_beads / INNER_ARM_R) + rng_neb.normal(-0.06, 0.035, num_beads)

        x_beads = list(r_beads * np.cos(th_beads))
        z_beads = list(r_beads * np.sin(th_beads))
        # Delicate pinpoint knots matching Reference A scale
        size_beads = list(rng_neb.uniform(0.10, 0.26, num_beads))
        int_beads = list(rng_neb.uniform(0.5, 1.4, num_beads))

        # Starburst complexes at the two bar tips (where gas crashes into inner arms)
        sin_b = math.sin(BAR_ANGLE)
        cos_b = math.cos(BAR_ANGLE)
        for tip_sign in [-1.0, 1.0]:
            tip_u = tip_sign * 6.6
            for _ in range(16):
                u_clump = tip_u + rng_neb.normal(0.0, 0.50)
                v_clump = tip_sign * 0.95 + rng_neb.normal(0.0, 0.35)
                x_clump = u_clump * sin_b - v_clump * cos_b
                z_clump = u_clump * cos_b + v_clump * sin_b
                x_beads.append(x_clump)
                z_beads.append(z_clump)
                size_beads.append(rng_neb.uniform(0.12, 0.30))
                int_beads.append(rng_neb.uniform(0.7, 1.6))

        for bx, bz, bs, bi in zip(x_beads, z_beads, size_beads, int_beads):
            ix0 = max(0, int((bx - 3.0 * bs + hs) / (2.0 * hs) * res))
            ix1 = min(res, int((bx + 3.0 * bs + hs) / (2.0 * hs) * res) + 1)
            iz0 = max(0, int((bz - 3.0 * bs + hs) / (2.0 * hs) * res))
            iz1 = min(res, int((bz + 3.0 * bs + hs) / (2.0 * hs) * res) + 1)
            if ix1 > ix0 and iz1 > iz0:
                subX = X[iz0:iz1, ix0:ix1]
                subZ = Z[iz0:iz1, ix0:ix1]
                d2 = (subX - bx)**2 + (subZ - bz)**2
                mask = np.exp(-0.5 * d2 / (bs * 0.30)**2) * bi + np.exp(-0.5 * d2 / (bs * 0.70)**2) * (bi * 0.20)
                is_blue = rng_neb.random() < 0.32
                if is_blue:
                    # Luminous young blue/cyan star cluster knot
                    b_col = np.array([0.55, 0.88, 1.40], dtype=np.float32)
                else:
                    # Natural H-alpha emission (magenta/pink)
                    hue_var = rng_neb.uniform(-0.05, 0.05)
                    b_col = np.array([1.25 + hue_var, 0.20, 0.52 - hue_var], dtype=np.float32)
                for c in range(3):
                    neb_rgb[iz0:iz1, ix0:ix1, c] += mask * b_col[c]

        # Assemble Dust & Nebula Texture (RGBA16F):
        # Channel 0 (R): Dust optical depth
        # Channel 1 (G): Nebula Red
        # Channel 2 (B): Nebula Green
        # Channel 3 (A): Nebula Blue
        dust_neb_img = np.zeros((res, res, 4), dtype=np.float32)
        dust_neb_img[:, :, 0] = dust_optical_depth
        dust_neb_img[:, :, 1] = neb_rgb[:, :, 0]
        dust_neb_img[:, :, 2] = neb_rgb[:, :, 1]
        dust_neb_img[:, :, 3] = neb_rgb[:, :, 2]

        return stellar_img.astype(np.float16), dust_neb_img.astype(np.float16)

    def render(self, camera, aspect_ratio, visibility, rotation_angle):
        """Renders the far galaxy representation on the galactic plane in 3D perspective."""
        if visibility <= 0.001:
            return

        # Pure additive accumulation for pre-weighted layer slices
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE

        # Bind precomputed maps
        self.stellar_tex.use(location=0)
        self.dust_nebula_tex.use(location=1)

        if "u_stellarMap" in self.program:
            self.program["u_stellarMap"].value = 0
        if "u_dustNebulaMap" in self.program:
            self.program["u_dustNebulaMap"].value = 1
        if "u_visibility" in self.program:
            self.program["u_visibility"].value = visibility

        if "u_view" in self.program:
            self.program["u_view"].write(camera.get_view_matrix())
        if "u_projection" in self.program:
            self.program["u_projection"].write(camera.get_projection_matrix(aspect_ratio))

        # Rotate strictly in synchrony with the galaxy pattern
        model = glm.rotate(glm.mat4(1.0), rotation_angle, glm.vec3(0.0, 1.0, 0.0))
        if "u_model" in self.program:
            self.program["u_model"].write(model)

        # Render layered slices (6 vertices per slice)
        num_layers = len(self.layers)
        for i in range(num_layers):
            if "u_layerWeight" in self.program:
                self.program["u_layerWeight"].value = self.layers[i]["weight"]
            self.vao.render(mode=moderngl.TRIANGLES, vertices=6, first=i * 6)

        # Restore standard point-star additive blending equation
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    def render_depth(self, camera, aspect_ratio, rotation_angle):
        """Writes central galactic plane depth for dust and nebula occlusion."""
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = True

        if "u_view" in self.program:
            self.program["u_view"].write(camera.get_view_matrix())
        if "u_projection" in self.program:
            self.program["u_projection"].write(camera.get_projection_matrix(aspect_ratio))
        model = glm.rotate(glm.mat4(1.0), rotation_angle, glm.vec3(0.0, 1.0, 0.0))
        if "u_model" in self.program:
            self.program["u_model"].write(model)
        if "u_visibility" in self.program:
            self.program["u_visibility"].value = 1.0
        if "u_layerWeight" in self.program:
            self.program["u_layerWeight"].value = 1.0

        self.stellar_tex.use(location=0)
        self.dust_nebula_tex.use(location=1)
        self.vao.render(mode=moderngl.TRIANGLES, vertices=6, first=0)

    def release(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.stellar_tex.release()
        self.dust_nebula_tex.release()
        self.program.release()
