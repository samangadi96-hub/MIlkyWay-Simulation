import math
import numpy as np
import moderngl


class ISMVolumeManager:
    """
    Precomputed 3D Interstellar Medium (ISM) generator for Phase 3.

    Generates once at initialization (zero per-frame CPU overhead):
      1. Seamless tileable 3D multi-scale noise volume (64^3 RGBA8) containing
         broad FBM, ridged filaments, periodic Worley clumps, and fine turbulence.
      2. High-resolution galactic-plane ISM spiral/bar structure map (512^2 RGBA16F)
         following galactic_disk.py's exact 4-arm logarithmic spiral & curved bar
         leading-edge shocks, plus spiral spurs/feathers and diffuse inter-arm dust.
      3. True 3D Molecular Cloud & Local H-II Illumination Volume (384x384x24 RGBA16F)
         built from a parent-child catalog linking:
           Spiral Arm -> Dense Dust/Gas Lane -> Molecular Cloud Complex
           -> Irregular Clumps & Filaments -> Local Star-Forming Nebulae.

    Physical scale conversion verified from galactic_disk.py:
      1 simulation unit ~= 700 pc
      Thin dust component scale height  :  81 pc ~= 0.1157 units
      Thicker diffuse dust scale height : 152 pc ~= 0.2171 units
      Cold molecular gas scale height   :  55 pc ~= 0.0786 units
    """

    # Verified galactic coordinate constants (matching galactic_disk.py & nebula.py)
    UNIT_PC = 700.0
    H_DUST_THIN = 81.0 / UNIT_PC    # ~0.1157 units
    H_DUST_THICK = 152.0 / UNIT_PC  # ~0.2171 units
    H_MOL_CLOUD = 55.0 / UNIT_PC    # ~0.0786 units

    ARM_COUNT = 4
    SPIRAL_TIGHTNESS = 2.5
    INNER_ARM_R = 3.5
    OUTER_RADIUS = 45.0
    BAR_ANGLE = 0.488
    ARM_BASE_OFFSET = (math.pi / 2.0 - BAR_ANGLE) - SPIRAL_TIGHTNESS * math.log(7.0 / INNER_ARM_R)

    MACRO_HALF_EXTENT = 47.5
    MOL_HALF_EXTENT = 47.0
    MOL_Y_HALF = 0.90

    def __init__(self, ctx: moderngl.Context, nebula_mgr=None, seed: int = 2026):
        self.ctx = ctx
        self.seed = seed
        self.cloud_catalog = []

        # 1. Precomputed 3D tileable multi-scale noise texture (64^3, ~1 MB)
        noise_vol = self._generate_periodic_noise_3d(size=64, seed=self.seed + 11)
        self.noise_vol_tex = self.ctx.texture3d(
            (64, 64, 64), 4,
            data=(np.clip(noise_vol, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes(),
            dtype='f1'
        )
        self.noise_vol_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.noise_vol_tex.repeat_x = True
        self.noise_vol_tex.repeat_y = True
        self.noise_vol_tex.repeat_z = True

        # 2. Galactic plane macro ISM map (512^2 RGBA16F, ~2 MB)
        spiral_map = self._generate_spiral_ism_map(res=512, seed=self.seed + 37)
        self.spiral_map_tex = self.ctx.texture(
            (512, 512), 4,
            data=spiral_map.astype(np.float16).tobytes(),
            dtype='f2'
        )
        self.spiral_map_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.spiral_map_tex.repeat_x = False
        self.spiral_map_tex.repeat_y = False

        # 3. 3D Molecular Cloud & Local H-II Illumination Volume (384x384x24 RGBA16F, ~28 MB)
        mol_vol = self._generate_molecular_clouds_volume(
            nebula_mgr=nebula_mgr, nx=384, nz=384, ny=24, seed=self.seed + 101
        )
        self.mol_cloud_tex = self.ctx.texture3d(
            (384, 384, 24), 4,
            data=mol_vol.astype(np.float16).tobytes(),
            dtype='f2'
        )
        self.mol_cloud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.mol_cloud_tex.repeat_x = False
        self.mol_cloud_tex.repeat_y = False
        self.mol_cloud_tex.repeat_z = False

    @staticmethod
    def _smoothstep(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _periodic_lattice_3d(grid_n: int, period: int, rng: np.random.Generator) -> np.ndarray:
        """Vectorized C2-continuous periodic 3D value noise on [0, grid_n)^3."""
        lat = rng.random((period, period, period), dtype=np.float32)
        c = np.arange(grid_n, dtype=np.float32) * (period / float(grid_n))
        i0 = np.floor(c).astype(np.int32) % period
        i1 = (i0 + 1) % period
        f = c - np.floor(c)
        u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)

        uz = u[:, None, None]
        uy = u[None, :, None]
        ux = u[None, None, :]

        c000 = lat[np.ix_(i0, i0, i0)]
        c100 = lat[np.ix_(i0, i0, i1)]
        c010 = lat[np.ix_(i0, i1, i0)]
        c110 = lat[np.ix_(i0, i1, i1)]
        c001 = lat[np.ix_(i1, i0, i0)]
        c101 = lat[np.ix_(i1, i0, i1)]
        c011 = lat[np.ix_(i1, i1, i0)]
        c111 = lat[np.ix_(i1, i1, i1)]

        c00 = c000 * (1.0 - ux) + c100 * ux
        c10 = c010 * (1.0 - ux) + c110 * ux
        c01 = c001 * (1.0 - ux) + c101 * ux
        c11 = c011 * (1.0 - ux) + c111 * ux

        c0 = c00 * (1.0 - uy) + c10 * uy
        c1 = c01 * (1.0 - uy) + c11 * uy
        return (c0 * (1.0 - uz) + c1 * uz).astype(np.float32)

    @staticmethod
    def _periodic_worley_3d(grid_n: int, period: int, rng: np.random.Generator) -> np.ndarray:
        """Tileable cellular clumping field computed fast via half-res evaluation + trilinear upsample."""
        eval_n = max(period * 4, grid_n // 2) if grid_n >= 48 else grid_n
        pts = rng.uniform(0.12, 0.88, size=(period, period, period, 3)).astype(np.float32)
        g = (np.arange(eval_n, dtype=np.float32) + 0.5) * (period / float(eval_n))
        cz, cy, cx = np.meshgrid(g, g, g, indexing='ij')
        iz = np.floor(cz).astype(np.int32)
        iy = np.floor(cy).astype(np.int32)
        ix = np.floor(cx).astype(np.int32)
        fz = cz - iz
        fy = cy - iy
        fx = cx - ix

        min_d2 = np.full((eval_n, eval_n, eval_n), 10.0, dtype=np.float32)
        for dz in (-1, 0, 1):
            pz = (iz + dz) % period
            for dy in (-1, 0, 1):
                py = (iy + dy) % period
                for dx in (-1, 0, 1):
                    px = (ix + dx) % period
                    p = pts[pz, py, px]
                    ddz = p[..., 0] + dz - fz
                    ddy = p[..., 1] + dy - fy
                    ddx = p[..., 2] + dx - fx
                    min_d2 = np.minimum(min_d2, ddx * ddx + ddy * ddy + ddz * ddz)

        field = np.clip(1.0 - np.sqrt(min_d2) * 1.15, 0.0, 1.0).astype(np.float32)
        if eval_n == grid_n:
            return field

        # Smooth periodic trilinear upsample to grid_n
        c = np.arange(grid_n, dtype=np.float32) * (eval_n / float(grid_n))
        i0 = np.floor(c).astype(np.int32) % eval_n
        i1 = (i0 + 1) % eval_n
        f = c - np.floor(c)
        uz, uy, ux = f[:, None, None], f[None, :, None], f[None, None, :]
        c00 = field[np.ix_(i0, i0, i0)] * (1.0 - ux) + field[np.ix_(i0, i0, i1)] * ux
        c10 = field[np.ix_(i0, i1, i0)] * (1.0 - ux) + field[np.ix_(i0, i1, i1)] * ux
        c01 = field[np.ix_(i1, i0, i0)] * (1.0 - ux) + field[np.ix_(i1, i0, i1)] * ux
        c11 = field[np.ix_(i1, i1, i0)] * (1.0 - ux) + field[np.ix_(i1, i1, i1)] * ux
        c0 = c00 * (1.0 - uy) + c10 * uy
        c1 = c01 * (1.0 - uy) + c11 * uy
        return (c0 * (1.0 - uz) + c1 * uz).astype(np.float32)

    def _generate_periodic_noise_3d(self, size: int = 64, seed: int = 2037) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n4 = self._periodic_lattice_3d(size, 4, rng)
        n8 = self._periodic_lattice_3d(size, 8, rng)
        n16 = self._periodic_lattice_3d(size, 16, rng)
        n32 = self._periodic_lattice_3d(size, 32, rng)

        # Normalize helper to [0, 1] with tight contrast preservation
        def norm01(a):
            lo, hi = np.percentile(a, 1.5), np.percentile(a, 98.5)
            return np.clip((a - lo) / max(hi - lo, 1e-5), 0.0, 1.0).astype(np.float32)

        # R: Multi-octave FBM (broad cloud structure & warp)
        fbm_broad = norm01(0.50 * n4 + 0.28 * n8 + 0.14 * n16 + 0.08 * n32)

        # G: Ridged filament field: high along crisp connected filament spines
        def ridge(n):
            return (1.0 - np.abs(2.0 * n - 1.0)) ** 2.1

        ridged = norm01(0.42 * ridge(n4) + 0.30 * ridge(n8) + 0.18 * ridge(n16) + 0.10 * ridge(n32))

        # B: Multi-scale Worley clumps (dense cores + nested knots)
        w4 = self._periodic_worley_3d(size, 4, rng)
        w8 = self._periodic_worley_3d(size, 8, rng)
        clump = norm01(0.52 * w4 + 0.34 * w8 + 0.14 * n16)

        # A: Fine turbulence (frayed outer edges & high-freq wisps)
        fine = norm01(0.50 * n8 + 0.32 * n16 + 0.18 * n32)

        vol = np.stack([fbm_broad, ridged, clump, fine], axis=-1)
        return vol

    @staticmethod
    def _periodic_lattice_2d(grid_n: int, period: int, rng: np.random.Generator) -> np.ndarray:
        lat = rng.random((period, period), dtype=np.float32)
        c = np.arange(grid_n, dtype=np.float32) * (period / float(grid_n))
        i0 = np.floor(c).astype(np.int32) % period
        i1 = (i0 + 1) % period
        f = c - np.floor(c)
        u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
        ux = u[None, :]
        uy = u[:, None]
        # Fast vectorized gather via advanced broadcasting (avoids slow np.ix_ full allocation copies)
        r0 = lat[i0]
        r1 = lat[i1]
        row0 = r0[:, i0] * (1.0 - ux) + r0[:, i1] * ux
        row1 = r1[:, i0] * (1.0 - ux) + r1[:, i1] * ux
        return (row0 * (1.0 - uy) + row1 * uy).astype(np.float32)

    def _generate_spiral_ism_map(self, res: int = 512, seed: int = 2063) -> np.ndarray:
        rng = np.random.default_rng(seed)
        hs = self.MACRO_HALF_EXTENT
        step = (2.0 * hs) / res
        coords = np.linspace(-hs + 0.5 * step, hs - 0.5 * step, res, dtype=np.float32)
        X, Z = np.meshgrid(coords, coords)  # shape (res, res): axis0=Z (v), axis1=X (u)
        R = np.sqrt(X * X + Z * Z)
        Theta = np.arctan2(Z, X)

        # Multi-scale 2D deterministic seeded FBM (evaluated cleanly with periodic noise)
        def make_fbm(weights):
            acc = np.zeros((res, res), dtype=np.float32)
            tot = 0.0
            for p, w in weights:
                acc += w * self._periodic_lattice_2d(res, p, rng)
                tot += w
            return acc / max(tot, 1e-5)

        fbm_warp_x = make_fbm([(6, 0.55), (14, 0.30), (28, 0.15)]) - 0.5
        fbm_warp_z = make_fbm([(6, 0.55), (14, 0.30), (28, 0.15)]) - 0.5
        fbm_cloud = make_fbm([(5, 0.45), (11, 0.28), (22, 0.17), (44, 0.10)])
        fbm_clump = make_fbm([(9, 0.35), (20, 0.35), (42, 0.30)])
        fbm_spur = make_fbm([(12, 0.45), (28, 0.35), (56, 0.20)])

        # Domain warp scaled so filaments meander naturally without losing spiral structure
        warp_amp = 1.65 * self._smoothstep(2.5, 16.0, R) + 0.45 * self._smoothstep(0.8, 3.0, R)
        Xw = X + fbm_warp_x * warp_amp
        Zw = Z + fbm_warp_z * warp_amp
        Rw = np.sqrt(Xw * Xw + Zw * Zw)
        Tw = np.arctan2(Zw, Xw)
        r_arm_w = np.maximum(Rw, self.INNER_ARM_R)

        # Radial profile: preserve clean Sgr A* core (r < 0.7), peak in inner-mid disk, fade at R ~ 46
        inner_ramp = self._smoothstep(0.68, 2.2, R)
        disk_rad = (R * np.exp(-R / 11.8)) / (11.8 * math.exp(-1.0))
        rad_env = (0.62 * disk_rad + 0.38 * self._smoothstep(1.2, 5.5, R) * np.exp(-R / 22.0))
        outer_fade = 1.0 - self._smoothstep(35.5, 45.8, R)
        radial_mask = np.clip(inner_ramp * outer_fade, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 1. Spiral Arm Dust Lanes (physical perpendicular offset, fragmented, with spurs)
        # ------------------------------------------------------------------
        arm_thin = np.zeros((res, res), dtype=np.float32)
        arm_diff = np.zeros((res, res), dtype=np.float32)
        mol_ridge = np.zeros((res, res), dtype=np.float32)
        arm_weights = [1.05, 0.68, 1.02, 0.66]  # Major (0,2) vs Minor (1,3) matching galactic_disk.py

        s_norm = np.clip((R - self.INNER_ARM_R) / (self.OUTER_RADIUS - self.INNER_ARM_R), 0.0, 1.2)
        r_safe = np.maximum(R, 3.0)

        for arm_i in range(self.ARM_COUNT):
            is_maj = (arm_i % 2 == 0)
            arm_base = arm_i * (2.0 * math.pi / self.ARM_COUNT)
            ideal_th = arm_base + self.ARM_BASE_OFFSET + self.SPIRAL_TIGHTNESS * np.log(r_arm_w / self.INNER_ARM_R)

            # Wrapped signed angular offset relative to spiral centerline (continuous across +-pi!)
            d_th = (Tw - ideal_th + math.pi) % (2.0 * math.pi) - math.pi

            phi_arm = arm_i * 1.5708
            knot_wave = 0.07 * np.sin(2.0 * math.pi * 3.2 * s_norm + phi_arm) + 0.035 * np.sin(2.0 * math.pi * 6.5 * s_norm)
            # Physical perpendicular distance (1 unit ~ 700 pc)
            # Inner compression dust ridge meanders naturally across ~ -0.15..-0.45 units
            perp_target = -0.22 - 0.12 * self._smoothstep(8.0, 35.0, R) + 0.35 * (fbm_cloud - 0.5)
            d_perp_lane = d_th * r_safe - (perp_target + 0.35 * knot_wave * r_safe)

            r_gate = self._smoothstep(5.0, 7.2, R) if is_maj else self._smoothstep(8.8, 11.4, R)
            arm_w = arm_weights[arm_i] * r_gate

            # Primary multi-scale compression lane (wider organic filamentary zone broken further by 3D noise)
            w_core = 0.55 + 0.55 * self._smoothstep(6.0, 38.0, R)
            core_lane = np.exp(-0.5 * (d_perp_lane / w_core) ** 2)

            # Braided secondary filament weaving across stellar arm ridge
            perp_sub = 0.35 + 0.55 * np.sin(4.2 * np.log(r_arm_w / 3.5) + phi_arm * 1.7)
            d_perp_sub = d_th * r_safe - perp_sub
            w_sub = w_core * 1.15
            sub_lane = np.exp(-0.5 * (d_perp_sub / w_sub) ** 2) * 0.62

            # Fragmentation & local dense complexes (strong multi-scale clumps break uniform ribbons)
            frag_mod = 0.15 + 1.45 * self._smoothstep(0.22, 0.76, fbm_clump)
            ridge_core = (core_lane + sub_lane) * frag_mod * arm_w

            # Sheared spiral spurs & feathers peeling outward downstream at shallower pitch
            # Uses exact integer harmonics of wrapped Tw + log(r) -> ZERO branch-cut seams!
            d_perp_pos = d_th * r_safe
            spur_scale = 3.2 + 2.8 * self._smoothstep(8.0, 32.0, R)
            in_spur_band = self._smoothstep(0.02, 0.50, d_perp_pos) * np.exp(-np.maximum(0.0, d_perp_pos) / spur_scale)
            spur_phase = Tw - 1.05 * np.log(r_arm_w / 3.5)
            spur_w1 = np.cos(12.0 * spur_phase + 3.4 * (fbm_spur - 0.5) + phi_arm)
            spur_w2 = np.cos(20.0 * spur_phase + 2.6 * (fbm_clump - 0.5) - phi_arm * 1.3)
            spur_feathers = 0.70 * (np.maximum(0.0, spur_w1) ** 1.65) + 0.35 * (np.maximum(0.0, spur_w2) ** 1.9)
            spur_noise = 0.22 + 0.88 * self._smoothstep(0.24, 0.76, fbm_spur)
            spur_strength = in_spur_band * spur_feathers * spur_noise * arm_w * 1.25

            # Broad diffuse arm envelope
            w_env = 1.45 + 1.35 * self._smoothstep(6.0, 36.0, R)
            env_lane = np.exp(-0.5 * ((d_th * r_safe) / w_env) ** 2) * arm_w * 0.78

            arm_thin += ridge_core + spur_strength
            arm_diff += env_lane + 0.50 * spur_strength
            mol_ridge = np.maximum(mol_ridge, np.clip((core_lane * 1.15 + sub_lane * 0.70 + spur_strength * 0.45) * arm_w, 0.0, 1.6))

        # ------------------------------------------------------------------
        # 2. Curved Bar Leading-Edge Dust Shocks + Circumnuclear Ring
        #    (Replaces old straight scratches across galactic bar)
        # ------------------------------------------------------------------
        sin_b = math.sin(self.BAR_ANGLE)
        cos_b = math.cos(self.BAR_ANGLE)
        u_bar = Xw * sin_b + Zw * cos_b
        v_bar = -Xw * cos_b + Zw * sin_b
        u_abs = np.abs(u_bar)

        # Curve leading-edge lane smoothly from inner nuclear ring into bar tip arc
        tip_bend = np.sign(u_bar) * self._smoothstep(4.2, 7.2, u_abs) * 0.45 * np.maximum(0.0, u_abs - 4.2)
        v_shock = np.sign(u_bar) * (0.28 + 0.20 * self._smoothstep(1.6, 5.2, u_abs)) + tip_bend
        dv_bar = v_bar - v_shock

        w_bar = 0.28 + 0.16 * self._smoothstep(1.5, 6.5, u_abs)
        bar_clump_mod = 0.55 + 0.70 * self._smoothstep(0.25, 0.78, fbm_clump)
        bar_env_u = self._smoothstep(1.3, 2.5, u_abs) * (1.0 - self._smoothstep(6.3, 7.6, u_abs))
        bar_lane = np.exp(-0.5 * (dv_bar / w_bar) ** 2) * bar_env_u * bar_clump_mod * 0.88

        # Subtle clumpy circumnuclear dust ring / inner dust streams around 1.4..2.4 units
        cnr_ring = np.exp(-0.5 * ((R - 1.85) / 0.42) ** 2) * (0.45 + 0.55 * self._smoothstep(0.28, 0.72, fbm_clump)) * 0.55

        thin_total = (arm_thin + bar_lane + cnr_ring) * rad_env * radial_mask

        # ------------------------------------------------------------------
        # 3. Diffuse Inter-Arm Dust & Thick Envelope
        # ------------------------------------------------------------------
        inter_arm_noise = 0.50 + 0.95 * self._smoothstep(0.22, 0.78, fbm_cloud)
        inter_arm_base = (0.26 * rad_env + 0.14 * np.exp(-R / 19.0)) * inter_arm_noise * radial_mask
        thick_diffuse_total = (arm_diff * 0.78 * rad_env * radial_mask) + inter_arm_base

        # Diffuse cirrus wisps blended gently into thin channel for textured inter-arm extinction
        thin_total = np.clip(thin_total * 1.12 + inter_arm_base * 0.48, 0.0, 2.5)
        thick_diffuse_total = np.clip(thick_diffuse_total, 0.0, 1.75)
        mol_ridge = np.clip(
            (mol_ridge + bar_lane * 0.95 + cnr_ring * 0.75) * radial_mask * (0.70 + 0.45 * rad_env),
            0.0, 1.40
        )

        # Pack (H=res [Z], W=res [X], 4) -> OpenGL texture (width=X, height=Z)
        out = np.stack([thin_total, thick_diffuse_total, mol_ridge, radial_mask], axis=-1).astype(np.float32)
        return out

    def _deposit_ellipsoid_cloud(
        self, vol_env, vol_core, x_coords, y_coords, z_coords,
        cx, cy, cz, rx, ry, rz, yaw, pitch_tilt, strength, core_frac, rng
    ):
        """Stamp an irregular multi-lobe 3D cloud complex into its tight local AABB."""
        nx = len(x_coords)
        ny = len(y_coords)
        nz = len(z_coords)
        hs = self.MOL_HALF_EXTENT
        yh = self.MOL_Y_HALF
        dx_v = (2.0 * hs) / nx
        dy_v = (2.0 * yh) / ny
        dz_v = (2.0 * hs) / nz

        max_rxz = max(rx, rz) * 1.45
        max_ry = ry * 1.55

        ix0 = max(0, int((cx - max_rxz + hs) / dx_v))
        ix1 = min(nx, int((cx + max_rxz + hs) / dx_v) + 2)
        iz0 = max(0, int((cz - max_rxz + hs) / dz_v))
        iz1 = min(nz, int((cz + max_rxz + hs) / dz_v) + 2)
        iy0 = max(0, int((cy - max_ry + yh) / dy_v))
        iy1 = min(ny, int((cy + max_ry + yh) / dy_v) + 2)

        if ix1 <= ix0 or iz1 <= iz0 or iy1 <= iy0:
            return

        gx = x_coords[ix0:ix1][None, None, :] - cx
        gy = y_coords[iy0:iy1][:, None, None] - cy
        gz = z_coords[iz0:iz1][None, :, None] - cz

        # Rotate into cloud-aligned local frame (spiral pitch alignment + tiny vertical tilt)
        ca, sa = math.cos(yaw), math.sin(yaw)
        u = gx * ca + gz * sa
        v = -gx * sa + gz * ca
        cp, sp = math.cos(pitch_tilt), math.sin(pitch_tilt)
        w = gy * cp + v * sp
        v_rot = -gy * sp + v * cp

        # Perturb with asymmetric lobe warping for genuinely non-spherical silhouettes
        skew = rng.uniform(-0.28, 0.28)
        u_w = u + skew * v_rot
        q2 = (u_w / rx) ** 2 + (v_rot / rz) ** 2 + (w / ry) ** 2
        env = np.clip(1.0 - q2, 0.0, 1.0)
        # Soft outer skirt + dense interior gradient
        env_prof = (env ** 1.35) * strength
        core_prof = (np.clip((env - 0.35) / 0.65, 0.0, 1.0) ** 1.5) * strength * core_frac

        vol_env[iy0:iy1, iz0:iz1, ix0:ix1] = np.maximum(
            vol_env[iy0:iy1, iz0:iz1, ix0:ix1], env_prof
        )
        vol_core[iy0:iy1, iz0:iz1, ix0:ix1] = np.maximum(
            vol_core[iy0:iy1, iz0:iz1, ix0:ix1], core_prof
        )

    def _generate_molecular_clouds_volume(
        self, nebula_mgr=None, nx: int = 384, nz: int = 384, ny: int = 24, seed: int = 2127
    ) -> np.ndarray:
        """
        Construct hierarchical 3D molecular cloud complexes + local nebula radiation field.
        Texture layout: (depth=ny [Y], height=nz [Z], width=nx [X], channels=4).
        Channels:
          R: Molecular cloud complex envelope density
          G: Dense dark internal core density
          B: Local warm H-II (H-alpha) dust illumination potential
          A: Local cool OB/reflection dust illumination potential
        """
        rng = np.random.default_rng(seed)
        hs = self.MOL_HALF_EXTENT
        yh = self.MOL_Y_HALF
        dx = (2.0 * hs) / nx
        dz = (2.0 * hs) / nz
        dy = (2.0 * yh) / ny

        x_coords = np.linspace(-hs + 0.5 * dx, hs - 0.5 * dx, nx, dtype=np.float32)
        z_coords = np.linspace(-hs + 0.5 * dz, hs - 0.5 * dz, nz, dtype=np.float32)
        y_coords = np.linspace(-yh + 0.5 * dy, yh - 0.5 * dy, ny, dtype=np.float32)

        vol_env = np.zeros((ny, nz, nx), dtype=np.float32)
        vol_core = np.zeros((ny, nz, nx), dtype=np.float32)
        illum_warm = np.zeros((ny, nz, nx), dtype=np.float32)
        illum_cool = np.zeros((ny, nz, nx), dtype=np.float32)

        # Helper: exact logarithmic spiral tangent yaw in (x,z) plane
        # Note dx/dr = cos(th) - b*sin(th), dz/dr = sin(th) + b*cos(th) => angle = th + atan2(b, 1)
        def spiral_tangent_yaw(theta_pos: float) -> float:
            pitch_offset = math.atan2(self.SPIRAL_TIGHTNESS, 1.0)
            return theta_pos + pitch_offset

        # ------------------------------------------------------------------
        # 1. Tier A: Parent-Child GMCs Physically Linked to H-II Nebulae
        #    Spiral Arm -> Dust Shock -> Molecular Cloud Complex -> H-II / Stars
        # ------------------------------------------------------------------
        if nebula_mgr is not None and hasattr(nebula_mgr, "get_nebulae_metadata"):
            for meta in nebula_mgr.get_nebulae_metadata():
                nx_p, ny_p, nz_p = meta["x"], meta["y"], meta["z"]
                nr = meta["radius"]
                th_p = meta["theta"]

                if rng.random() < 0.32:
                    perp_shift = -rng.uniform(0.08, 0.25) * max(1.0, nr * 1.4)
                else:
                    perp_shift = -rng.uniform(0.22, 0.52) * max(1.0, nr * 1.45)
                along_shift = rng.normal(0.0, 0.28 * max(0.6, nr))

                tang_yaw = spiral_tangent_yaw(th_p)
                tx, tz = math.cos(tang_yaw), math.sin(tang_yaw)
                px_u, pz_u = -tz, tx

                gmc_x = nx_p + px_u * perp_shift + tx * along_shift
                gmc_z = nz_p + pz_u * perp_shift + tz * along_shift
                gmc_y = ny_p + float(rng.normal(0.0, 0.025))

                base_len = float(np.clip(nr * rng.uniform(1.8, 2.8), 0.62, 2.05))
                base_wid = float(base_len * rng.uniform(0.38, 0.62))
                base_h = float(rng.uniform(0.095, 0.160))  # resolves cleanly on dy=0.075 grid
                str_parent = float(rng.uniform(1.05, 1.60))

                self.cloud_catalog.append((gmc_x, gmc_y, gmc_z, base_len, "nebula_parent"))

                num_sub = 3 if meta["size_tier"] == "hero" else 2
                for k in range(num_sub):
                    frac = (k - 0.5 * (num_sub - 1)) / max(num_sub - 1, 1)
                    sx = gmc_x + tx * frac * base_len * 0.72 + rng.normal(0.0, base_wid * 0.35)
                    sz = gmc_z + tz * frac * base_len * 0.72 + rng.normal(0.0, base_wid * 0.35)
                    sy = gmc_y + float(rng.normal(0.0, 0.022))
                    self._deposit_ellipsoid_cloud(
                        vol_env, vol_core, x_coords, y_coords, z_coords,
                        sx, sy, sz,
                        base_len * rng.uniform(0.55, 0.85),
                        base_h * rng.uniform(0.85, 1.20),
                        base_wid * rng.uniform(0.58, 0.90),
                        tang_yaw + rng.normal(0.0, 0.22),
                        rng.normal(0.0, 0.06),
                        str_parent * rng.uniform(0.75, 1.15),
                        core_frac=rng.uniform(0.85, 1.25), rng=rng
                    )

        # ------------------------------------------------------------------
        # 2. Tier B: Quiescent Dark Spiral-Arm & Bar GMCs (non-glowing absorbers)
        # ------------------------------------------------------------------
        n_arm_gmcs = 195
        for _ in range(n_arm_gmcs):
            arm_i = int(rng.choice([0, 1, 2, 3], p=[0.31, 0.20, 0.30, 0.19]))
            is_maj = (arm_i % 2 == 0)
            r_min = 5.6 if is_maj else 9.6
            r = float(np.clip(rng.gamma(shape=3.2, scale=5.2) + r_min - 1.5, r_min, 37.5))

            arm_base = arm_i * (2.0 * math.pi / self.ARM_COUNT)
            ideal_th = arm_base + self.ARM_BASE_OFFSET + self.SPIRAL_TIGHTNESS * math.log(max(r, self.INNER_ARM_R) / self.INNER_ARM_R)

            perp = float(np.clip(rng.normal(-0.22, 0.18), -0.58, 0.40))
            th = ideal_th + perp / r
            cx = r * math.cos(th)
            cz = r * math.sin(th)
            cy = float(rng.normal(0.0, self.H_MOL_CLOUD * 0.72))

            roll = rng.random()
            if roll < 0.24:  # Giant Molecular Association (~600-1400 pc envelope, broken by 3D noise)
                rx = float(rng.uniform(1.05, 1.85))
                rz = float(rx * rng.uniform(0.35, 0.54))
                ry = float(rng.uniform(0.105, 0.155))
                str_gmc = float(rng.uniform(1.10, 1.65))
                n_clumps = 3
            elif roll < 0.62:  # Medium GMC complex
                rx = float(rng.uniform(0.60, 1.08))
                rz = float(rx * rng.uniform(0.38, 0.62))
                ry = float(rng.uniform(0.090, 0.135))
                str_gmc = float(rng.uniform(0.85, 1.35))
                n_clumps = 2
            else:  # Compact dark cloud complex
                rx = float(rng.uniform(0.38, 0.64))
                rz = float(rx * rng.uniform(0.45, 0.75))
                ry = float(rng.uniform(0.085, 0.120))
                str_gmc = float(rng.uniform(0.75, 1.15))
                n_clumps = 2

            yaw = spiral_tangent_yaw(th) + rng.normal(0.0, 0.22)
            tx, tz = math.cos(yaw), math.sin(yaw)
            self.cloud_catalog.append((cx, cy, cz, rx, "arm_gmc"))

            for k in range(n_clumps):
                frac = (k - 0.5 * (n_clumps - 1)) / max(n_clumps - 1, 1)
                sx = cx + tx * frac * rx * 0.75 + rng.normal(0.0, rz * 0.40)
                sz = cz + tz * frac * rx * 0.75 + rng.normal(0.0, rz * 0.40)
                sy = cy + float(rng.normal(0.0, 0.02))
                self._deposit_ellipsoid_cloud(
                    vol_env, vol_core, x_coords, y_coords, z_coords,
                    sx, sy, sz,
                    rx * rng.uniform(0.58, 0.88),
                    ry * rng.uniform(0.85, 1.15),
                    rz * rng.uniform(0.60, 0.92),
                    yaw + rng.normal(0.0, 0.25),
                    rng.normal(0.0, 0.08),
                    str_gmc * rng.uniform(0.75, 1.15),
                    core_frac=rng.uniform(0.75, 1.15), rng=rng
                )

        # Curved Bar Leading-Edge & Bar-Tip Starburst GMCs (~18 complexes)
        sin_b, cos_b = math.sin(self.BAR_ANGLE), math.cos(self.BAR_ANGLE)
        for _ in range(18):
            sign_u = float(rng.choice([-1.0, 1.0]))
            u_b = sign_u * float(rng.uniform(2.0, 6.7))
            u_abs = abs(u_b)
            tip_bend = sign_u * (max(0.0, (u_abs - 4.2) / 3.0) ** 1.5) * 0.95
            v_b = sign_u * (0.28 + 0.20 * min(1.0, max(0.0, (u_abs - 1.6) / 3.6))) + tip_bend + float(rng.normal(0.0, 0.14))
            cx = u_b * sin_b - v_b * cos_b
            cz = u_b * cos_b + v_b * sin_b
            cy = float(rng.normal(0.0, 0.04))
            rx = float(rng.uniform(0.45, 0.85))
            rz = float(rx * rng.uniform(0.38, 0.58))
            ry = float(rng.uniform(0.085, 0.125))
            yaw = math.atan2(cos_b, sin_b) + float(rng.normal(0.0, 0.25))
            self.cloud_catalog.append((cx, cy, cz, rx, "bar_gmc"))
            for k in range(2):
                sx = cx + rng.normal(0.0, rx * 0.30)
                sz = cz + rng.normal(0.0, rx * 0.30)
                self._deposit_ellipsoid_cloud(
                    vol_env, vol_core, x_coords, y_coords, z_coords,
                    sx, cy + rng.normal(0.0, 0.02), sz,
                    rx * 0.72, ry, rz * 0.75,
                    yaw, 0.0, float(rng.uniform(0.75, 1.10)), 0.85, rng
                )

        # ------------------------------------------------------------------
        # 3. Tier C: Small Diffuse Spur & Outer/Inter-Arm Cloudlets (~95 small clouds)
        # ------------------------------------------------------------------
        for _ in range(95):
            arm_i = int(rng.choice([0, 1, 2, 3]))
            r = float(np.clip(rng.uniform(9.0, 35.0), 9.0, 35.0))
            arm_base = arm_i * (2.0 * math.pi / self.ARM_COUNT)
            ideal_th = arm_base + self.ARM_BASE_OFFSET + self.SPIRAL_TIGHTNESS * math.log(r / self.INNER_ARM_R)
            spur_perp = float(rng.uniform(-0.45, 1.65))
            th = ideal_th + spur_perp / r + float(rng.normal(0.0, 0.06))
            cx = r * math.cos(th)
            cz = r * math.sin(th)
            cy = float(rng.normal(0.0, 0.075))
            rx = float(rng.uniform(0.42, 0.80))
            rz = float(rx * rng.uniform(0.45, 0.75))
            ry = float(rng.uniform(0.085, 0.125))
            yaw = th + math.atan2(1.35, 1.0) + float(rng.normal(0.0, 0.3))
            self.cloud_catalog.append((cx, cy, cz, rx, "spur_cloudlet"))
            self._deposit_ellipsoid_cloud(
                vol_env, vol_core, x_coords, y_coords, z_coords,
                cx, cy, cz, rx, ry, rz, yaw, 0.0,
                float(rng.uniform(0.55, 0.95)), core_frac=0.65, rng=rng
            )

        # ------------------------------------------------------------------
        # 4. Subtle Local Nebula / Young Cluster Radiation Field (Illuminating Nearby Dust)
        #    Multiplied by physical local dust density in shader -> zero empty-space glow
        # ------------------------------------------------------------------
        if nebula_mgr is not None and hasattr(nebula_mgr, "nebulae_data"):
            for n in nebula_mgr.nebulae_data:
                nx_p, ny_p, nz_p, nr = n["position"]
                cr, cg, cb, strength = n["color"]
                rad_glow = max(nr * 2.5, 0.60)
                rad_y = max(nr * 0.95, 0.25)

                ix0 = max(0, int((nx_p - rad_glow + hs) / dx))
                ix1 = min(nx, int((nx_p + rad_glow + hs) / dx) + 2)
                iz0 = max(0, int((nz_p - rad_glow + hs) / dz))
                iz1 = min(nz, int((nz_p + rad_glow + hs) / dz) + 2)
                iy0 = max(0, int((ny_p - rad_y + yh) / dy))
                iy1 = min(ny, int((ny_p + rad_y + yh) / dy) + 2)

                if ix1 <= ix0 or iz1 <= iz0 or iy1 <= iy0:
                    continue

                gx = x_coords[ix0:ix1][None, None, :] - nx_p
                gy = y_coords[iy0:iy1][:, None, None] - ny_p
                gz = z_coords[iz0:iz1][None, :, None] - nz_p

                d2_xz = (gx * gx + gz * gz) / (rad_glow * rad_glow)
                d2_y = (gy * gy) / (rad_y * rad_y)
                glow = np.exp(-2.1 * (d2_xz + d2_y)).astype(np.float32) * float(strength) * 0.48

                is_blue = cb > cr * 1.05
                warm_w = 0.22 if is_blue else 1.0
                cool_w = 1.0 if is_blue else (0.15 + 0.35 * max(0.0, cb - cg))
                illum_warm[iy0:iy1, iz0:iz1, ix0:ix1] += glow * warm_w
                illum_cool[iy0:iy1, iz0:iz1, ix0:ix1] += glow * cool_w

        vol_env = np.clip(vol_env, 0.0, 2.0)
        vol_core = np.clip(vol_core, 0.0, 2.0)
        illum_warm = np.clip(illum_warm, 0.0, 2.5)
        illum_cool = np.clip(illum_cool, 0.0, 2.5)

        return np.stack([vol_env, vol_core, illum_warm, illum_cool], axis=-1).astype(np.float32)

    def bind_textures(self, program: moderngl.Program, base_unit: int = 0):
        """Bind ISM 2D spiral map and 3D molecular cloud / noise volumes to shader uniforms."""
        if "u_ismSpiralMap" in program:
            self.spiral_map_tex.use(location=base_unit)
            program["u_ismSpiralMap"].value = base_unit
        if "u_molCloudTex" in program:
            self.mol_cloud_tex.use(location=base_unit + 1)
            program["u_molCloudTex"].value = base_unit + 1
        if "u_noiseVol3D" in program:
            self.noise_vol_tex.use(location=base_unit + 2)
            program["u_noiseVol3D"].value = base_unit + 2
        if "u_hasISMTextures" in program:
            program["u_hasISMTextures"].value = 1

    def release(self):
        self.noise_vol_tex.release()
        self.spiral_map_tex.release()
        self.mol_cloud_tex.release()
