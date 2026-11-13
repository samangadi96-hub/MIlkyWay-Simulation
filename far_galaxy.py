import math
from pathlib import Path
import numpy as np
import moderngl
import glm


class FarGalaxy:
    """
    Physically Stratified 3D Volumetric Level-of-Detail (LOD) Galaxy Representation.
    
    Constructed directly from the 3D stellar populations:
    - 2.85 million stars from GalacticBulge, GalacticBar, and GalacticDisk
    - Warm golden Population II core and elongated central bar (28 deg tilt)
    - Boxy/peanut central bulge profile extending vertically (|y| <= 1.8 units)
    - Thin stellar disk (young OB arm stars with scale height ~280 pc)
    - Thick stellar disk (diffuse older population with scale height ~800 pc)
    - Realistic outer-disk flaring and subtle radial warp
    - Separate ultra-thin dust layer (scale height ~100 pc) with wavelength-dependent CCM extinction
    - Localized H-II star-forming complexes strictly derived from shared NebulaManager catalogue
    
    Rendered via volume raymarching through a 3D bounding slab on the galactic plane,
    providing completely seamless, continuous, artifact-free views from face-on,
    oblique, and edge-on orientations.
    """

    def __init__(self, ctx, base_dir, nebula_mgr=None, disk=None, bar=None, bulge=None, res=512):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.res = res
        self.num_slices = 20
        self.half_size = 48.0
        self.y_max = 3.2 # Vertical half-extent (-3.2 to +3.2 units ~ 4.5 kpc)

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
        # 1. Precompute 3D Multichannel Galaxy Representation
        # ------------------------------------------------------------------
        stellar_data, dust_neb_data = self._generate_far_textures(nebula_mgr, disk, bar, bulge)

        # ModernGL 3D texture size convention: (width, height, depth) = (res, res, num_slices)
        self.stellar_tex = self.ctx.texture3d(
            (self.res, self.res, self.num_slices), 4,
            data=stellar_data.tobytes(), dtype='f2'
        )
        self.stellar_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.stellar_tex.repeat_x = False
        self.stellar_tex.repeat_y = False
        self.stellar_tex.repeat_z = False

        self.dust_nebula_tex = self.ctx.texture3d(
            (self.res, self.res, self.num_slices), 4,
            data=dust_neb_data.tobytes(), dtype='f2'
        )
        self.dust_nebula_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.dust_nebula_tex.repeat_x = False
        self.dust_nebula_tex.repeat_y = False
        self.dust_nebula_tex.repeat_z = False

        # ------------------------------------------------------------------
        # 2. Build Bounding Box Proxy Geometry (12 Triangles / 36 Vertices)
        # ------------------------------------------------------------------
        hs = self.half_size
        ym = self.y_max

        cube_verts = np.array([
            # Back face (Z = -hs)
            -hs, -ym, -hs,   hs,  ym, -hs,   hs, -ym, -hs,
            -hs, -ym, -hs,  -hs,  ym, -hs,   hs,  ym, -hs,
            # Front face (Z = +hs)
            -hs, -ym,  hs,   hs, -ym,  hs,   hs,  ym,  hs,
            -hs, -ym,  hs,   hs,  ym,  hs,  -hs,  ym,  hs,
            # Left face (X = -hs)
            -hs,  ym,  hs,  -hs,  ym, -hs,  -hs, -ym, -hs,
            -hs,  ym,  hs,  -hs, -ym, -hs,  -hs, -ym,  hs,
            # Right face (X = +hs)
             hs,  ym,  hs,   hs, -ym, -hs,   hs,  ym, -hs,
             hs,  ym,  hs,   hs, -ym,  hs,   hs, -ym, -hs,
            # Bottom face (Y = -ym)
            -hs, -ym, -hs,   hs, -ym, -hs,   hs, -ym,  hs,
            -hs, -ym, -hs,   hs, -ym,  hs,  -hs, -ym,  hs,
            # Top face (Y = +ym)
            -hs,  ym, -hs,   hs,  ym,  hs,   hs,  ym, -hs,
            -hs,  ym, -hs,  -hs,  ym,  hs,   hs,  ym,  hs,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(cube_verts.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')]
        )

        # Depth Midplane Quad (Y = 0.0) for depth buffer rasterization
        depth_verts = np.array([
            -hs, 0.0, -hs,
             hs, 0.0, -hs,
            -hs, 0.0,  hs,
            -hs, 0.0,  hs,
             hs, 0.0, -hs,
             hs, 0.0,  hs,
        ], dtype='f4')
        self.depth_vbo = self.ctx.buffer(depth_verts.tobytes())
        self.depth_vao = self.ctx.vertex_array(
            self.program,
            [(self.depth_vbo, '3f', 'in_position')]
        )

    def _generate_far_textures(self, nebula_mgr, disk=None, bar=None, bulge=None):
        """Precomputes genuine 3D stellar radiance and dust/nebula textures from galaxy stars."""
        res = self.res
        hs = self.half_size
        num_slices = self.num_slices
        y_max = self.y_max
        delta_y = (2.0 * y_max) / num_slices
        slice_y = np.array([-y_max + (k + 0.5) * delta_y for k in range(num_slices)], dtype=np.float32)

        x_coords = np.linspace(-hs, hs, res, dtype=np.float32)
        z_coords = np.linspace(-hs, hs, res, dtype=np.float32)
        X, Z = np.meshgrid(x_coords, z_coords)
        R = np.sqrt(X * X + Z * Z)

        def deposit_stars_3d(x, y, z, rgb, lum):
            """Conserves total stellar energy via continuous linear interpolation between vertical slices."""
            s = (y + y_max) / delta_y - 0.5
            s_clamped = np.clip(s, 0.0, num_slices - 1.0001)
            k0 = np.floor(s_clamped).astype(np.int32)
            f = (s_clamped - k0).astype(np.float32)
            k1 = np.minimum(k0 + 1, num_slices - 1)

            ix = np.clip(((x + hs) / (2.0 * hs) * res).astype(np.int32), 0, res - 1)
            iz = np.clip(((z + hs) / (2.0 * hs) * res).astype(np.int32), 0, res - 1)
            pixel_idx = iz * res + ix

            vol_r = np.zeros((num_slices, res * res), dtype=np.float32)
            vol_g = np.zeros((num_slices, res * res), dtype=np.float32)
            vol_b = np.zeros((num_slices, res * res), dtype=np.float32)

            w0 = 1.0 - f
            w1 = f
            r_lum = (rgb[:, 0] * lum).astype(np.float32)
            g_lum = (rgb[:, 1] * lum).astype(np.float32)
            b_lum = (rgb[:, 2] * lum).astype(np.float32)

            np.add.at(vol_r, (k0, pixel_idx), r_lum * w0)
            np.add.at(vol_r, (k1, pixel_idx), r_lum * w1)
            np.add.at(vol_g, (k0, pixel_idx), g_lum * w0)
            np.add.at(vol_g, (k1, pixel_idx), g_lum * w1)
            np.add.at(vol_b, (k0, pixel_idx), b_lum * w0)
            np.add.at(vol_b, (k1, pixel_idx), b_lum * w1)

            return (
                vol_r.reshape(num_slices, res, res),
                vol_g.reshape(num_slices, res, res),
                vol_b.reshape(num_slices, res, res)
            )

        def fft_blur_3d(arr3d, sigma):
            """Vectorized 2D FFT Gaussian filter operating simultaneously across all 20 vertical slices."""
            D, H, W = arr3d.shape
            ky = np.fft.fftfreq(H)[:, None]
            kx = np.fft.rfftfreq(W)[None, :]
            filt = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (ky ** 2 + kx ** 2)).astype(np.float32)
            f_arr = np.fft.rfft2(arr3d)
            return np.fft.irfft2(f_arr * filt).astype(np.float32)

        def smoothstep_np(e0, e1, x):
            t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # ----------------------------------------------------------
        # 1. Read 3D Stellar Data from VBO Buffers
        # ----------------------------------------------------------
        if disk is not None and hasattr(disk, 'vbo') and bar is not None and bulge is not None:
            d_disk = np.frombuffer(disk.vbo.read(), dtype=np.float32).copy().reshape(-1, 8)
            d_bar = np.frombuffer(bar.vbo.read(), dtype=np.float32).copy().reshape(-1, 8)
            d_bulge = np.frombuffer(bulge.vbo.read(), dtype=np.float32).copy().reshape(-1, 8)
        else:
            from galaxy import GalacticBulge, GalacticBar
            from galactic_disk import GalacticDisk
            _b = GalacticBulge(self.ctx, self.base_dir)
            _bar = GalacticBar(self.ctx, self.base_dir)
            _d = GalacticDisk(self.ctx, self.base_dir)
            d_bulge = np.frombuffer(_b.vbo.read(), dtype=np.float32).copy().reshape(-1, 8)
            d_bar = np.frombuffer(_bar.vbo.read(), dtype=np.float32).copy().reshape(-1, 8)
            d_disk = np.frombuffer(_d.vbo.read(), dtype=np.float32).copy().reshape(-1, 8)
            for _obj in [_b, _bar, _d]:
                if hasattr(_obj, 'vbo'): _obj.vbo.release()
                if hasattr(_obj, 'vao'): _obj.vao.release()
                if hasattr(_obj, 'program'): _obj.program.release()

        # Separate disk into arm and diffuse components
        n_arm = getattr(disk, "n_arm", int(len(d_disk) * 0.48))
        x_arm, y_arm, z_arm = d_disk[:n_arm, 0], d_disk[:n_arm, 1], d_disk[:n_arm, 2]
        rgb_arm, lum_arm = d_disk[:n_arm, 3:6], d_disk[:n_arm, 7]
        r_arm_stars = np.sqrt(x_arm * x_arm + z_arm * z_arm)
        th_arm_stars = np.arctan2(z_arm, x_arm)

        x_diff, y_diff, z_diff = d_disk[n_arm:, 0], d_disk[n_arm:, 1], d_disk[n_arm:, 2]
        rgb_diff, lum_diff = d_disk[n_arm:, 3:6], d_disk[n_arm:, 7]

        # Spiral Arm Geometry & Gating (matching GalacticDisk Reference A)
        ARM_COUNT = 4
        SPIRAL_TIGHTNESS = 2.5
        INNER_ARM_R = 3.5
        BAR_ANGLE = 0.488
        BAR_ANGLE_POLAR = math.pi / 2.0 - BAR_ANGLE
        ARM_BASE_OFFSET = BAR_ANGLE_POLAR - SPIRAL_TIGHTNESS * math.log(7.0 / INNER_ARM_R)

        # Gentle broad inter-bar resonance cavity along bar minor axis (r < 9.5):
        # Prevents diffuse inner-disk haze from washing out the realistic bar while
        # transitioning smoothly into the inner spiral arms without dark pinch artifacts.
        r_diff = np.sqrt(x_diff * x_diff + z_diff * z_diff)
        th_diff = np.arctan2(z_diff, x_diff)
        minor_prox = np.abs(np.sin(th_diff - BAR_ANGLE_POLAR))
        cavity_att = np.where(
            r_diff < 9.5,
            1.0 - (1.0 - smoothstep_np(5.0, 9.5, r_diff)) * (0.52 * minor_prox),
            1.0,
        )
        lum_diff_cav = lum_diff * cavity_att

        r_clamped = np.maximum(r_arm_stars, INNER_ARM_R)
        arm_spine = ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * np.log(r_clamped / INNER_ARM_R)
        rel_theta = (th_arm_stars - arm_spine) % (2.0 * math.pi)
        arm_idx = np.floor(rel_theta / (2.0 * math.pi / ARM_COUNT) + 0.5).astype(int) % ARM_COUNT

        gate_major = smoothstep_np(6.0, 7.8, r_arm_stars)
        gate_minor = smoothstep_np(9.2, 11.5, r_arm_stars)
        is_major = (arm_idx == 0) | (arm_idx == 2)
        arm_gate = np.where(is_major, gate_major, gate_minor)
        lum_arm_gated = lum_arm * arm_gate

        # Deposit 3D Stellar Components
        arm_vr, arm_vg, arm_vb = deposit_stars_3d(x_arm, y_arm, z_arm, rgb_arm, lum_arm_gated)
        diff_vr, diff_vg, diff_vb = deposit_stars_3d(x_diff, y_diff, z_diff, rgb_diff, lum_diff_cav)
        bar_vr, bar_vg, bar_vb = deposit_stars_3d(d_bar[:, 0], d_bar[:, 1], d_bar[:, 2], d_bar[:, 3:6], d_bar[:, 7])
        bulge_vr, bulge_vg, bulge_vb = deposit_stars_3d(d_bulge[:, 0], d_bulge[:, 1], d_bulge[:, 2], d_bulge[:, 3:6], d_bulge[:, 7])

        # ----------------------------------------------------------
        # 2. Multi-Scale 3D Filtering
        # ----------------------------------------------------------
        # 1. Arm stars: granular clusters (sigma=1.0) + arm spine (sigma=2.8) + diffuse envelope (sigma=6.0)
        arm_s_r = fft_blur_3d(arm_vr, 1.0)
        arm_s_g = fft_blur_3d(arm_vg, 1.0)
        arm_s_b = fft_blur_3d(arm_vb, 1.0)
        arm_m_r = fft_blur_3d(arm_vr, 2.8)
        arm_m_g = fft_blur_3d(arm_vg, 2.8)
        arm_m_b = fft_blur_3d(arm_vb, 2.8)
        arm_w_r = fft_blur_3d(arm_vr, 6.0)
        arm_w_g = fft_blur_3d(arm_vg, 6.0)
        arm_w_b = fft_blur_3d(arm_vb, 6.0)

        # Keep the outskirts readable without increasing their total light:
        # trade part of the broad envelope for the fine arm component, then
        # add a deterministic, low-amplitude stellar-cluster modulation.
        outer_detail = smoothstep_np(14.0, 32.0, R) * (1.0 - smoothstep_np(42.0, 47.0, R))
        theta = np.arctan2(Z, X)
        outer_grain = np.sin(0.85 * R + 9.0 * theta) * np.sin(1.75 * R - 5.0 * theta)
        outer_cluster_detail = 1.0 + outer_detail * outer_grain * 0.24
        sharp_weight = 0.40 + outer_detail * 0.20
        medium_weight = 0.42 - outer_detail * 0.10
        wide_weight = 0.18 - outer_detail * 0.10

        arm_r = (arm_s_r * sharp_weight + arm_m_r * medium_weight + arm_w_r * wide_weight) * 0.42
        arm_g = (arm_s_g * sharp_weight + arm_m_g * medium_weight + arm_w_g * wide_weight) * 0.42
        arm_b = (arm_s_b * sharp_weight + arm_m_b * medium_weight + arm_w_b * wide_weight) * 0.42
        arm_r *= outer_cluster_detail[None, :, :]
        arm_g *= outer_cluster_detail[None, :, :]
        arm_b *= outer_cluster_detail[None, :, :]

        # Signature celestial blue-white hue along the spiral arms.  Fade the
        # outer, young-star component strongly so its blue glow does not
        # overpower the rest of the galaxy at far LOD.
        arm_trans = smoothstep_np(6.0, 9.5, R)
        arm_tint_r = (1.0 - arm_trans) * 1.15 + arm_trans * 0.68
        arm_tint_g = (1.0 - arm_trans) * 1.05 + arm_trans * 0.85
        arm_tint_b = (1.0 - arm_trans) * 0.75 + arm_trans * 1.48
        outer_blue_intensity = 1.0 - arm_trans * 0.85
        arm_r *= arm_tint_r[None, :, :] * outer_blue_intensity[None, :, :]
        arm_g *= arm_tint_g[None, :, :] * outer_blue_intensity[None, :, :]
        arm_b *= arm_tint_b[None, :, :] * outer_blue_intensity[None, :, :]

        # 2. Diffuse disk: captures thin & thick disk profile and radial flare
        diff_r = fft_blur_3d(diff_vr, 5.0) * 0.22
        diff_g = fft_blur_3d(diff_vg, 5.0) * 0.22
        diff_b = fft_blur_3d(diff_vb, 5.0) * 0.26

        # 3. Bar: Balanced multi-scale spine + soft body + gentle core-to-tip gradient
        # Avoids flat highlight clipping/neon-yellow blowout at maximum zoom-out while
        # preserving authentic peanut lobes (|y| ~ 0.8 - 1.4) and inner-arm connection.
        bar_s_r = fft_blur_3d(bar_vr, 1.5)
        bar_s_g = fft_blur_3d(bar_vg, 1.5)
        bar_s_b = fft_blur_3d(bar_vb, 1.5)
        bar_m_r = fft_blur_3d(bar_vr, 3.0)
        bar_m_g = fft_blur_3d(bar_vg, 3.0)
        bar_m_b = fft_blur_3d(bar_vb, 3.0)
        bar_w_r = fft_blur_3d(bar_vr, 5.2)
        bar_w_g = fft_blur_3d(bar_vg, 5.2)
        bar_w_b = fft_blur_3d(bar_vb, 5.2)
        bar_raw_r = bar_s_r * 0.32 + bar_m_r * 0.38 + bar_w_r * 0.30
        bar_raw_g = bar_s_g * 0.32 + bar_m_g * 0.38 + bar_w_g * 0.30
        bar_raw_b = bar_s_b * 0.32 + bar_m_b * 0.38 + bar_w_b * 0.30

        bar_lum = np.sqrt(bar_raw_r**2 + bar_raw_g**2 + bar_raw_b**2)
        bar_scale = 1.0 / (1.0 + bar_lum * 0.005)
        bar_r = bar_raw_r * bar_scale * 0.100 * 1.26
        bar_g = bar_raw_g * bar_scale * 0.100 * 1.12
        bar_b = bar_raw_b * bar_scale * 0.100 * 0.80

        # 4. Central Nuclear Bulge: warm spheroidal core peaking softly around Sgr A*
        bulge_s_r = fft_blur_3d(bulge_vr, 1.1)
        bulge_s_g = fft_blur_3d(bulge_vg, 1.1)
        bulge_s_b = fft_blur_3d(bulge_vb, 1.1)
        bulge_m_r = fft_blur_3d(bulge_vr, 2.8)
        bulge_m_g = fft_blur_3d(bulge_vg, 2.8)
        bulge_m_b = fft_blur_3d(bulge_vb, 2.8)
        bulge_raw_r = bulge_s_r * 0.65 + bulge_m_r * 0.35
        bulge_raw_g = bulge_s_g * 0.65 + bulge_m_g * 0.35
        bulge_raw_b = bulge_s_b * 0.65 + bulge_m_b * 0.35
        bulge_lum = np.sqrt(bulge_raw_r**2 + bulge_raw_g**2 + bulge_raw_b**2)
        bulge_scale = 1.0 / (1.0 + bulge_lum * 0.15)
        bulge_r = bulge_raw_r * bulge_scale * 0.74 * 1.36
        bulge_g = bulge_raw_g * bulge_scale * 0.74 * 1.22
        bulge_b = bulge_raw_b * bulge_scale * 0.74 * 0.90

        # Combine stellar components
        stars_r = arm_r + diff_r + bar_r + bulge_r
        stars_g = arm_g + diff_g + bar_g + bulge_g
        stars_b = arm_b + diff_b + bar_b + bulge_b

        # Outer disk falloff: smooth exponential decay starting at r=35
        outer_mask = np.where(R <= 35.0, 1.0, np.exp(-0.5 * ((R - 35.0) / 4.5)**2)).astype(np.float32)
        stars_r *= outer_mask[None, :, :]
        stars_g *= outer_mask[None, :, :]
        stars_b *= outer_mask[None, :, :]

        # Calibrate total column brightness anchored to baseline disk/arm exposure
        # (prevents lower central bar brightness from inflating spiral arm normalization)
        col_lum = np.sqrt(np.sum(stars_r**2 + stars_g**2 + stars_b**2, axis=0))
        target_peak = 8.5
        ref_peak = max(float(col_lum.max()), 133.261)
        col_scale = (target_peak / (ref_peak + 1e-6)) * (1.0 / (1.0 + col_lum * 0.04)) * 3.6

        stars_r *= col_scale[None, :, :]
        stars_g *= col_scale[None, :, :]
        stars_b *= col_scale[None, :, :]

        stellar_density = np.sqrt(stars_r**2 + stars_g**2 + stars_b**2)
        alpha_map = np.clip(stellar_density * 0.40, 0.0, 1.0) * outer_mask[None, :, :]

        stellar_img = np.zeros((num_slices, res, res, 4), dtype=np.float32)
        stellar_img[:, :, :, 0] = stars_r
        stellar_img[:, :, :, 1] = stars_g
        stellar_img[:, :, :, 2] = stars_b
        stellar_img[:, :, :, 3] = alpha_map

        # ----------------------------------------------------------
        # 3. Dust Map Generation (Filamentary lanes along inner arms & bar leading edge)
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
            w_arm_d = 1.0 if is_maj else 0.60
            arm_base = float(i) * (2.0 * math.pi / ARM_COUNT)
            ideal_theta = arm_base + ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * np.log(r_arm_w / INNER_ARM_R)
            d_theta = (Theta_warped - ideal_theta + math.pi) % (2.0 * math.pi) - math.pi
            d_arm = d_theta - (-0.14)
            linear_dist = np.abs(d_arm) * R_warped
            lane_w = (0.040 + 0.030 * (R / 45.0)) * R
            core_lane = np.exp(-0.5 * (linear_dist / lane_w)**2) * w_arm_d * 1.25
            d_arm_sub = d_theta - (-0.02)
            sub_lane = np.exp(-0.5 * (np.abs(d_arm_sub) * R_warped / (lane_w * 1.5))**2) * w_arm_d * 0.35
            s_norm = (R - INNER_ARM_R) / 41.5
            wisp_mod = 0.85 + 0.35 * np.sin(2.0 * math.pi * 5.5 * s_norm + i * 1.57) + 0.25 * noise_dust
            lane_combined = (core_lane + sub_lane) * np.clip(wisp_mod, 0.20, 1.80)
            arm_dust_total += lane_combined * smoothstep_np(r_start - 0.5, r_start + 1.2, R) * (R <= 42.0)

        # Curved leading-edge bar dust shocks (domain-warped & smoothly tapered, matching ISM)
        sin_b = math.sin(BAR_ANGLE)
        cos_b = math.cos(BAR_ANGLE)
        u_bar = X_warped * sin_b + Z_warped * cos_b
        v_bar = -X_warped * cos_b + Z_warped * sin_b
        u_abs = np.abs(u_bar)
        tip_bend = np.sign(u_bar) * smoothstep_np(4.2, 7.2, u_abs) * 0.45 * np.maximum(0.0, u_abs - 4.2)
        v_shock = np.sign(u_bar) * (0.36 + 0.18 * smoothstep_np(1.6, 5.2, u_abs)) + tip_bend
        w_shock = 0.40 + 0.20 * smoothstep_np(1.5, 6.5, u_abs)
        bar_env_u = smoothstep_np(1.8, 3.2, u_abs) * (1.0 - smoothstep_np(6.0, 7.5, u_abs))
        bar_mod = np.clip(0.70 + 0.45 * noise_dust, 0.25, 1.40)
        bar_dust = np.exp(-0.5 * ((v_bar - v_shock) / w_shock)**2) * bar_env_u * bar_mod * 0.36

        tau_col = (arm_dust_total + bar_dust) * outer_mask

        # Vertically stratify dust: confined strictly to thin midplane (scale height ~ 0.28 units ~ 200 pc)
        dust_neb_img = np.zeros((num_slices, res, res, 4), dtype=np.float32)
        for k in range(num_slices):
            yk = slice_y[k]
            phi_dust = math.exp(-0.5 * (yk / 0.28)**2)
            dust_neb_img[k, :, :, 0] = tau_col * phi_dust

        # ----------------------------------------------------------
        # 4. Nebulae Map (Derived from Shared NebulaManager Catalogue)
        # ----------------------------------------------------------
        if nebula_mgr is not None and hasattr(nebula_mgr, 'nebulae_data'):
            for n in nebula_mgr.nebulae_data:
                nx, ny, nz, nr = n['position']
                ncol = np.array(n['color'][:3], dtype=np.float32) * n['color'][3]

                ix0 = max(0, int((nx - 3.0 * nr + hs) / (2.0 * hs) * res))
                ix1 = min(res, int((nx + 3.0 * nr + hs) / (2.0 * hs) * res) + 1)
                iz0 = max(0, int((nz - 3.0 * nr + hs) / (2.0 * hs) * res))
                iz1 = min(res, int((nz + 3.0 * nr + hs) / (2.0 * hs) * res) + 1)

                if ix1 > ix0 and iz1 > iz0:
                    subX = X[iz0:iz1, ix0:ix1]
                    subZ = Z[iz0:iz1, ix0:ix1]
                    d2 = (subX - nx)**2 + (subZ - nz)**2
                    mask_2d = np.exp(-0.5 * d2 / (nr * 0.35)**2) * 1.6 + np.exp(-0.5 * d2 / (nr * 0.75)**2) * 0.40

                    # Confined vertically to thin star-forming midplane
                    sigma_y = max(0.08, 0.30 * nr)
                    wy = np.exp(-0.5 * ((slice_y - ny) / sigma_y)**2)
                    wy_sum = np.sum(wy)
                    if wy_sum > 1e-4:
                        wy /= wy_sum

                    for k in range(num_slices):
                        if wy[k] > 0.005:
                            for c in range(3):
                                dust_neb_img[k, iz0:iz1, ix0:ix1, c + 1] += mask_2d * (wy[k] * ncol[c])

                    # Child sub-knots for medium and hero complexes
                    if nr > 0.35:
                        num_children = 4 if nr > 0.65 else 2
                        rng_child = np.random.default_rng(int(abs(nx * 10000 + nz * 100)) % 100000)
                        for ch in range(num_children):
                            ch_r = rng_child.uniform(0.25, 0.60) * nr
                            ch_th = rng_child.uniform(0.0, 2.0 * math.pi)
                            cx = nx + ch_r * math.cos(ch_th)
                            cz = nz + ch_r * math.sin(ch_th)
                            cy = ny + rng_child.normal(0.0, 0.03)
                            c_size = rng_child.uniform(0.20, 0.32) * nr

                            cix0 = max(0, int((cx - 3.0 * c_size + hs) / (2.0 * hs) * res))
                            cix1 = min(res, int((cx + 3.0 * c_size + hs) / (2.0 * hs) * res) + 1)
                            ciz0 = max(0, int((cz - 3.0 * c_size + hs) / (2.0 * hs) * res))
                            ciz1 = min(res, int((cz + 3.0 * c_size + hs) / (2.0 * hs) * res) + 1)
                            if cix1 > cix0 and ciz1 > ciz0:
                                csubX = X[ciz0:ciz1, cix0:cix1]
                                csubZ = Z[ciz0:ciz1, cix0:cix1]
                                cd2 = (csubX - cx)**2 + (csubZ - cz)**2
                                cmask_2d = np.exp(-0.5 * cd2 / (c_size * 0.35)**2) * 1.8

                                is_blue = rng_child.random() < 0.28
                                ch_col = np.array([0.55, 0.88, 1.40], dtype=np.float32) * (n['color'][3] * 0.75) if is_blue else ncol * 0.70
                                cwy = np.exp(-0.5 * ((slice_y - cy) / 0.08)**2)
                                cwy /= (np.sum(cwy) + 1e-6)

                                for k in range(num_slices):
                                    if cwy[k] > 0.005:
                                        for c in range(3):
                                            dust_neb_img[k, ciz0:ciz1, cix0:cix1, c + 1] += cmask_2d * (cwy[k] * ch_col[c])

        # Bar tip starburst complexes (at bar ends u = +/- 6.6 where gas streams intersect inner arms)
        for tip_sign in [-1.0, 1.0]:
            tip_u = tip_sign * 6.6
            rng_tip = np.random.default_rng(int(abs(tip_sign * 4242)) % 100000)
            for _ in range(8):
                u_clump = tip_u + rng_tip.normal(0.0, 0.45)
                v_clump = tip_sign * 0.90 + rng_tip.normal(0.0, 0.30)
                y_clump = rng_tip.normal(0.0, 0.04)
                bx = float(u_clump * sin_b - v_clump * cos_b)
                bz = float(u_clump * cos_b + v_clump * sin_b)
                bs = float(rng_tip.uniform(0.18, 0.38))
                bint = float(rng_tip.uniform(0.8, 1.6))

                bix0 = max(0, int((bx - 3.0 * bs + hs) / (2.0 * hs) * res))
                bix1 = min(res, int((bx + 3.0 * bs + hs) / (2.0 * hs) * res) + 1)
                biz0 = max(0, int((bz - 3.0 * bs + hs) / (2.0 * hs) * res))
                biz1 = min(res, int((bz + 3.0 * bs + hs) / (2.0 * hs) * res) + 1)
                if bix1 > bix0 and biz1 > biz0:
                    bsubX = X[biz0:biz1, bix0:bix1]
                    bsubZ = Z[biz0:biz1, bix0:bix1]
                    bd2 = (bsubX - bx)**2 + (bsubZ - bz)**2
                    bmask_2d = np.exp(-0.5 * bd2 / (bs * 0.35)**2) * bint

                    is_blue = rng_tip.random() < 0.35
                    b_col = np.array([0.55, 0.88, 1.40], dtype=np.float32) if is_blue else np.array([1.25, 0.20, 0.50], dtype=np.float32)
                    bwy = np.exp(-0.5 * ((slice_y - y_clump) / 0.08)**2)
                    bwy /= (np.sum(bwy) + 1e-6)

                    for k in range(num_slices):
                        if bwy[k] > 0.005:
                            for c in range(3):
                                dust_neb_img[k, biz0:biz1, bix0:bix1, c + 1] += bmask_2d * (bwy[k] * b_col[c])

        return stellar_img.astype(np.float16), dust_neb_img.astype(np.float16)

    def render(self, camera, aspect_ratio, visibility, rotation_angle):
        """Renders the 3D volume representation on the galactic plane with perspective correctness."""
        if visibility <= 0.001:
            return

        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = 'front' # Draw back faces so rays enter correctly even if camera is inside
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE

        # Bind 3D volume textures
        self.stellar_tex.use(location=0)
        self.dust_nebula_tex.use(location=1)

        if "u_stellarMap" in self.program:
            self.program["u_stellarMap"].value = 0
        if "u_dustNebulaMap" in self.program:
            self.program["u_dustNebulaMap"].value = 1
        if "u_visibility" in self.program:
            self.program["u_visibility"].value = visibility
        if "u_layerWeight" in self.program:
            self.program["u_layerWeight"].value = 1.0

        hs = self.half_size
        ym = self.y_max
        if "u_boxMin" in self.program:
            self.program["u_boxMin"].value = (-hs, -ym, -hs)
        if "u_boxMax" in self.program:
            self.program["u_boxMax"].value = ( hs,  ym,  hs)
        if "u_camPos" in self.program:
            self.program["u_camPos"].value = camera.get_position()

        if "u_view" in self.program:
            self.program["u_view"].write(camera.get_view_matrix())
        if "u_projection" in self.program:
            self.program["u_projection"].write(camera.get_projection_matrix(aspect_ratio))

        # Rotate model and pass inverse model for object-space raymarching
        model = glm.rotate(glm.mat4(1.0), rotation_angle, glm.vec3(0.0, 1.0, 0.0))
        inv_model = glm.inverse(model)
        if "u_model" in self.program:
            self.program["u_model"].write(model)
        if "u_invModel" in self.program:
            self.program["u_invModel"].write(inv_model)

        # 12 triangles (36 vertices)
        self.vao.render(mode=moderngl.TRIANGLES, vertices=36)

        # Restore context state
        self.ctx.disable(moderngl.CULL_FACE)
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
        inv_model = glm.inverse(model)
        if "u_model" in self.program:
            self.program["u_model"].write(model)
        if "u_invModel" in self.program:
            self.program["u_invModel"].write(inv_model)
        if "u_visibility" in self.program:
            self.program["u_visibility"].value = 1.0
        if "u_layerWeight" in self.program:
            self.program["u_layerWeight"].value = 1.0

        hs = self.half_size
        ym = self.y_max
        if "u_boxMin" in self.program:
            self.program["u_boxMin"].value = (-hs, -ym, -hs)
        if "u_boxMax" in self.program:
            self.program["u_boxMax"].value = ( hs,  ym,  hs)
        if "u_camPos" in self.program:
            self.program["u_camPos"].value = camera.get_position()

        self.stellar_tex.use(location=0)
        self.dust_nebula_tex.use(location=1)
        self.depth_vao.render(mode=moderngl.TRIANGLES, vertices=6)

    def release(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.depth_vbo.release()
        self.depth_vao.release()
        self.stellar_tex.release()
        self.dust_nebula_tex.release()
        self.program.release()
