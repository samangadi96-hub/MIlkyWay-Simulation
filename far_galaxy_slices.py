import math
import numpy as np


# Stratified symmetric vertical intervals covering >99.97% of bulge, bar, and disk stars
# while resolving the thin star-forming midplane (|y| <= 0.07), dust layer, young arm disk,
# boxy/peanut bar lobes (|y| ~ 0.35 - 1.10), and extended thick disk out to |y| <= 2.60.
DEFAULT_Y_EDGES = np.array(
    [-2.60, -1.65, -1.10, -0.78, -0.54, -0.35, -0.19, -0.07,
      0.07,  0.19,  0.35,  0.54,  0.78,  1.10,  1.65,  2.60],
    dtype=np.float32,
)


def _smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _erf_np(x):
    """Vectorized error function using NumPy (avoids scipy dependency)."""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


class FarGalaxySliceBuilder:
    """
    Memory-efficient, CPU-side vertical slice accumulator and energy-preserving normalizer
    for the True 3D Far-LOD Galaxy representation.

    Design goals:
    - Zero GPU VBO readbacks (accumulates directly while CPU arrays exist during init).
    - Zero persistent copies of the 2.85M particle arrays.
    - Exact conservation of the reference 2D face-on integrated stellar, dust, and nebula maps.
    - Authentic vertical stellar distributions (thin young disk, thick old disk with flare,
      nuclear bulge, and 3D boxy/peanut galactic bar lobes) with zero cross-slice duplication.
    """

    ARM_COUNT = 4
    SPIRAL_TIGHTNESS = 2.5
    INNER_ARM_R = 3.5
    BAR_ANGLE = 0.488
    ARM_BASE_OFFSET = (math.pi / 2.0 - 0.488) - 2.5 * math.log(7.0 / 3.5)

    def __init__(self, res: int = 1024, half_size: float = 48.0, y_edges=None):
        self.res = int(res)
        self.hs = float(half_size)
        self._extent = 2.0 * self.hs
        self._inv_extent = 1.0 / self._extent

        edges = DEFAULT_Y_EDGES if y_edges is None else np.asarray(y_edges, dtype=np.float32)
        if edges.ndim != 1 or len(edges) < 4:
            raise ValueError("y_edges must be a 1D monotonically increasing array of >= 4 edges.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("y_edges must be strictly increasing.")

        self.y_edges = edges.copy()
        self.inner_edges = self.y_edges[1:-1]
        self.num_slices = len(self.y_edges) - 1
        self.slice_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.slice_widths = self.y_edges[1:] - self.y_edges[:-1]
        self.y_min = float(self.y_edges[0])
        self.y_max = float(self.y_edges[-1])

        # Shared XZ world coordinate grids matching cc90d58 meshgrid convention (axis0=Z, axis1=X)
        x_coords = np.linspace(-self.hs, self.hs, self.res, dtype=np.float32)
        z_coords = np.linspace(-self.hs, self.hs, self.res, dtype=np.float32)
        self.X, self.Z = np.meshgrid(x_coords, z_coords)
        self.R = np.sqrt(self.X * self.X + self.Z * self.Z)
        self.outer_mask = np.where(
            self.R <= 35.0,
            1.0,
            np.exp(-0.5 * ((self.R - 35.0) / 4.5) ** 2),
        ).astype(np.float32)

        # Precompute Fourier frequency squared grid for fast batched 2D FFT filtering
        ky = np.fft.fftfreq(self.res)[:, None]
        kx = np.fft.rfftfreq(self.res)[None, :]
        self._k2 = (ky * ky + kx * kx).astype(np.float32)
        self._gauss_cache = {}

        # Accumulated C-contiguous (3, num_slices, res, res) float32 stellar radiance volume
        self.stellar_rgb = np.zeros((3, self.num_slices, self.res, self.res), dtype=np.float32)
        # Reused single-component scratch buffer (freed in build_volumes)
        self._comp_buf = np.zeros((3, self.num_slices, self.res, self.res), dtype=np.float32)

        # Per-component slice counts for verification and diagnostics
        self.component_stats = {}

    def _fft_kernel(self, sigma: float) -> np.ndarray:
        key = float(sigma)
        kern = self._gauss_cache.get(key)
        if kern is None:
            kern = np.exp(-2.0 * (np.pi ** 2) * (key ** 2) * self._k2).astype(np.float32)
            self._gauss_cache[key] = kern
        return kern

    def _deposit_into_comp(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           rgb: np.ndarray, lum: np.ndarray,
                           chunk_size: int = 400000) -> np.ndarray:
        """
        Chunks stars directly from CPU arrays into exact vertical Y intervals (0 duplication)
        and XZ pixels using fast C-backed np.bincount.
        Returns slice_star_counts (num_slices,).
        """
        buf = self._comp_buf
        buf.fill(0.0)
        res = self.res
        res2 = res * res
        hs = self.hs
        inv_ext = self._inv_extent
        inner_edges = self.inner_edges
        num_slices = self.num_slices
        counts = np.zeros(num_slices, dtype=np.int64)

        n = len(x)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            xc = np.asarray(x[start:end], dtype=np.float32)
            yc = np.asarray(y[start:end], dtype=np.float32)
            zc = np.asarray(z[start:end], dtype=np.float32)

            mask = (xc >= -hs) & (xc <= hs) & (zc >= -hs) & (zc <= hs)
            if not np.any(mask):
                continue

            xv = xc[mask]
            yv = yc[mask]
            zv = zc[mask]
            rgbc = np.asarray(rgb[start:end], dtype=np.float32)[mask]
            lumc = np.asarray(lum[start:end], dtype=np.float32)[mask]

            ix = np.clip(np.floor((xv + hs) * inv_ext * res).astype(np.int32), 0, res - 1)
            iz = np.clip(np.floor((zv + hs) * inv_ext * res).astype(np.int32), 0, res - 1)
            k = np.searchsorted(inner_edges, yv, side='right').astype(np.int32)
            pix = iz * res + ix

            # Sort indices by slice k in C so each slice is a contiguous slice without boolean re-scans
            if len(k) > 1:
                order = np.argsort(k, kind='mergesort')
                k_sorted = k[order]
                pix_s = pix[order]
                w_rgb = rgbc[order] * lumc[order, None]
            else:
                k_sorted = k
                pix_s = pix
                w_rgb = rgbc * lumc[:, None]

            bounds = np.searchsorted(k_sorted, np.arange(num_slices + 1, dtype=np.int32))
            for s in range(num_slices):
                b0, b1 = int(bounds[s]), int(bounds[s + 1])
                if b1 <= b0:
                    continue
                counts[s] += (b1 - b0)
                ps = pix_s[b0:b1]
                ws = w_rgb[b0:b1]
                for c in range(3):
                    binned = np.bincount(ps, weights=ws[:, c], minlength=res2)
                    buf[c, s].ravel()[:] += binned.astype(np.float32)

        return counts

    def _blur_active_slices_batched(self, counts: np.ndarray, fourier_kernel: np.ndarray):
        """Batched 2D FFT Gaussian blur across active non-empty slices only."""
        active = np.nonzero(counts > 0)[0]
        if len(active) == 0:
            return
        res = self.res
        buf = self._comp_buf
        for c in range(3):
            f_arr = np.fft.rfft2(buf[c, active], axes=(-2, -1))
            blurred = np.fft.irfft2(f_arr * fourier_kernel[None, :, :], s=(res, res), axes=(-2, -1))
            buf[c, active] = np.maximum(blurred, 0.0).astype(np.float32)

    def _blur_adaptive_with_col_norm(self, counts: np.ndarray, base_sigma: float,
                                     slice_sigmas: np.ndarray, channel_mult=(1.0, 1.0, 1.0)):
        """
        Computes exact 2D reference target from column sum (using base_sigma), applies
        per-slice adaptive XZ smoothing (preventing sparse noise in high-|y| outer tails),
        and renormalizes each XZ column so vertical slice sum strictly matches the 2D target.
        """
        res = self.res
        buf = self._comp_buf
        kern_base = self._fft_kernel(base_sigma)
        raw_col_2d = np.sum(buf, axis=1)  # (3, res, res)

        target_2d = np.zeros((3, res, res), dtype=np.float32)
        for c in range(3):
            f_col = np.fft.rfft2(raw_col_2d[c])
            t_c = np.fft.irfft2(f_col * kern_base, s=(res, res)) * float(channel_mult[c])
            target_2d[c] = np.maximum(t_c, 0.0).astype(np.float32)

        # Group active slices sharing the same sigma so we still use fast batched FFTs
        active = np.nonzero(counts > 0)[0]
        unique_sigmas = np.unique(slice_sigmas[active])
        for sig in unique_sigmas:
            group = active[slice_sigmas[active] == sig]
            kern_g = self._fft_kernel(float(sig))
            for c in range(3):
                f_arr = np.fft.rfft2(buf[c, group], axes=(-2, -1))
                bl = np.fft.irfft2(f_arr * kern_g[None, :, :], s=(res, res), axes=(-2, -1)) * float(channel_mult[c])
                buf[c, group] = np.maximum(bl, 0.0).astype(np.float32)

        sum_slices = np.sum(buf, axis=1)
        for c in range(3):
            valid = sum_slices[c] > 1e-8
            scale = np.ones((res, res), dtype=np.float32)
            scale[valid] = target_2d[c, valid] / sum_slices[c, valid]
            buf[c] *= scale[None, :, :]

        return target_2d

    # ------------------------------------------------------------------
    # Component Accumulation Hooks (called during star buffer creation)
    # ------------------------------------------------------------------

    def accumulate_bulge(self, positions: np.ndarray, colors: np.ndarray, brightness: np.ndarray):
        """Accumulate GalacticBulge 3D stars into unique vertical slices."""
        counts = self._deposit_into_comp(
            positions[:, 0], positions[:, 1], positions[:, 2], colors, brightness
        )
        self._blur_active_slices_batched(counts, self._fft_kernel(1.0))

        # Match cc90d58 column-integrated nonlinear core compression
        col_2d = np.sum(self._comp_buf, axis=1)
        bulge_lum = np.sqrt(col_2d[0] ** 2 + col_2d[1] ** 2 + col_2d[2] ** 2)
        bulge_scale = (1.0 / (1.0 + bulge_lum * 0.10)) * 1.25
        tint = (1.42, 1.25, 0.88)

        for c in range(3):
            scale_c = (bulge_scale * tint[c]).astype(np.float32)
            self.stellar_rgb[c] += self._comp_buf[c] * scale_c[None, :, :]

        self.component_stats["bulge"] = {
            "stars": int(len(positions)),
            "slice_counts": counts,
        }

    def accumulate_bar(self, positions: np.ndarray, colors: np.ndarray, brightness: np.ndarray):
        """Accumulate GalacticBar 3D stars (preserving peanut/X lobes) into unique slices."""
        counts = self._deposit_into_comp(
            positions[:, 0], positions[:, 1], positions[:, 2], colors, brightness
        )

        # Preserve cc90d58 exact bar 2D target (0.65*sigma=1.2 + 0.35*sigma=2.6)
        res = self.res
        buf = self._comp_buf
        kern_base = (0.65 * self._fft_kernel(1.2) + 0.35 * self._fft_kernel(2.6)).astype(np.float32)
        raw_col_2d = np.sum(buf, axis=1)
        bar_raw_2d = np.zeros((3, res, res), dtype=np.float32)
        for c in range(3):
            f_col = np.fft.rfft2(raw_col_2d[c])
            bar_raw_2d[c] = np.maximum(np.fft.irfft2(f_col * kern_base, s=(res, res)), 0.0).astype(np.float32)

        # Gentle XZ stabilization ONLY on sparse outer bar tails (|y| >= 0.90) eliminates isolated
        # vertical streaks while preserving peanut lobes (0.35..0.85) with native crisp cc90d58 kernels.
        abs_yc = np.abs(self.slice_centers)
        active = np.nonzero(counts > 0)[0]
        base_group = active[abs_yc[active] < 0.90]
        mid_tail = active[(abs_yc[active] >= 0.90) & (abs_yc[active] < 1.30)]
        far_tail = active[abs_yc[active] >= 1.30]

        for group, kern_g in [
            (base_group, kern_base),
            (mid_tail, self._fft_kernel(2.8)),
            (far_tail, self._fft_kernel(4.5)),
        ]:
            if len(group) == 0:
                continue
            for c in range(3):
                f_arr = np.fft.rfft2(buf[c, group], axes=(-2, -1))
                bl = np.fft.irfft2(f_arr * kern_g[None, :, :], s=(res, res), axes=(-2, -1))
                buf[c, group] = np.maximum(bl, 0.0).astype(np.float32)

        # Column-wise exact normalization to bar_raw_2d before applying cc90d58 nonlinear compression
        sum_raw = np.sum(buf, axis=1)
        for c in range(3):
            valid = sum_raw[c] > 1e-8
            scale = np.ones((res, res), dtype=np.float32)
            scale[valid] = bar_raw_2d[c, valid] / sum_raw[c, valid]
            buf[c] *= scale[None, :, :]

        bar_lum = np.sqrt(bar_raw_2d[0] ** 2 + bar_raw_2d[1] ** 2 + bar_raw_2d[2] ** 2)
        bar_scale = (1.0 / (1.0 + bar_lum * 0.12)) * 4.2
        tint = (1.42, 1.18, 0.60)

        for c in range(3):
            scale_c = (bar_scale * tint[c]).astype(np.float32)
            self.stellar_rgb[c] += buf[c] * scale_c[None, :, :]

        self.component_stats["bar"] = {
            "stars": int(len(positions)),
            "slice_counts": counts,
        }

    def accumulate_disk(self, positions: np.ndarray, colors: np.ndarray,
                        brightness: np.ndarray, n_arm: int):
        """Accumulate GalacticDisk arm and diffuse populations separately while arrays live on CPU."""
        # --- 1. Arm Population ---
        pos_arm = positions[:n_arm]
        x_a = pos_arm[:, 0]
        z_a = pos_arm[:, 2]
        r_a = np.sqrt(x_a * x_a + z_a * z_a)
        th_a = np.arctan2(z_a, x_a)

        r_clamped = np.maximum(r_a, self.INNER_ARM_R)
        arm_spine = self.ARM_BASE_OFFSET + self.SPIRAL_TIGHTNESS * np.log(r_clamped / self.INNER_ARM_R)
        rel_theta = (th_a - arm_spine) % (2.0 * math.pi)
        arm_idx = np.floor(rel_theta / (2.0 * math.pi / self.ARM_COUNT) + 0.5).astype(np.int32) % self.ARM_COUNT

        gate_major = _smoothstep_np(6.0, 7.8, r_a)
        gate_minor = _smoothstep_np(9.2, 11.5, r_a)
        is_major = (arm_idx == 0) | (arm_idx == 2)
        arm_gate = np.where(is_major, gate_major, gate_minor).astype(np.float32)
        lum_arm_gated = brightness[:n_arm] * arm_gate

        arm_counts = self._deposit_into_comp(
            x_a, pos_arm[:, 1], z_a, colors[:n_arm], lum_arm_gated
        )

        kern_arm = (
            (0.40 * self._fft_kernel(1.0)
             + 0.42 * self._fft_kernel(2.8)
             + 0.18 * self._fft_kernel(6.0)) * 0.42
        ).astype(np.float32)
        self._blur_active_slices_batched(arm_counts, kern_arm)

        # Exact cc90d58 blue-white spiral arm radial tint
        arm_trans = _smoothstep_np(6.0, 9.5, self.R)
        arm_tints = (
            ((1.0 - arm_trans) * 1.15 + arm_trans * 0.68).astype(np.float32),
            ((1.0 - arm_trans) * 1.05 + arm_trans * 0.85).astype(np.float32),
            ((1.0 - arm_trans) * 0.75 + arm_trans * 1.48).astype(np.float32),
        )
        active_arm = np.nonzero(arm_counts > 0)[0]
        for c in range(3):
            self.stellar_rgb[c, active_arm] += self._comp_buf[c, active_arm] * arm_tints[c][None, :, :]

        # --- 2. Diffuse Disk Population (Thin + Thick Disk) ---
        pos_diff = positions[n_arm:]
        diff_counts = self._deposit_into_comp(
            pos_diff[:, 0], pos_diff[:, 1], pos_diff[:, 2], colors[n_arm:], brightness[n_arm:]
        )

        abs_yc = np.abs(self.slice_centers)
        slice_sigmas = np.where(abs_yc >= 1.80, 10.5, np.where(abs_yc >= 1.20, 7.2, 5.0)).astype(np.float32)
        self._blur_adaptive_with_col_norm(
            diff_counts,
            base_sigma=5.0,
            slice_sigmas=slice_sigmas,
            channel_mult=(0.22, 0.22, 0.26),
        )
        self.stellar_rgb += self._comp_buf

        self.component_stats["arm"] = {
            "stars": int(n_arm),
            "slice_counts": arm_counts,
        }
        self.component_stats["diffuse"] = {
            "stars": int(len(positions) - n_arm),
            "slice_counts": diff_counts,
        }

    # ------------------------------------------------------------------
    # Dust & Nebula Volume Generation + Final GPU-Ready Slice Packaging
    # ------------------------------------------------------------------

    def _gaussian_interval_weights(self, mean_y, sigma_y: float) -> np.ndarray:
        """Exact Gaussian CDF integral over each disjoint slice interval; sums strictly to 1.0."""
        inv_s2 = 1.0 / (math.sqrt(2.0) * max(sigma_y, 1e-4))
        lo = _erf_np((self.y_edges[:-1] - mean_y) * inv_s2)
        hi = _erf_np((self.y_edges[1:] - mean_y) * inv_s2)
        w = np.maximum(0.5 * (hi - lo), 0.0).astype(np.float32)
        tot = np.sum(w, axis=0, keepdims=True)
        w = np.where(tot > 1e-8, w / tot, 0.0)
        return w

    def _build_dust_slices(self) -> np.ndarray:
        """
        Generates vertical dust optical depth slices whose vertical sum reproduces cc90d58
        dust_optical_depth while strictly confining dust near the midplane according to the
        two-component vertical scale height from dust_vol_frag.glsl.
        """
        X, Z, R = self.X, self.Z, self.R

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
        X_w = X + noise_warp_x * 0.80
        Z_w = Z + noise_warp_z * 0.80
        R_w = np.sqrt(X_w * X_w + Z_w * Z_w)
        Theta_w = np.arctan2(Z_w, X_w)
        r_arm_w = np.maximum(R_w, self.INNER_ARM_R)

        noise_dust = (
            0.40 * np.sin(X * 0.48 + Z * 0.38) * np.cos(Z * 0.58 - X * 0.28) +
            0.30 * np.sin(X * 1.25 - Z * 0.95) * np.sin(Z * 1.35 + X * 0.75) +
            0.18 * np.cos(X * 2.65 + Z * 2.35) * np.sin(Z * 3.05 - X * 2.55)
        )

        arm_dust_total = np.zeros_like(R, dtype=np.float32)
        for i in range(self.ARM_COUNT):
            is_maj = (i % 2 == 0)
            r_start = 6.6 if is_maj else 10.5
            w_arm = 1.0 if is_maj else 0.60

            arm_base = float(i) * (2.0 * math.pi / self.ARM_COUNT)
            ideal_theta = arm_base + self.ARM_BASE_OFFSET + self.SPIRAL_TIGHTNESS * np.log(r_arm_w / self.INNER_ARM_R)
            d_theta = (Theta_w - ideal_theta + math.pi) % (2.0 * math.pi) - math.pi

            d_arm = d_theta - (-0.14)
            linear_dist = np.abs(d_arm) * R_w

            lane_w = (0.040 + 0.030 * (R / 45.0)) * R
            core_lane = np.exp(-0.5 * (linear_dist / np.maximum(lane_w, 1e-4)) ** 2) * w_arm * 2.5

            d_arm_sub = d_theta - (-0.02)
            sub_lane = np.exp(-0.5 * (np.abs(d_arm_sub) * R_w / np.maximum(lane_w * 1.5, 1e-4)) ** 2) * w_arm * 0.65

            s_norm = (R - self.INNER_ARM_R) / 41.5
            wisp_mod = 0.85 + 0.35 * np.sin(2.0 * math.pi * 5.5 * s_norm + i * 1.57) + 0.25 * noise_dust
            lane_combined = (core_lane + sub_lane) * np.clip(wisp_mod, 0.20, 1.80)
            arm_dust_total += lane_combined * _smoothstep_np(r_start - 0.5, r_start + 1.2, R) * (R <= 42.0)

        tau_col_2d = (arm_dust_total * self.outer_mask).astype(np.float32)

        h_thin = (0.10 + 0.08 * _smoothstep_np(4.0, 30.0, R)).astype(np.float32)
        h_thick = (0.28 + 0.14 * _smoothstep_np(4.0, 30.0, R)).astype(np.float32)

        dust_slices = np.zeros((self.num_slices, self.res, self.res), dtype=np.float32)
        inv_s2_thin = 1.0 / (math.sqrt(2.0) * h_thin)
        inv_s2_thick = 1.0 / (math.sqrt(2.0) * h_thick)
        for s in range(self.num_slices):
            y0 = float(self.y_edges[s])
            y1 = float(self.y_edges[s + 1])
            w_thin = 0.5 * (_erf_np(y1 * inv_s2_thin) - _erf_np(y0 * inv_s2_thin))
            w_thick = 0.5 * (_erf_np(y1 * inv_s2_thick) - _erf_np(y0 * inv_s2_thick))
            dust_slices[s] = tau_col_2d * (0.72 * w_thin + 0.28 * w_thick)

        return dust_slices

    def _build_nebula_slices(self, nebula_mgr=None) -> np.ndarray:
        """
        Generates localized H-II & reflection nebula slices confined tightly near each
        complex's true vertical Y position while strictly preserving the cc90d58 face-on nebula map.
        """
        res = self.res
        hs = self.hs
        X, Z = self.X, self.Z
        neb_slices = np.zeros((self.num_slices, res, res, 3), dtype=np.float32)

        # 1. Shared NebulaManager complexes (placed at actual 3D ny)
        if nebula_mgr is not None and hasattr(nebula_mgr, 'nebulae_data'):
            for n in nebula_mgr.nebulae_data:
                nx, ny, nz, nr = n['position']
                col = np.array(n['color'][:3], dtype=np.float32) * float(n['color'][3])
                ix0 = max(0, int((nx - 3.0 * nr + hs) / (2.0 * hs) * res))
                ix1 = min(res, int((nx + 3.0 * nr + hs) / (2.0 * hs) * res) + 1)
                iz0 = max(0, int((nz - 3.0 * nr + hs) / (2.0 * hs) * res))
                iz1 = min(res, int((nz + 3.0 * nr + hs) / (2.0 * hs) * res) + 1)
                if ix1 > ix0 and iz1 > iz0:
                    subX = X[iz0:iz1, ix0:ix1]
                    subZ = Z[iz0:iz1, ix0:ix1]
                    d2 = (subX - nx) ** 2 + (subZ - nz) ** 2
                    mask_2d = (
                        np.exp(-0.5 * d2 / (nr * 0.40) ** 2) * 1.8
                        + np.exp(-0.5 * d2 / (nr * 0.80) ** 2) * 0.4
                    ).astype(np.float32)

                    sigma_y = max(0.06, min(0.18, 0.25 * float(nr)))
                    wy = self._gaussian_interval_weights(float(ny), sigma_y)
                    for s in range(self.num_slices):
                        if wy[s] <= 1e-4:
                            continue
                        w_col = col * float(wy[s])
                        for c in range(3):
                            neb_slices[s, iz0:iz1, ix0:ix1, c] += mask_2d * w_col[c]

        # 2. Clustered H-II star-forming arm beads & bar-tip starbursts
        rng_neb = np.random.default_rng(888)
        rng_bead_y = np.random.default_rng(1888)

        num_beads = 480
        r_beads = rng_neb.uniform(6.8, 39.0, num_beads)
        arm_beads = rng_neb.choice(4, size=num_beads, p=[0.38, 0.12, 0.38, 0.12])
        base_th_beads = arm_beads * (2.0 * math.pi / 4.0)
        th_beads = (
            base_th_beads
            + self.ARM_BASE_OFFSET
            + self.SPIRAL_TIGHTNESS * np.log(r_beads / self.INNER_ARM_R)
            + rng_neb.normal(-0.06, 0.035, num_beads)
        )

        x_beads = list(r_beads * np.cos(th_beads))
        z_beads = list(r_beads * np.sin(th_beads))
        size_beads = list(rng_neb.uniform(0.10, 0.26, num_beads))
        int_beads = list(rng_neb.uniform(0.5, 1.4, num_beads))

        sin_b = math.sin(self.BAR_ANGLE)
        cos_b = math.cos(self.BAR_ANGLE)
        for tip_sign in [-1.0, 1.0]:
            tip_u = tip_sign * 6.6
            for _ in range(16):
                u_clump = tip_u + rng_neb.normal(0.0, 0.50)
                v_clump = tip_sign * 0.95 + rng_neb.normal(0.0, 0.35)
                x_beads.append(u_clump * sin_b - v_clump * cos_b)
                z_beads.append(u_clump * cos_b + v_clump * sin_b)
                size_beads.append(rng_neb.uniform(0.12, 0.30))
                int_beads.append(rng_neb.uniform(0.7, 1.6))

        y_beads = rng_bead_y.normal(0.0, 0.055, size=len(x_beads))
        for bx, by, bz, bs, bi in zip(x_beads, y_beads, z_beads, size_beads, int_beads):
            ix0 = max(0, int((bx - 3.0 * bs + hs) / (2.0 * hs) * res))
            ix1 = min(res, int((bx + 3.0 * bs + hs) / (2.0 * hs) * res) + 1)
            iz0 = max(0, int((bz - 3.0 * bs + hs) / (2.0 * hs) * res))
            iz1 = min(res, int((bz + 3.0 * bs + hs) / (2.0 * hs) * res) + 1)
            if ix1 > ix0 and iz1 > iz0:
                subX = X[iz0:iz1, ix0:ix1]
                subZ = Z[iz0:iz1, ix0:ix1]
                d2 = (subX - bx) ** 2 + (subZ - bz) ** 2
                mask_2d = (
                    np.exp(-0.5 * d2 / (bs * 0.30) ** 2) * bi
                    + np.exp(-0.5 * d2 / (bs * 0.70) ** 2) * (bi * 0.20)
                ).astype(np.float32)

                is_blue = rng_neb.random() < 0.32
                if is_blue:
                    b_col = np.array([0.55, 0.88, 1.40], dtype=np.float32)
                else:
                    hue_var = rng_neb.uniform(-0.05, 0.05)
                    b_col = np.array([1.25 + hue_var, 0.20, 0.52 - hue_var], dtype=np.float32)

                wy = self._gaussian_interval_weights(float(by), 0.065)
                for s in range(self.num_slices):
                    if wy[s] <= 1e-4:
                        continue
                    w_col = b_col * float(wy[s])
                    for c in range(3):
                        neb_slices[s, iz0:iz1, ix0:ix1, c] += mask_2d * w_col[c]

        return neb_slices

    def build_volumes(self, nebula_mgr=None, as_float16: bool = True, free_staging: bool = True):
        """
        Finalizes the vertical slices and packs GPU-ready multichannel textures:
        - stellar_vol:  (num_slices, res, res, 4) -> RGB = slice stellar radiance, A = density alpha
        - dust_neb_vol: (num_slices, res, res, 4) -> R = dust optical depth, GBA = nebula RGB
        All column sums across slices preserve the exact cc90d58 reference 2D projections.
        """
        self._comp_buf = None
        outer_mask = self.outer_mask

        # Apply outer radial mask in-place to avoid allocating duplicate volume arrays
        self.stellar_rgb *= outer_mask[None, None, :, :]
        sr = self.stellar_rgb[0]
        sg = self.stellar_rgb[1]
        sb = self.stellar_rgb[2]

        col_r = np.sum(sr, axis=0)
        col_g = np.sum(sg, axis=0)
        col_b = np.sum(sb, axis=0)
        col_density = np.sqrt(col_r * col_r + col_g * col_g + col_b * col_b)
        alpha_col_target = np.clip(col_density * 0.50, 0.0, 1.0) * outer_mask

        slice_dens = np.sqrt(sr * sr + sg * sg + sb * sb)
        sum_slice_dens = np.sum(slice_dens, axis=0)
        valid_a = sum_slice_dens > 1e-8
        alpha_scale = np.zeros((self.res, self.res), dtype=np.float32)
        alpha_scale[valid_a] = alpha_col_target[valid_a] / sum_slice_dens[valid_a]

        dtype = np.float16 if as_float16 else np.float32
        stellar_vol = np.empty((self.num_slices, self.res, self.res, 4), dtype=dtype)
        stellar_vol[:, :, :, 0] = sr
        stellar_vol[:, :, :, 1] = sg
        stellar_vol[:, :, :, 2] = sb
        stellar_vol[:, :, :, 3] = slice_dens * alpha_scale[None, :, :]

        if free_staging:
            self.stellar_rgb = None
            del slice_dens, sum_slice_dens, alpha_scale, col_density

        dust_slices = self._build_dust_slices()
        neb_slices = self._build_nebula_slices(nebula_mgr=nebula_mgr)

        dust_neb_vol = np.empty((self.num_slices, self.res, self.res, 4), dtype=dtype)
        dust_neb_vol[:, :, :, 0] = dust_slices
        dust_neb_vol[:, :, :, 1] = neb_slices[:, :, :, 0]
        dust_neb_vol[:, :, :, 2] = neb_slices[:, :, :, 1]
        dust_neb_vol[:, :, :, 3] = neb_slices[:, :, :, 2]

        if free_staging:
            self._k2 = None
            self._gauss_cache.clear()

        return stellar_vol, dust_neb_vol

    def release_cpu_buffers(self):
        """Frees CPU staging buffers after GPU texture upload."""
        self.stellar_rgb = None
        self._comp_buf = None
        self._k2 = None
        self._gauss_cache.clear()
