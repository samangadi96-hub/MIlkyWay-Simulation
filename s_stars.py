"""
Sagittarius A* Nuclear S-Star Cluster & Toggleable Identification Overlay.

Astrophysically grounded 3D Keplerian + leading-order 1PN Schwarzschild dynamics
anchored strictly to the central supermassive black hole (Sgr A*).

References:
- S2 reference orbit & cluster catalog: Gillessen et al. (2017, ApJ 837, 30),
  GRAVITY Collaboration (2019, A&A 625, L10; 2020, A&A 636, L5; 2022, A&A 657, L12).
- Ultra-short-period eccentric stars (S62, S4714): Peissker et al. (2020, ApJ 889, 61;
  ApJ 899, 50).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import glm
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Physical constants (IAU exact definitions)
AU_KM = 149_597_870.7
JULIAN_YEAR_S = 31_557_600.0
C_KMS = 299_792.458
AU_PER_YR_TO_KMS = AU_KM / JULIAN_YEAR_S  # 4.740470463533348 km/s per (AU/yr)
GM_SUN_OVER_C2_KM = 1.476625038050125     # km


@dataclass(frozen=True)
class SStarSpec:
    name: str
    a_arcsec: float
    e: float
    inc_deg: float
    omega_node_deg: float
    arg_peri_deg: float
    m0_rad: float
    star_rgb: tuple[float, float, float]
    orbit_rgb: tuple[float, float, float]
    point_size: float
    brightness: float
    is_measured: bool = True
    is_reference: bool = False


def r_au_to_visual(r_au: np.ndarray | float) -> np.ndarray | float:
    """
    Monotonically map physical orbital radius in AU into the local Sgr A*
    rendering frame without flattening elliptical shape, orientation, or
    pericentre hierarchy.

    Keeps even the tightest pericentres (S4714 ~12.5 AU, S2 ~119 AU) strictly
    outside the black hole event horizon / photon shadow while framing the full
    cluster within close Galactic Centre camera views and keeping far-galaxy
    views clean of any artificial ring.
    """
    return 0.24 + 0.017 * np.power(np.maximum(r_au, 0.0), 0.54)


def _solve_kepler_array(mean_anom: np.ndarray, ecc: np.ndarray) -> np.ndarray:
    """
    Vectorized Halley-Newton solver for Kepler's equation M = E - e * sin(E).
    Converges to < 1e-13 rad across all eccentricities up to e = 0.99.
    """
    M = np.mod(mean_anom + math.pi, 2.0 * math.pi) - math.pi
    # Good initial guess for high-e orbits
    E = M + ecc * np.sin(M) + 0.5 * (ecc ** 2) * np.sin(2.0 * M)
    high_e = ecc > 0.78
    if np.any(high_e):
        sign_M = np.where(M >= 0.0, 1.0, -1.0)
        E_hi = sign_M * np.power(np.maximum(6.0 * np.abs(M), 1e-12), 1.0 / 3.0)
        E_small = M / np.maximum(1.0 - ecc, 0.02)
        E = np.where(high_e & (np.abs(M) < 0.35), np.where(np.abs(M) < 0.06, E_small, E_hi), E)

    for _ in range(9):
        sin_E = np.sin(E)
        cos_E = np.cos(E)
        f = E - ecc * sin_E - M
        fp = 1.0 - ecc * cos_E
        fpp = ecc * sin_E
        # Halley third-order step clamped safely for stability
        denom = fp - 0.5 * f * fpp / np.maximum(fp, 1e-8)
        step = f / np.where(np.abs(denom) > 1e-10, denom, fp)
        step = np.clip(step, -0.45, 0.45)
        E = E - step
    return E


def _build_identified_catalog() -> list[SStarSpec]:
    """
    Observationally constrained reference S-stars orbiting Sagittarius A*.
    Angles (i, Omega, omega) are astrometric Euler angles in degrees; m0_rad
    sets an illustrative initial orbital epoch with S2 approaching pericentre.
    """
    return [
        # S2 (S0-2) — Primary relativistic testbed (GRAVITY / Keck)
        SStarSpec(
            name="S2",
            a_arcsec=0.12505,
            e=0.8840,
            inc_deg=134.57,
            omega_node_deg=228.17,
            arg_peri_deg=66.26,
            m0_rad=-0.45,
            star_rgb=(0.90, 0.95, 1.00),
            orbit_rgb=(1.00, 0.87, 0.44),
            point_size=2.25,
            brightness=2.45,
            is_measured=True,
            is_reference=True,
        ),
        # S62 — Ultra-short-period ~9.9 yr eccentric orbit (Peissker et al. 2020)
        SStarSpec(
            name="S62",
            a_arcsec=0.0897,
            e=0.9760,
            inc_deg=72.76,
            omega_node_deg=122.61,
            arg_peri_deg=42.62,
            m0_rad=1.45,
            star_rgb=(0.78, 0.89, 1.00),
            orbit_rgb=(0.45, 0.80, 1.00),
            point_size=1.78,
            brightness=1.90,
            is_measured=True,
        ),
        # S4714 — Extreme eccentricity plunging star (Peissker et al. 2020)
        SStarSpec(
            name="S4714",
            a_arcsec=0.1020,
            e=0.9850,
            inc_deg=127.70,
            omega_node_deg=129.28,
            arg_peri_deg=357.25,
            m0_rad=-1.75,
            star_rgb=(0.75, 0.87, 1.00),
            orbit_rgb=(0.42, 0.78, 1.00),
            point_size=1.75,
            brightness=1.88,
            is_measured=True,
        ),
        # S55 (S0-102) — Tight 12.8 yr orbit (Meyer et al. / Gillessen et al.)
        SStarSpec(
            name="S55",
            a_arcsec=0.1078,
            e=0.7209,
            inc_deg=150.10,
            omega_node_deg=325.50,
            arg_peri_deg=331.50,
            m0_rad=2.50,
            star_rgb=(0.82, 0.91, 1.00),
            orbit_rgb=(0.48, 0.78, 0.98),
            point_size=1.70,
            brightness=1.82,
            is_measured=True,
        ),
        # S38 (S0-38) — Face-on retrograde high-e inner star
        SStarSpec(
            name="S38",
            a_arcsec=0.1416,
            e=0.8201,
            inc_deg=171.10,
            omega_node_deg=101.06,
            arg_peri_deg=17.99,
            m0_rad=-2.35,
            star_rgb=(0.94, 0.95, 1.00),
            orbit_rgb=(0.50, 0.78, 0.98),
            point_size=1.70,
            brightness=1.80,
            is_measured=True,
        ),
        # S13 (S0-20) — Intermediate moderate-eccentricity B-star
        SStarSpec(
            name="S13",
            a_arcsec=0.2641,
            e=0.4250,
            inc_deg=24.70,
            omega_node_deg=74.50,
            arg_peri_deg=245.20,
            m0_rad=0.68,
            star_rgb=(0.82, 0.91, 1.00),
            orbit_rgb=(0.45, 0.75, 0.96),
            point_size=1.68,
            brightness=1.78,
            is_measured=True,
        ),
        # S9 (S0-5) — Late-type / warm intermediate member
        SStarSpec(
            name="S9",
            a_arcsec=0.2724,
            e=0.6440,
            inc_deg=82.41,
            omega_node_deg=156.60,
            arg_peri_deg=150.60,
            m0_rad=-0.95,
            star_rgb=(1.00, 0.90, 0.76),
            orbit_rgb=(0.48, 0.75, 0.96),
            point_size=1.72,
            brightness=1.80,
            is_measured=True,
        ),
        # S1 (S0-1) — Luminous early-type star in the outer S-cluster
        SStarSpec(
            name="S1",
            a_arcsec=0.5950,
            e=0.5560,
            inc_deg=119.14,
            omega_node_deg=342.04,
            arg_peri_deg=122.30,
            m0_rad=1.90,
            star_rgb=(0.76, 0.88, 1.00),
            orbit_rgb=(0.44, 0.74, 0.96),
            point_size=1.82,
            brightness=1.90,
            is_measured=True,
        ),
    ]


def _build_synthetic_tracers(count: int = 28, seed: int = 4207) -> list[SStarSpec]:
    """
    Static seeded population of representative synthetic S-star model tracers
    with thermal-like eccentricities and isotropic 3D inclinations (explicitly
    distinguished in metadata/legend from measured stars).
    """
    rng = np.random.default_rng(seed)
    tracers: list[SStarSpec] = []

    for idx in range(count):
        # Log-uniform semi-major axes spanning ~0.09" to ~0.55" (~740 - 4500 AU)
        a_arcsec = float(np.exp(rng.uniform(math.log(0.095), math.log(0.52))))
        # Sample e in [0.30, 0.93] with thermal bias while keeping r_p > 38 AU
        for _ in range(16):
            u = rng.uniform(0.30 ** 2, 0.93 ** 2)
            e = float(math.sqrt(u))
            if a_arcsec * 8200.0 * (1.0 - e) >= 38.0:
                break
        else:
            e = float(rng.uniform(0.32, 0.78))

        # Isotropic 3D orbital planes
        cos_i = float(rng.uniform(-1.0, 1.0))
        inc_deg = math.degrees(math.acos(np.clip(cos_i, -1.0, 1.0)))
        omega_node_deg = float(rng.uniform(0.0, 360.0))
        arg_peri_deg = float(rng.uniform(0.0, 360.0))
        m0_rad = float(rng.uniform(-math.pi, math.pi))

        # Realistic S-cluster spectral tint mix (predominantly B/A main-sequence)
        temp_roll = rng.random()
        if temp_roll < 0.65:
            star_rgb = (0.76, 0.88, 1.00)
        elif temp_roll < 0.88:
            star_rgb = (0.94, 0.94, 0.98)
        else:
            star_rgb = (1.00, 0.86, 0.70)

        tracers.append(
            SStarSpec(
                name=f"SYN-{idx + 1:02d}",
                a_arcsec=a_arcsec,
                e=e,
                inc_deg=inc_deg,
                omega_node_deg=omega_node_deg,
                arg_peri_deg=arg_peri_deg,
                m0_rad=m0_rad,
                star_rgb=star_rgb,
                orbit_rgb=(0.46, 0.57, 0.74),
                point_size=float(rng.uniform(0.95, 1.25)),
                brightness=float(rng.uniform(0.75, 1.10)),
                is_measured=False,
                is_reference=False,
            )
        )

    return tracers


class SStarCluster:
    """
    Renders and simulates the Sagittarius A* S-star cluster:
    - Exact static Keplerian catalog (identified GRAVITY/Keck stars + seeded tracers)
    - Leading-order 1PN Schwarzschild periapse precession, gravitational redshift,
      transverse Doppler effect, and subtle radial relativistic Doppler modulation
    - Front/back depth split relative to Sgr A* for seamless black-hole shadow and
      accretion-disk occlusion
    - Screen-space toggleable identification overlay (`Y`) strictly independent of
      cluster/orbit visibility (`O`)
    """

    ORBIT_SEGMENTS = 160

    def __init__(
        self,
        ctx: moderngl.Context,
        base_dir: Path | str,
        bh_mass: float = 4.30e6,
    ):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.bh_mass = float(bh_mass)

        # Toggle states (independent)
        self.show_cluster = True    # O key: 3D orbits + stars
        self.show_labels = False    # Y key: screen-space badges + HUD legend

        # Distance fade calibration (visible in close/GC views, invisible in galaxy/far views)
        self.fade_start_dist = 11.0
        self.fade_end_dist = 17.5

        # Self-consistent Galactic Centre distance (GRAVITY 2019: 8178 pc at 4.154e6 M_sun;
        # GRAVITY 2022: 8277 pc at 4.297e6 M_sun). Scales smoothly via M ~ R0^3 so
        # angular orbital periods match observations across the 4.0-4.3e6 M_sun band.
        self.r0_pc = 8178.0 * ((self.bh_mass / 4.154e6) ** (1.0 / 3.0))
        self.au_per_arcsec = self.r0_pc  # 1 arcsec at D pc = D AU by definition

        # Schwarzschild radius R_s = 2 G M / c^2 in AU
        self.rs_km = 2.0 * self.bh_mass * GM_SUN_OVER_C2_KM
        self.rs_au = self.rs_km / AU_KM

        # Assemble immutable orbital catalog
        self.specs: list[SStarSpec] = _build_identified_catalog() + _build_synthetic_tracers(28, seed=4207)
        self.num_stars = len(self.specs)
        self.identified_indices = [i for i, s in enumerate(self.specs) if s.is_measured]

        # Precompute orbital invariants (vectorized over N stars)
        self._init_orbital_arrays()
        self._init_gpu_resources()
        self._build_overlay_atlas()

        # Perform initial evaluation at t = 0
        self._last_time_yr = None
        self._overlay_vert_count = 0
        self.update(0.0, (0.0, 0.25, 1.0))

    # ==================================================================
    # Initialization
    # ==================================================================

    def _init_orbital_arrays(self) -> None:
        self.a_au = np.array([s.a_arcsec * self.au_per_arcsec for s in self.specs], dtype=np.float64)
        self.e = np.array([s.e for s in self.specs], dtype=np.float64)
        self.inc = np.radians([s.inc_deg for s in self.specs]).astype(np.float64)
        self.omega_node = np.radians([s.omega_node_deg for s in self.specs]).astype(np.float64)
        self.arg_peri0 = np.radians([s.arg_peri_deg for s in self.specs]).astype(np.float64)
        self.m0 = np.array([s.m0_rad for s in self.specs], dtype=np.float64)

        self.star_rgb = np.array([s.star_rgb for s in self.specs], dtype=np.float32)
        self.orbit_rgb = np.array([s.orbit_rgb for s in self.specs], dtype=np.float32)
        self.star_size = np.array([s.point_size for s in self.specs], dtype=np.float32)
        self.star_brightness = np.array([s.brightness for s in self.specs], dtype=np.float32)
        self.is_measured = np.array([s.is_measured for s in self.specs], dtype=bool)
        self.is_reference = np.array([s.is_reference for s in self.specs], dtype=bool)

        # Kepler's 3rd Law: P [Julian yr] = sqrt(a_AU^3 / M_BH [M_sun])
        self.period_yr = np.sqrt((self.a_au ** 3) / self.bh_mass)
        self.mean_motion = (2.0 * math.pi) / self.period_yr  # rad / yr

        # Semi-minor ratio sqrt(1 - e^2)
        self.sqrt_1_minus_e2 = np.sqrt(1.0 - self.e ** 2)

        # Leading-order Schwarzschild 1PN periapse precession:
        # delta_omega = 6 pi G M / (c^2 a (1 - e^2)) = 3 pi R_s / (a (1 - e^2)) rad / orbit
        self.delta_omega_per_orbit = (3.0 * math.pi * self.rs_au) / (
            self.a_au * (1.0 - self.e ** 2)
        )
        self.d_omega_dt = self.delta_omega_per_orbit / self.period_yr  # rad / yr

        # Base perifocal unit vectors (P0, Q0) mapped cleanly into 3D space where
        # (X, Y) forms the astrometric reference plane and +Z faces the default camera.
        cO = np.cos(self.omega_node)
        sO = np.sin(self.omega_node)
        ci = np.cos(self.inc)
        si = np.sin(self.inc)
        cw = np.cos(self.arg_peri0)
        sw = np.sin(self.arg_peri0)

        self.P0 = np.stack([
            cO * cw - sO * sw * ci,
            sO * cw + cO * sw * ci,
            sw * si,
        ], axis=1)  # (N, 3)

        self.Q0 = np.stack([
            -cO * sw - sO * cw * ci,
            -sO * sw + cO * cw * ci,
            cw * si,
        ], axis=1)  # (N, 3)

        # Precompute co-precessing ellipse points uniformly sampled in eccentric anomaly E.
        # Uniform E naturally refines vertex spacing around high-curvature pericentres.
        n_pts = self.ORBIT_SEGMENTS
        E_grid = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False, dtype=np.float64)
        cos_Eg = np.cos(E_grid)[None, :]  # (1, K)
        sin_Eg = np.sin(E_grid)[None, :]  # (1, K)

        a_col = self.a_au[:, None]
        e_col = self.e[:, None]
        b_col = self.sqrt_1_minus_e2[:, None]

        x_orb = a_col * (cos_Eg - e_col)
        y_orb = a_col * b_col * sin_Eg
        r_pt = np.maximum(a_col * (1.0 - e_col * cos_Eg), 1e-6)
        r_vis_pt = r_au_to_visual(r_pt)
        scale_pt = r_vis_pt / r_pt

        # Pre-scaled visual-frame perifocal coordinates (rotated coherently by P(t), Q(t) per frame)
        self.orb_ux = (x_orb * scale_pt).astype(np.float32)  # (N, K)
        self.orb_uy = (y_orb * scale_pt).astype(np.float32)  # (N, K)
        # Mean anomaly M(E) = E - e sin(E) in [0, 2pi) for trailing brightness gradient
        self.orb_M = np.mod(E_grid[None, :] - e_col * sin_Eg, 2.0 * math.pi)

        # Base segment alphas by population tier (linear HDR light; gamma 2.2 lifts low values)
        self.orb_alpha_min = np.where(
            self.is_reference, 0.22,
            np.where(self.is_measured, 0.060, 0.012)
        ).astype(np.float32)
        self.orb_alpha_max = np.where(
            self.is_reference, 0.62,
            np.where(self.is_measured, 0.25, 0.048)
        ).astype(np.float32)

        # Live dynamical state arrays (exposed for verification & overlay)
        self.pos_au = np.zeros((self.num_stars, 3), dtype=np.float64)
        self.vel_kms = np.zeros((self.num_stars, 3), dtype=np.float64)
        self.pos_world = np.zeros((self.num_stars, 3), dtype=np.float32)
        self.r_au = np.zeros(self.num_stars, dtype=np.float64)
        self.speed_kms = np.zeros(self.num_stars, dtype=np.float64)
        self.redshift_z = np.zeros(self.num_stars, dtype=np.float64)
        self.beaming_factor = np.ones(self.num_stars, dtype=np.float64)
        self.mean_anomaly = np.zeros(self.num_stars, dtype=np.float64)

        # Reusable interleaved vertex scratch buffers
        # Star vertex layout: 3f pos, 3f color, 1f size, 1f brightness -> 8 floats/star
        self._star_vbo_data = np.zeros((self.num_stars, 8), dtype=np.float32)
        self._star_vbo_data[:, 6] = self.star_size
        # Orbit vertex layout: 2 endpoints/seg -> (N, K, 2, 7) where 7 = (xyz, rgba)
        self._orbit_vbo_data = np.zeros((self.num_stars, n_pts, 2, 7), dtype=np.float32)

        # Allocate space for HUD overlay dynamic vertices (reticles + leaders + badges + legend)
        self._overlay_vbo_data = np.zeros((256 * 6, 8), dtype=np.float32)

    def _init_gpu_resources(self) -> None:
        shader_dir = self.base_dir / "shaders"

        with open(shader_dir / "s_star_vert.glsl", encoding="utf-8") as f:
            sv = f.read()
        with open(shader_dir / "s_star_frag.glsl", encoding="utf-8") as f:
            sf = f.read()
        self.star_program = self.ctx.program(vertex_shader=sv, fragment_shader=sf)

        with open(shader_dir / "s_orbit_vert.glsl", encoding="utf-8") as f:
            ov = f.read()
        with open(shader_dir / "s_orbit_frag.glsl", encoding="utf-8") as f:
            of = f.read()
        self.orbit_program = self.ctx.program(vertex_shader=ov, fragment_shader=of)

        with open(shader_dir / "s_overlay_vert.glsl", encoding="utf-8") as f:
            lv = f.read()
        with open(shader_dir / "s_overlay_frag.glsl", encoding="utf-8") as f:
            lf = f.read()
        self.overlay_program = self.ctx.program(vertex_shader=lv, fragment_shader=lf)

        self.star_vbo = self.ctx.buffer(reserve=self._star_vbo_data.nbytes, dynamic=True)
        self.star_vao = self.ctx.vertex_array(
            self.star_program,
            [(self.star_vbo, "3f 3f 1f 1f", "in_position", "in_color", "in_size", "in_brightness")],
        )

        self.orbit_vbo = self.ctx.buffer(reserve=self._orbit_vbo_data.nbytes, dynamic=True)
        self.orbit_vao = self.ctx.vertex_array(
            self.orbit_program,
            [(self.orbit_vbo, "3f 4f", "in_position", "in_color")],
        )

        self.overlay_vbo = self.ctx.buffer(reserve=self._overlay_vbo_data.nbytes, dynamic=True)
        self.overlay_vao = self.ctx.vertex_array(
            self.overlay_program,
            [(self.overlay_vbo, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
        )

    def _build_overlay_atlas(self) -> None:
        """
        Pre-render high-contrast outlined label badges for identified S-stars and a
        translucent dark-glass astrophysical legend panel into one RGBA texture atlas.
        """
        ATLAS_W, ATLAS_H = 1024, 512
        img = Image.new("RGBA", (ATLAS_W, ATLAS_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        def load_font(candidates: list[str], size: int):
            for fp in candidates:
                if Path(fp).exists():
                    try:
                        return ImageFont.truetype(fp, size)
                    except Exception:
                        pass
            return ImageFont.load_default()

        bold_fonts = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        reg_fonts = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        font_badge = load_font(bold_fonts, 14)
        font_title = load_font(bold_fonts, 14)
        font_sub = load_font(reg_fonts, 12)
        font_body = load_font(reg_fonts, 12)
        font_hint = load_font(bold_fonts, 11)

        # Metadata: star_idx -> dict(uv=(u0,v0,u1,v1), w_px, h_px)
        self._badge_meta: dict[int, dict] = {}

        cur_x, cur_y = 8, 8
        row_h = 0

        for idx in self.identified_indices:
            spec = self.specs[idx]
            text = spec.name
            bbox = draw.textbbox((0, 0), text, font=font_badge)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            dot_pad = 13 if spec.is_reference else 11
            pad_x = 7
            pad_y = 4
            w_px = dot_pad + text_w + pad_x + 4
            h_px = max(21, text_h + pad_y * 2 + 2)

            if cur_x + w_px + 8 > ATLAS_W:
                cur_x = 8
                cur_y += row_h + 8
                row_h = 0

            x0, y0 = cur_x, cur_y
            x1, y1 = x0 + w_px - 1, y0 + h_px - 1

            if spec.is_reference:
                bg = (16, 20, 32, 224)
                border = (255, 198, 72, 240)
                text_col = (255, 242, 204, 255)
                dot_col = (255, 200, 74, 255)
            else:
                bg = (11, 17, 29, 212)
                border = (108, 194, 255, 215)
                text_col = (232, 245, 255, 255)
                dot_col = (118, 204, 255, 255)

            draw.rounded_rectangle([x0, y0, x1, y1], radius=5, fill=bg, outline=border, width=1)

            # Small crisp indicator symbol inside badge left edge (avoids font glyph bugs)
            cx_dot = x0 + 7
            cy_dot = (y0 + y1) // 2
            if spec.is_reference:
                r_d = 3
                draw.polygon(
                    [(cx_dot, cy_dot - r_d), (cx_dot + r_d, cy_dot),
                     (cx_dot, cy_dot + r_d), (cx_dot - r_d, cy_dot)],
                    fill=dot_col,
                )
            else:
                draw.ellipse([cx_dot - 2, cy_dot - 2, cx_dot + 2, cy_dot + 2], fill=dot_col)

            tx = x0 + dot_pad - bbox[0]
            ty = y0 + (h_px - text_h) // 2 - bbox[1]
            draw.text((tx, ty), text, font=font_badge, fill=text_col)

            self._badge_meta[idx] = {
                "uv": (x0 / ATLAS_W, y0 / ATLAS_H, (x1 + 1) / ATLAS_W, (y1 + 1) / ATLAS_H),
                "w": float(w_px),
                "h": float(h_px),
            }

            cur_x += w_px + 8
            row_h = max(row_h, h_px)

        # Place HUD Legend panel cleanly below badge row
        leg_x = 8
        leg_y = cur_y + row_h + 14
        leg_w = 376
        leg_h = 136

        draw.rounded_rectangle(
            [leg_x, leg_y, leg_x + leg_w - 1, leg_y + leg_h - 1],
            radius=8,
            fill=(10, 14, 24, 222),
            outline=(92, 145, 215, 175),
            width=1,
        )

        # Header
        mass_m = self.bh_mass / 1e6
        draw.text(
            (leg_x + 12, leg_y + 9),
            "SGR A* NUCLEAR S-STAR CLUSTER",
            font=font_title,
            fill=(255, 224, 140, 255),
        )
        draw.text(
            (leg_x + 12, leg_y + 27),
            f"M_BH = {mass_m:.2f} x 10^6 Msun  |  Keplerian + 1PN Relativistic",
            font=font_sub,
            fill=(192, 214, 242, 245),
        )
        draw.line(
            [(leg_x + 12, leg_y + 46), (leg_x + leg_w - 12, leg_y + 46)],
            fill=(85, 125, 185, 125),
            width=1,
        )

        # Row 1: S2 Reference Orbit
        r1_y = leg_y + 55
        draw.line([(leg_x + 12, r1_y + 7), (leg_x + 28, r1_y + 7)], fill=(255, 218, 108, 245), width=2)
        draw.polygon(
            [(leg_x + 20, r1_y + 3), (leg_x + 24, r1_y + 7),
             (leg_x + 20, r1_y + 11), (leg_x + 16, r1_y + 7)],
            fill=(255, 226, 130, 255),
        )
        draw.text(
            (leg_x + 36, r1_y - 1),
            "S2 Reference (GRAVITY/Keck: P=16.05y, e=0.884, rp=120 AU)",
            font=font_body,
            fill=(250, 236, 200, 255),
        )

        # Row 2: Other Observationally Identified S-Stars
        r2_y = leg_y + 75
        draw.line([(leg_x + 12, r2_y + 7), (leg_x + 28, r2_y + 7)], fill=(108, 194, 255, 230), width=2)
        draw.ellipse([leg_x + 17, r2_y + 4, leg_x + 23, r2_y + 10], fill=(175, 226, 255, 255))
        draw.text(
            (leg_x + 36, r2_y - 1),
            "Identified S-Stars (S1, S9, S13, S38, S55, S62, S4714)",
            font=font_body,
            fill=(222, 240, 255, 255),
        )

        # Row 3: Representative Synthetic Model Tracers
        r3_y = leg_y + 95
        draw.line([(leg_x + 12, r3_y + 7), (leg_x + 28, r3_y + 7)], fill=(116, 140, 178, 175), width=1)
        draw.ellipse([leg_x + 18, r3_y + 5, leg_x + 22, r3_y + 9], fill=(165, 185, 215, 215))
        draw.text(
            (leg_x + 36, r3_y - 1),
            "Model Synthetic Tracers (28 Keplerian+1PN 3D orbits)",
            font=font_body,
            fill=(176, 194, 222, 240),
        )

        # Footer shortcuts
        draw.text(
            (leg_x + 12, leg_y + 115),
            "[O] Cluster/Orbits   [Y] ID Overlay   [SPACE] Pause   [ / ] Speed",
            font=font_hint,
            fill=(174, 206, 246, 245),
        )

        self._legend_meta = {
            "uv": (
                leg_x / ATLAS_W,
                leg_y / ATLAS_H,
                (leg_x + leg_w) / ATLAS_W,
                (leg_y + leg_h) / ATLAS_H,
            ),
            "w": float(leg_w),
            "h": float(leg_h),
        }

        self.atlas_tex = self.ctx.texture((ATLAS_W, ATLAS_H), 4, img.tobytes())
        self.atlas_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.atlas_tex.repeat_x = False
        self.atlas_tex.repeat_y = False

    # ==================================================================
    # Simulation Physics & GPU Buffer Update
    # ==================================================================

    def update(self, sim_years: float, cam_pos: tuple[float, float, float]) -> None:
        """
        Deterministically compute all 3D Keplerian + 1PN positions, velocities,
        relativistic Doppler / gravitational redshift shifts, and trailing orbit arcs
        at elapsed simulation time `sim_years`.
        """
        t_yr = float(sim_years)

        # 1. Solve Kepler's equation deterministically from simulation time
        M = self.m0 + self.mean_motion * t_yr
        self.mean_anomaly[:] = np.mod(M, 2.0 * math.pi)
        E = _solve_kepler_array(M, self.e)
        cos_E = np.cos(E)
        sin_E = np.sin(E)

        # 2. Leading-order 1PN Schwarzschild periapse precession rotating orbital plane axes
        d_omega = self.d_omega_dt * t_yr
        cos_dw = np.cos(d_omega)[:, None]
        sin_dw = np.sin(d_omega)[:, None]
        P = cos_dw * self.P0 + sin_dw * self.Q0
        Q = -sin_dw * self.P0 + cos_dw * self.Q0

        # 3. Physical position (AU) and orbital velocity (km/s)
        x_orb = self.a_au * (cos_E - self.e)
        y_orb = self.a_au * self.sqrt_1_minus_e2 * sin_E
        r_au = np.maximum(self.a_au * (1.0 - self.e * cos_E), 1e-6)
        self.r_au[:] = r_au

        self.pos_au[:] = x_orb[:, None] * P + y_orb[:, None] * Q

        v_pref = (self.mean_motion * (self.a_au ** 2) / r_au) * AU_PER_YR_TO_KMS
        vx_orb = -v_pref * sin_E
        vy_orb = v_pref * self.sqrt_1_minus_e2 * cos_E
        self.vel_kms[:] = vx_orb[:, None] * P + vy_orb[:, None] * Q
        self.speed_kms[:] = np.linalg.norm(self.vel_kms, axis=1)

        # 4. Monotonic radial mapping into visual Sgr A* local coordinate frame
        r_vis = r_au_to_visual(r_au)
        pos_w = self.pos_au * (r_vis / r_au)[:, None]
        self.pos_world[:] = pos_w.astype(np.float32)

        # 5. Relativistic gravitational redshift + transverse Doppler + radial Doppler
        #    Note: camera -> star unit vector means v_los > 0 corresponds to recession (redshift).
        cam_arr = np.array(cam_pos, dtype=np.float64)[None, :]
        los_vec = pos_w - cam_arr
        los_dist = np.maximum(np.linalg.norm(los_vec, axis=1, keepdims=True), 1e-6)
        los_hat = los_vec / los_dist

        v_los_kms = np.sum(self.vel_kms * los_hat, axis=1)
        beta = np.clip(self.speed_kms / C_KMS, 0.0, 0.65)
        beta_los = np.clip(v_los_kms / C_KMS, -0.65, 0.65)

        gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta ** 2, 1e-4))
        grav_factor = 1.0 / np.sqrt(np.maximum(1.0 - self.rs_au / r_au, 1e-4))
        one_plus_z = grav_factor * gamma * (1.0 + beta_los)
        z_rel = one_plus_z - 1.0
        self.redshift_z[:] = z_rel

        doppler_factor = 1.0 / np.maximum(one_plus_z, 0.35)
        # Subtle, physically restrained Doppler brightening & spectral shift (no cartoon flashing)
        beaming = np.clip(np.power(doppler_factor, 2.2), 0.80, 1.26)
        self.beaming_factor[:] = beaming

        shift = np.clip(-z_rel * 1.8, -0.14, 0.14)  # + -> mild blue shift, - -> mild red shift
        blue_target = np.array([0.82, 0.91, 1.00], dtype=np.float32)
        red_target = np.array([1.00, 0.80, 0.62], dtype=np.float32)

        mod_col = self.star_rgb.copy()
        pos_s = shift > 0.0
        neg_s = ~pos_s
        if np.any(pos_s):
            s_w = shift[pos_s, None].astype(np.float32)
            mod_col[pos_s] = mod_col[pos_s] * (1.0 - s_w) + blue_target[None, :] * s_w
        if np.any(neg_s):
            s_w = (-shift[neg_s, None]).astype(np.float32)
            mod_col[neg_s] = mod_col[neg_s] * (1.0 - s_w) + red_target[None, :] * s_w

        # Populate star VBO buffer and upload once
        self._star_vbo_data[:, 0:3] = self.pos_world
        self._star_vbo_data[:, 3:6] = mod_col
        self._star_vbo_data[:, 7] = self.star_brightness * beaming.astype(np.float32)
        self.star_vbo.orphan(self._star_vbo_data.nbytes)
        self.star_vbo.write(self._star_vbo_data.tobytes())

        # 6. Precessing 3D elliptical orbit trajectory curves with trailing phase wake
        P32 = P.astype(np.float32)
        Q32 = Q.astype(np.float32)
        pts = (
            self.orb_ux[:, :, None] * P32[:, None, :]
            + self.orb_uy[:, :, None] * Q32[:, None, :]
        )  # (N, K, 3)

        # Phase distance behind the star along the orbit (0 right behind star -> 2pi ahead)
        dM = np.mod(self.mean_anomaly[:, None] - self.orb_M, 2.0 * math.pi)
        # Fade synthetic tracer orbit guidelines smoothly when zoomed inside the
        # outer cluster (dist <= 1.15) so the close-up black-hole hero view stays clean.
        cam_dist = float(np.linalg.norm(cam_arr))
        t_sc = min(max((cam_dist - 1.15) / 1.15, 0.0), 1.0)
        synth_close = np.float32(t_sc * t_sc * (3.0 - 2.0 * t_sc))
        tier_scale = np.where(self.is_measured, np.float32(1.0), synth_close)[:, None]

        # Trailing highlight decays smoothly along the backward arc while keeping full ellipse readable
        wake = np.exp(-dM * 0.95).astype(np.float32)
        alpha_pt = (
            self.orb_alpha_min[:, None]
            + (self.orb_alpha_max - self.orb_alpha_min)[:, None] * wake
        ) * tier_scale

        # Slight warm luminance lift right along the trailing arc behind identified stars
        rgb_base = self.orbit_rgb[:, None, :] * (1.0 + 0.30 * wake[:, :, None])
        rgba_pt = np.concatenate([rgb_base, alpha_pt[:, :, None]], axis=2)  # (N, K, 4)

        pts_next = np.roll(pts, -1, axis=1)
        rgba_next = np.roll(rgba_pt, -1, axis=1)

        self._orbit_vbo_data[:, :, 0, 0:3] = pts
        self._orbit_vbo_data[:, :, 0, 3:7] = rgba_pt
        self._orbit_vbo_data[:, :, 1, 0:3] = pts_next
        self._orbit_vbo_data[:, :, 1, 3:7] = rgba_next

        self.orbit_vbo.orphan(self._orbit_vbo_data.nbytes)
        self.orbit_vbo.write(self._orbit_vbo_data.tobytes())

    # ==================================================================
    # Distance Fade Helper
    # ==================================================================

    def get_visibility(self, cam_distance: float) -> float:
        """
        1.0 throughout close BH/Galactic Centre zoom levels, fading smoothly
        to 0.0 before medium/far galaxy scales so no artificial ring appears.
        """
        if cam_distance <= self.fade_start_dist:
            return 1.0
        if cam_distance >= self.fade_end_dist:
            return 0.0
        t = (cam_distance - self.fade_start_dist) / (self.fade_end_dist - self.fade_start_dist)
        return float(1.0 - (t * t * (3.0 - 2.0 * t)))

    # ==================================================================
    # 3D Scene Rendering (Depth-Split Around Sgr A*)
    # ==================================================================

    def render_side(
        self,
        camera,
        aspect_ratio: float,
        window_size: tuple[int, int],
        side_filter: int,
    ) -> None:
        """
        Render S-star 3D elliptical orbits and stars belonging to one depth half-space
        relative to Sgr A* (`side_filter = -1` behind Sgr A* before BH raymarching;
        `side_filter = +1` in front of Sgr A* after BH raymarching).
        """
        if not self.show_cluster:
            return

        vis = self.get_visibility(camera.absolute_distance)
        if vis <= 0.001:
            return

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect_ratio)
        bh_view_pos = view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
        bh_view_depth = float(-bh_view_pos.z)

        _, h = window_size
        fov_h_px = max(float(h), 1.0)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # 1. Orbital Trajectory Lines (clean translucent blending)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.orbit_program["u_view"].write(view_mat)
        self.orbit_program["u_projection"].write(proj_mat)
        self.orbit_program["u_bhViewDepth"].value = bh_view_depth
        self.orbit_program["u_visibility"].value = vis
        self.orbit_program["u_sideFilter"].value = int(side_filter)
        self.orbit_program["u_camPos"].value = camera.get_position()
        d_bh = max(camera.absolute_distance, 0.5)
        self.orbit_program["u_shadowRadius"].value = 0.186 + 0.056 / max(d_bh * d_bh, 0.7)
        self.orbit_vao.render(moderngl.LINES)

        # 2. Moving S-Stars (HDR additive Gaussian PSF)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        self.star_program["u_view"].write(view_mat)
        self.star_program["u_projection"].write(proj_mat)
        self.star_program["u_bhViewDepth"].value = bh_view_depth
        self.star_program["u_visibility"].value = vis
        self.star_program["u_fovHeightPx"].value = fov_h_px
        self.star_program["u_sideFilter"].value = int(side_filter)
        self.star_vao.render(moderngl.POINTS)

        # Restore standard BH pre-multiplied transmittance blend state
        self.ctx.blend_func = (moderngl.ONE, moderngl.SRC_ALPHA)

    # ==================================================================
    # 2D Screen-Space Identification Overlay (`Y` Key)
    # ==================================================================

    @staticmethod
    def _append_quad(
        buf: np.ndarray,
        cursor: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        u0: float,
        v0: float,
        u1: float,
        v1: float,
        rgba: tuple[float, float, float, float],
    ) -> int:
        r, g, b, a = rgba
        buf[cursor + 0] = (x0, y0, u0, v0, r, g, b, a)
        buf[cursor + 1] = (x0, y1, u0, v1, r, g, b, a)
        buf[cursor + 2] = (x1, y0, u1, v0, r, g, b, a)
        buf[cursor + 3] = (x1, y0, u1, v0, r, g, b, a)
        buf[cursor + 4] = (x0, y1, u0, v1, r, g, b, a)
        buf[cursor + 5] = (x1, y1, u1, v1, r, g, b, a)
        return cursor + 6

    @staticmethod
    def _append_segment_quad(
        buf: np.ndarray,
        cursor: int,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        thickness: float,
        rgba: tuple[float, float, float, float],
    ) -> int:
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length < 0.5:
            return cursor
        nx = (-dy / length) * (0.5 * thickness)
        ny = (dx / length) * (0.5 * thickness)
        r, g, b, a = rgba
        buf[cursor + 0] = (ax + nx, ay + ny, -1.0, 0.0, r, g, b, a)
        buf[cursor + 1] = (ax - nx, ay - ny, -1.0, 0.0, r, g, b, a)
        buf[cursor + 2] = (bx + nx, by + ny, -1.0, 0.0, r, g, b, a)
        buf[cursor + 3] = (bx + nx, by + ny, -1.0, 0.0, r, g, b, a)
        buf[cursor + 4] = (ax - nx, ay - ny, -1.0, 0.0, r, g, b, a)
        buf[cursor + 5] = (bx - nx, by - ny, -1.0, 0.0, r, g, b, a)
        return cursor + 6

    @staticmethod
    def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float) -> bool:
        return not (
            a[2] + pad <= b[0]
            or b[2] + pad <= a[0]
            or a[3] + pad <= b[1]
            or b[3] + pad <= a[1]
        )

    @staticmethod
    def _closest_point_on_box(
        px: float, py: float, box: tuple[float, float, float, float]
    ) -> tuple[float, float]:
        cx = min(max(px, box[0]), box[2])
        cy = min(max(py, box[1]), box[3])
        return cx, cy

    def render_overlay(
        self,
        camera,
        aspect_ratio: float,
        window_size: tuple[int, int],
    ) -> None:
        """
        Render screen-space tracking reticles, leader ticks, anti-overlap decluttered
        labels, and the HUD legend (`Y` key), strictly independent of `O`.
        """
        if not self.show_labels:
            return

        vis = self.get_visibility(camera.absolute_distance)
        if vis <= 0.02:
            return

        w, h = float(window_size[0]), float(window_size[1])
        if w < 10.0 or h < 10.0:
            return

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect_ratio)
        vp = proj_mat * view_mat

        # Screen projection and apparent BH shadow exclusion zone (prevents badges covering BH)
        bh_view = view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
        bh_view_depth = float(-bh_view.z)
        bh_clip = vp * glm.vec4(0.0, 0.0, 0.0, 1.0)
        if bh_clip.w > 1e-5:
            bh_sx = (bh_clip.x / bh_clip.w * 0.5 + 0.5) * w
            bh_sy = (0.5 - bh_clip.y / bh_clip.w * 0.5) * h
        else:
            bh_sx, bh_sy = w * 0.5, h * 0.5

        cam_local_r = max(camera.absolute_distance * 12.0, 3.0)
        bh_shadow_ndc_y = 2.0 * 2.6 / cam_local_r
        bh_shadow_px = bh_shadow_ndc_y * 0.5 * h
        bh_shadow_excl = max(bh_shadow_px + 12.0, 26.0)

        buf = self._overlay_vbo_data
        cur = 0

        # 1. Fixed Top-Left Translucent HUD Legend Card
        leg_x0 = 16.0
        leg_y0 = 16.0
        leg_x1 = leg_x0 + self._legend_meta["w"]
        leg_y1 = leg_y0 + self._legend_meta["h"]
        u0, v0, u1, v1 = self._legend_meta["uv"]
        cur = self._append_quad(
            buf, cur, leg_x0, leg_y0, leg_x1, leg_y1, u0, v0, u1, v1,
            (1.0, 1.0, 1.0, 1.0),
        )

        # Collision avoidance boxes (starting with HUD legend)
        occupied_boxes: list[tuple[float, float, float, float]] = [(leg_x0, leg_y0, leg_x1, leg_y1)]
        star_points_px: list[tuple[float, float]] = []

        # 2. Project all 8 observationally identified S-stars in priority order
        projected: list[tuple[int, float, float, float]] = []
        for idx in self.identified_indices:
            px, py, pz = map(float, self.pos_world[idx])
            vpos = view_mat * glm.vec4(px, py, pz, 1.0)
            if vpos.z >= -0.02:
                continue  # behind camera
            clip = proj_mat * vpos
            if clip.w <= 1e-5:
                continue
            sx = (clip.x / clip.w * 0.5 + 0.5) * w
            sy = (0.5 - clip.y / clip.w * 0.5) * h
            if sx < -40.0 or sx > w + 40.0 or sy < -40.0 or sy > h + 40.0:
                continue

            rel_depth = float(-vpos.z) - bh_view_depth
            d_bh = math.hypot(sx - bh_sx, sy - bh_sy)
            # Hide marker/label if star sits strictly behind opaque BH shadow disk
            if rel_depth > 0.0 and d_bh < bh_shadow_px * 0.90:
                continue

            projected.append((idx, sx, sy, rel_depth))
            if 0.0 <= sx <= w and 0.0 <= sy <= h:
                star_points_px.append((sx, sy))

        # 8 compass offset directions; sorted per-star so badges naturally flare outward from Sgr A*
        base_dirs = [
            (math.cos(math.radians(a)), math.sin(math.radians(a)))
            for a in (0, 45, -45, 90, -90, 135, -135, 180)
        ]

        placed_labels: list[tuple[int, float, float, tuple[float, float, float, float], tuple[float, float, float, float], bool]] = []

        for idx, sx, sy, rel_depth in projected:
            spec = self.specs[idx]
            meta = self._badge_meta[idx]
            bw, bh_px = meta["w"], meta["h"]

            if spec.is_reference:
                col_line = (1.00, 0.80, 0.30, 0.95)
            else:
                col_line = (0.48, 0.82, 1.00, 0.88)

            # Dim markers gently when far side transits accretion disk
            if rel_depth > 0.0:
                col_line = (col_line[0], col_line[1], col_line[2], col_line[3] * 0.78)

            # Outward radial preference vector away from Sgr A*
            rx = sx - bh_sx
            ry = sy - bh_sy
            rnorm = math.hypot(rx, ry)
            if rnorm > 1.0:
                ux, uy = rx / rnorm, ry / rnorm
            else:
                ux, uy = 0.707, -0.707

            dirs_sorted = sorted(base_dirs, key=lambda d: -(d[0] * ux + d[1] * uy))
            radii = (22.0, 38.0, 54.0) if spec.is_reference else (20.0, 34.0)

            chosen_box = None
            for r_off in radii:
                for dx, dy in dirs_sorted:
                    cx = sx + dx * r_off
                    cy = sy + dy * r_off
                    bx0 = cx - 0.5 * bw
                    by0 = cy - 0.5 * bh_px
                    bx1 = bx0 + bw
                    by1 = by0 + bh_px

                    # Keep safely inside screen margins
                    if bx0 < 8.0 or by0 < 8.0 or bx1 > w - 8.0 or by1 > h - 8.0:
                        continue

                    cand = (bx0, by0, bx1, by1)

                    # Check BH shadow circle exclusion
                    close_bh_x, close_bh_y = self._closest_point_on_box(bh_sx, bh_sy, cand)
                    if math.hypot(close_bh_x - bh_sx, close_bh_y - bh_sy) < bh_shadow_excl:
                        continue

                    # Check HUD legend & prior placed labels
                    if any(self._boxes_overlap(cand, ob, pad=6.0) for ob in occupied_boxes):
                        continue

                    # Avoid sitting directly on top of another tracked star point
                    hit_star = False
                    for spx, spy in star_points_px:
                        csx, csy = self._closest_point_on_box(spx, spy, cand)
                        if math.hypot(csx - spx, csy - spy) < 7.0:
                            hit_star = True
                            break
                    if hit_star:
                        continue

                    chosen_box = cand
                    break
                if chosen_box is not None:
                    break

            # Draw subtle targeting reticle + core point (if O is OFF) when star is inside viewport
            if 4.0 <= sx <= w - 4.0 and 4.0 <= sy <= h - 4.0:
                r_tick = 7.5 if spec.is_reference else 6.0
                # Open corner reticle diamond / crosshair ticks
                cur = self._append_segment_quad(buf, cur, sx - r_tick, sy, sx - 3.5, sy, 1.3, col_line)
                cur = self._append_segment_quad(buf, cur, sx + 3.5, sy, sx + r_tick, sy, 1.3, col_line)
                cur = self._append_segment_quad(buf, cur, sx, sy - r_tick, sx, sy - 3.5, 1.3, col_line)
                cur = self._append_segment_quad(buf, cur, sx, sy + 3.5, sx, sy + r_tick, 1.3, col_line)

                # Independent pinpoint anchor dot when O (cluster geometry) is hidden
                if not self.show_cluster:
                    dot_r = 2.6 if spec.is_reference else 2.2
                    dot_col = (1.0, 0.92, 0.65, 0.98) if spec.is_reference else (0.80, 0.92, 1.0, 0.95)
                    cur = self._append_quad(
                        buf, cur, sx - dot_r, sy - dot_r, sx + dot_r, sy + dot_r,
                        -1.0, 0.0, -1.0, 0.0, dot_col,
                    )

            if chosen_box is not None:
                occupied_boxes.append(chosen_box)
                placed_labels.append((idx, sx, sy, chosen_box, col_line, rel_depth > 0.0))

        # 3. Emit leader lines and badge quads
        for idx, sx, sy, box, col_line, _ in placed_labels:
            lx, ly = self._closest_point_on_box(sx, sy, box)
            dist_l = math.hypot(lx - sx, ly - sy)
            if dist_l > 8.5:
                gap = 5.5
                ax = sx + ((lx - sx) / dist_l) * gap
                ay = sy + ((ly - sy) / dist_l) * gap
                cur = self._append_segment_quad(
                    buf, cur, ax, ay, lx, ly, 1.1,
                    (col_line[0], col_line[1], col_line[2], col_line[3] * 0.72),
                )

            meta = self._badge_meta[idx]
            bu0, bv0, bu1, bv1 = meta["uv"]
            cur = self._append_quad(
                buf, cur, box[0], box[1], box[2], box[3], bu0, bv0, bu1, bv1,
                (1.0, 1.0, 1.0, 1.0),
            )

        self._overlay_vert_count = cur
        if cur == 0:
            return

        # Grow buffer gracefully if ever needed (preallocated 1536 verts handles far more)
        byte_len = cur * 8 * 4
        self.overlay_vbo.orphan(self._overlay_vbo_data.nbytes)
        self.overlay_vbo.write(self._overlay_vbo_data[:cur].tobytes())

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.atlas_tex.use(location=0)
        self.overlay_program["u_tex"].value = 0
        self.overlay_program["u_viewport"].value = (w, h)
        self.overlay_program["u_alpha"].value = vis

        self.overlay_vao.render(moderngl.TRIANGLES, vertices=cur)

