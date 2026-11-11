"""
Sagittarius A* Nuclear S-Star Cluster, Astrometric Orbits, HUD & Interactive Explorer.

Astrophysically grounded 3D Keplerian + leading-order 1PN Schwarzschild dynamics
anchored strictly to the central supermassive black hole (Sagittarius A*).

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

# Strictly uniform linear spatial conversion from physical AU to the local Sgr A*
# world coordinate frame (1.0 world unit ~ 465.1 AU). A uniform scalar preserves
# exact eccentricity, focus position at (0, 0, 0), orbital plane geometry, and
# relative semi-major axis hierarchy without radial curve distortion, while placing
# S2's pericentre (~118.6 AU -> 0.255 units) clearly outside the visual BH shadow (~0.217).
WORLD_PER_AU = 0.00215


def au_to_world(r_au: np.ndarray | float) -> np.ndarray | float:
    """Uniform linear conversion from AU into local Sgr A* world units."""
    return r_au * WORLD_PER_AU


@dataclass(frozen=True)
class SStarSpec:
    name: str
    catalog_label: str
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


def _solve_kepler_array(mean_anom: np.ndarray, ecc: np.ndarray) -> np.ndarray:
    """
    Vectorized Halley-Newton solver for Kepler's equation M = E - e * sin(E).
    Converges to < 1e-13 rad across all eccentricities up to e = 0.99.
    """
    M = np.mod(mean_anom + math.pi, 2.0 * math.pi) - math.pi
    E = M + ecc * np.sin(M) + 0.5 * (ecc ** 2) * np.sin(2.0 * M)
    high_e = ecc > 0.78
    if np.any(high_e):
        sign_M = np.where(M >= 0.0, 1.0, -1.0)
        E_hi = sign_M * np.power(np.maximum(6.0 * np.abs(M), 1e-12), 1.0 / 3.0)
        E_small = M / np.maximum(1.0 - ecc, 0.02)
        E = np.where(
            high_e & (np.abs(M) < 0.35),
            np.where(np.abs(M) < 0.06, E_small, E_hi),
            E,
        )

    for _ in range(9):
        sin_E = np.sin(E)
        cos_E = np.cos(E)
        f = E - ecc * sin_E - M
        fp = 1.0 - ecc * cos_E
        fpp = ecc * sin_E
        denom = fp - 0.5 * f * fpp / np.maximum(fp, 1e-8)
        step = f / np.where(np.abs(denom) > 1e-10, denom, fp)
        step = np.clip(step, -0.45, 0.45)
        E = E - step
    return E


def _build_identified_catalog() -> list[SStarSpec]:
    """
    Observationally constrained reference S-stars orbiting Sagittarius A*.
    Restrained astronomical color hierarchy:
      - S2 primary reference star in warm gold;
      - Inner short-period companions in crisp cool slate-cyan;
      - Outer / late-type measured members in muted slate-blue / pale ivory.
    """
    return [
        # S2 (S0-2) — Primary relativistic testbed (GRAVITY / Keck)
        SStarSpec(
            name="S2",
            catalog_label="S2 (S0-2)  |  GRAVITY Reference",
            a_arcsec=0.12505,
            e=0.8840,
            inc_deg=134.57,
            omega_node_deg=228.17,
            arg_peri_deg=66.26,
            m0_rad=-0.45,
            star_rgb=(1.00, 0.94, 0.78),
            orbit_rgb=(0.94, 0.72, 0.30),
            point_size=2.35,
            brightness=2.35,
            is_measured=True,
            is_reference=True,
        ),
        # S62 — Ultra-short-period eccentric inner orbit (Peissker et al. 2020)
        SStarSpec(
            name="S62",
            catalog_label="S62  |  Measured Orbit",
            a_arcsec=0.0897,
            e=0.9760,
            inc_deg=72.76,
            omega_node_deg=122.61,
            arg_peri_deg=42.62,
            m0_rad=1.45,
            star_rgb=(0.82, 0.91, 1.00),
            orbit_rgb=(0.36, 0.68, 0.94),
            point_size=1.82,
            brightness=1.90,
            is_measured=True,
        ),
        # S4714 — Extreme eccentricity plunging star (Peissker et al. 2020)
        SStarSpec(
            name="S4714",
            catalog_label="S4714  |  Measured Orbit",
            a_arcsec=0.1020,
            e=0.9850,
            inc_deg=127.70,
            omega_node_deg=129.28,
            arg_peri_deg=357.25,
            m0_rad=-1.75,
            star_rgb=(0.80, 0.89, 1.00),
            orbit_rgb=(0.34, 0.64, 0.90),
            point_size=1.78,
            brightness=1.85,
            is_measured=True,
        ),
        # S55 (S0-102) — Tight 12.8 yr orbit (Meyer et al. / Gillessen et al.)
        SStarSpec(
            name="S55",
            catalog_label="S55 (S0-102)  |  Measured Orbit",
            a_arcsec=0.1078,
            e=0.7209,
            inc_deg=150.10,
            omega_node_deg=325.50,
            arg_peri_deg=331.50,
            m0_rad=2.50,
            star_rgb=(0.84, 0.92, 1.00),
            orbit_rgb=(0.38, 0.66, 0.92),
            point_size=1.75,
            brightness=1.82,
            is_measured=True,
        ),
        # S38 (S0-38) — High-eccentricity retrograde inner star
        SStarSpec(
            name="S38",
            catalog_label="S38 (S0-38)  |  Measured Orbit",
            a_arcsec=0.1416,
            e=0.8201,
            inc_deg=171.10,
            omega_node_deg=101.06,
            arg_peri_deg=17.99,
            m0_rad=-2.35,
            star_rgb=(0.94, 0.95, 1.00),
            orbit_rgb=(0.40, 0.65, 0.90),
            point_size=1.75,
            brightness=1.80,
            is_measured=True,
        ),
        # S13 (S0-20) — Intermediate moderate-eccentricity B-star
        SStarSpec(
            name="S13",
            catalog_label="S13 (S0-20)  |  Measured Orbit",
            a_arcsec=0.2641,
            e=0.4250,
            inc_deg=24.70,
            omega_node_deg=74.50,
            arg_peri_deg=245.20,
            m0_rad=0.68,
            star_rgb=(0.84, 0.91, 1.00),
            orbit_rgb=(0.35, 0.58, 0.84),
            point_size=1.72,
            brightness=1.76,
            is_measured=True,
        ),
        # S9 (S0-5) — Warm / late-type intermediate cluster member
        SStarSpec(
            name="S9",
            catalog_label="S9 (S0-5)  |  Measured Orbit",
            a_arcsec=0.2724,
            e=0.6440,
            inc_deg=82.41,
            omega_node_deg=156.60,
            arg_peri_deg=150.60,
            m0_rad=-0.95,
            star_rgb=(1.00, 0.90, 0.76),
            orbit_rgb=(0.68, 0.62, 0.50),
            point_size=1.75,
            brightness=1.78,
            is_measured=True,
        ),
        # S1 (S0-1) — Luminous outer S-cluster B-star
        SStarSpec(
            name="S1",
            catalog_label="S1 (S0-1)  |  Measured Orbit",
            a_arcsec=0.5950,
            e=0.5560,
            inc_deg=119.14,
            omega_node_deg=342.04,
            arg_peri_deg=122.30,
            m0_rad=1.90,
            star_rgb=(0.80, 0.89, 1.00),
            orbit_rgb=(0.34, 0.56, 0.82),
            point_size=1.84,
            brightness=1.86,
            is_measured=True,
        ),
    ]


def _build_synthetic_tracers(count: int = 28, seed: int = 4207) -> list[SStarSpec]:
    """
    Static seeded population of representative synthetic S-star model tracers
    with thermal eccentricities and isotropic 3D inclinations (explicitly
    distinguished in metadata/legend from observationally measured S-stars).
    """
    rng = np.random.default_rng(seed)
    tracers: list[SStarSpec] = []

    for idx in range(count):
        # Log-uniform semi-major axes spanning ~0.10" to ~0.48" (~800 - 3900 AU)
        a_arcsec = float(np.exp(rng.uniform(math.log(0.105), math.log(0.48))))
        # Sample e in [0.30, 0.88] with thermal weighting while keeping r_p >= 125 AU
        for _ in range(20):
            u = rng.uniform(0.30 ** 2, 0.88 ** 2)
            e = float(math.sqrt(u))
            if a_arcsec * 8200.0 * (1.0 - e) >= 125.0:
                break
        else:
            e = float(rng.uniform(0.30, 0.72))

        cos_i = float(rng.uniform(-1.0, 1.0))
        inc_deg = math.degrees(math.acos(np.clip(cos_i, -1.0, 1.0)))
        omega_node_deg = float(rng.uniform(0.0, 360.0))
        arg_peri_deg = float(rng.uniform(0.0, 360.0))
        m0_rad = float(rng.uniform(-math.pi, math.pi))

        temp_roll = rng.random()
        if temp_roll < 0.65:
            star_rgb = (0.78, 0.88, 1.00)
        elif temp_roll < 0.88:
            star_rgb = (0.94, 0.94, 0.98)
        else:
            star_rgb = (1.00, 0.86, 0.72)

        tracers.append(
            SStarSpec(
                name=f"SYN-{idx + 1:02d}",
                catalog_label=f"SYN-{idx + 1:02d}  |  Synthetic Tracer",
                a_arcsec=a_arcsec,
                e=e,
                inc_deg=inc_deg,
                omega_node_deg=omega_node_deg,
                arg_peri_deg=arg_peri_deg,
                m0_rad=m0_rad,
                star_rgb=star_rgb,
                orbit_rgb=(0.34, 0.42, 0.56),
                point_size=float(rng.uniform(0.92, 1.18)),
                brightness=float(rng.uniform(0.62, 0.92)),
                is_measured=False,
                is_reference=False,
            )
        )

    return tracers


class SStarCluster:
    """
    Sagittarius A* nuclear S-star cluster visualizer & interactive explorer:
    - Exact static Keplerian catalog (8 identified GRAVITY/Keck stars + 28 tracers)
    - Uniform linear spatial scale preserving true ellipses, eccentricity, and
      pericentre/apoapsis hierarchy centered on Sagittarius A*
    - Periapsis-adaptive closed-loop GPU-evaluated ribbon curves (zero per-frame
      CPU orbit geometry rebuild) with subpixel analytical anti-aliasing
    - Strictly independent toggles:
        show_stars (always visible within range)
        show_orbits (`O` key: trajectory curves & orbital guides only)
        show_labels (`Y` key: S-star identification name badges only)
        show_hud_card (`H` key: bottom-left legend & status card)
        show_synthetic_orbits (`G` key: subtle synthetic tracer orbit paths)
    - Interactive star selection, compact telemetry panel, and smooth camera focus
    """

    ORBIT_SEGMENTS = 320

    def __init__(
        self,
        ctx: moderngl.Context,
        base_dir: Path | str,
        bh_mass: float = 4.30e6,
    ):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.bh_mass = float(bh_mass)

        # Independent visibility states (show_stars always stays True in range)
        self.show_stars = True
        self.show_orbits = True             # `O` key: orbital trajectories only
        self.show_labels = False            # `Y` key: individual S-star name badges only
        self.show_hud_card = True           # `H` key: bottom-left HUD info/legend card
        self.show_synthetic_orbits = False  # `G` key: optional synthetic tracer orbits

        # Distance fade calibration (visible in close/GC views, invisible in galaxy/far views)
        self.fade_start_dist = 11.0
        self.fade_end_dist = 17.5

        # Self-consistent Galactic Centre distance R0 (GRAVITY 2019: 8178 pc at 4.154e6 M_sun)
        self.r0_pc = 8178.0 * ((self.bh_mass / 4.154e6) ** (1.0 / 3.0))
        self.au_per_arcsec = self.r0_pc

        # Schwarzschild radius R_s = 2 G M / c^2 in AU
        self.rs_km = 2.0 * self.bh_mass * GM_SUN_OVER_C2_KM
        self.rs_au = self.rs_km / AU_KM

        # Assemble immutable orbital catalog
        self.specs: list[SStarSpec] = _build_identified_catalog() + _build_synthetic_tracers(28, seed=4207)
        self.num_stars = len(self.specs)
        self.identified_indices = [i for i, s in enumerate(self.specs) if s.is_measured]

        # Interactive selection & smooth UI/focus state
        self.selected_idx: int | None = None
        self._display_selected_idx: int | None = None
        self._selection_blend = 0.0
        self._hud_blend = 1.0
        self._synth_blend = 0.0
        self._focus_active = False

        # Live clock status snapshot synced from main.py single clock
        self._sim_years = 0.0
        self._sim_speed = 1.0
        self._sim_paused = False

        # Cached clickable UI bounds in screen pixels for the current frame
        self._last_leg_box: tuple[float, float, float, float] | None = None
        self._last_sel_box: tuple[float, float, float, float] | None = None
        self._last_focus_btn: tuple[float, float, float, float] | None = None
        self._last_center_btn: tuple[float, float, float, float] | None = None
        self._last_close_btn: tuple[float, float, float, float] | None = None

        # Precompute orbital invariants & static closed-loop GPU geometry
        self._init_orbital_arrays()
        self._init_gpu_resources()
        self._build_overlay_atlas()

        self._overlay_vert_count = 0
        self.update(0.0, (0.0, 0.25, 1.0), dt=0.0)

    # ==================================================================
    # Initialization & Static Geometry Precomputation
    # ==================================================================

    def _init_orbital_arrays(self) -> None:
        self.a_au = np.array([s.a_arcsec * self.au_per_arcsec for s in self.specs], dtype=np.float64)
        self.a_world = au_to_world(self.a_au)
        self.e = np.array([s.e for s in self.specs], dtype=np.float64)
        self.inc = np.radians([s.inc_deg for s in self.specs]).astype(np.float64)
        self.omega_node = np.radians([s.omega_node_deg for s in self.specs]).astype(np.float64)
        self.arg_peri0 = np.radians([s.arg_peri_deg for s in self.specs]).astype(np.float64)
        self.m0 = np.array([s.m0_rad for s in self.specs], dtype=np.float64)

        self.rp_au = self.a_au * (1.0 - self.e)
        self.ra_au = self.a_au * (1.0 + self.e)
        self.rp_world = au_to_world(self.rp_au)
        self.ra_world = au_to_world(self.ra_au)

        self.star_rgb = np.array([s.star_rgb for s in self.specs], dtype=np.float32)
        self.orbit_rgb = np.array([s.orbit_rgb for s in self.specs], dtype=np.float32)
        self.star_size = np.array([s.point_size for s in self.specs], dtype=np.float32)
        self.star_brightness = np.array([s.brightness for s in self.specs], dtype=np.float32)
        self.is_measured = np.array([s.is_measured for s in self.specs], dtype=bool)
        self.is_reference = np.array([s.is_reference for s in self.specs], dtype=bool)

        # Kepler's 3rd Law: P [Julian yr] = sqrt(a_AU^3 / M_BH [M_sun])
        self.period_yr = np.sqrt((self.a_au ** 3) / self.bh_mass)
        self.mean_motion = (2.0 * math.pi) / self.period_yr  # rad / yr

        self.sqrt_1_minus_e2 = np.sqrt(1.0 - self.e ** 2)

        # Leading-order Schwarzschild 1PN periapse precession:
        # delta_omega = 3 pi R_s / (a (1 - e^2)) rad / orbit
        self.delta_omega_per_orbit = (3.0 * math.pi * self.rs_au) / (
            self.a_au * (1.0 - self.e ** 2)
        )
        self.d_omega_dt = self.delta_omega_per_orbit / self.period_yr

        # Perifocal unit axes P0 (periapsis direction), Q0 (90 deg ahead in orbital plane)
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
        ], axis=1)

        self.Q0 = np.stack([
            -cO * sw - sO * cw * ci,
            -sO * sw + cO * cw * ci,
            cw * si,
        ], axis=1)

        # Precompute curvature-balanced closed ellipse curve coordinates once!
        # Map uniform psi -> E(psi) = psi - kappa2(e) * sin(2*psi), which
        # concentrates vertices symmetrically at BOTH high-curvature apses
        # (periapsis E=0 and apoapsis E=pi) while keeping sides tight.
        K = self.ORBIT_SEGMENTS
        psi = np.linspace(0.0, 2.0 * math.pi, K, endpoint=False, dtype=np.float64)[None, :]  # (1, K)
        kappa2 = (0.34 * (self.e ** 2))[:, None]  # (N, 1), max slope reduction 0.68 < 1
        E_grid = psi - kappa2 * np.sin(2.0 * psi)  # (N, K)
        cos_Eg = np.cos(E_grid)
        sin_Eg = np.sin(E_grid)

        a_w = self.a_world[:, None]
        e_col = self.e[:, None]
        b_w = a_w * self.sqrt_1_minus_e2[:, None]

        # Exact mathematical Keplerian ellipse in linear world units (focus at 0,0,0)
        ux = (a_w * (cos_Eg - e_col)).astype(np.float32)  # (N, K)
        uy = (b_w * sin_Eg).astype(np.float32)            # (N, K)
        orb_M = np.mod(E_grid - e_col * sin_Eg, 2.0 * math.pi).astype(np.float32)

        ux_prev = np.roll(ux, 1, axis=1)
        uy_prev = np.roll(uy, 1, axis=1)
        ux_next = np.roll(ux, -1, axis=1)
        uy_next = np.roll(uy, -1, axis=1)

        # Append closed-loop seam vertex (index K == index 0) for seamless continuity
        ux_loop = np.concatenate([ux, ux[:, :1]], axis=1)
        uy_loop = np.concatenate([uy, uy[:, :1]], axis=1)
        uxp_loop = np.concatenate([ux_prev, ux_prev[:, :1]], axis=1)
        uyp_loop = np.concatenate([uy_prev, uy_prev[:, :1]], axis=1)
        uxn_loop = np.concatenate([ux_next, ux_next[:, :1]], axis=1)
        uyn_loop = np.concatenate([uy_next, uy_next[:, :1]], axis=1)
        m_loop = np.concatenate([orb_M, orb_M[:, :1]], axis=1)

        # Build static interleaved VBO & IBO for triangle ribbons:
        # 2 vertices per sample (side = -1, +1) -> (N, K+1, 2, 9 floats)
        verts_per_star = K + 1
        static_verts = np.zeros((self.num_stars, verts_per_star, 2, 9), dtype=np.float32)
        star_ids = np.arange(self.num_stars, dtype=np.float32)[:, None, None]
        static_verts[:, :, :, 0] = star_ids
        static_verts[:, :, :, 1] = ux_loop[:, :, None]
        static_verts[:, :, :, 2] = uy_loop[:, :, None]
        static_verts[:, :, :, 3] = uxp_loop[:, :, None]
        static_verts[:, :, :, 4] = uyp_loop[:, :, None]
        static_verts[:, :, :, 5] = uxn_loop[:, :, None]
        static_verts[:, :, :, 6] = uyn_loop[:, :, None]
        static_verts[:, :, :, 7] = m_loop[:, :, None]
        static_verts[:, :, 0, 8] = -1.0
        static_verts[:, :, 1, 8] = 1.0
        self._static_orbit_verts = static_verts.reshape(-1, 9)

        # Build static index buffer connecting quad strip k -> k+1 cleanly
        indices = []
        stride_star = verts_per_star * 2
        k_arr = np.arange(K, dtype=np.uint32)
        for i in range(self.num_stars):
            base = i * stride_star
            v00 = base + 2 * k_arr
            v01 = base + 2 * k_arr + 1
            v10 = base + 2 * (k_arr + 1)
            v11 = base + 2 * (k_arr + 1) + 1
            tri = np.stack([v00, v01, v10, v10, v01, v11], axis=1).reshape(-1)
            indices.append(tri)
        self._static_orbit_indices = np.concatenate(indices)

        # Restrained baseline hierarchy (S2 reference > measured > subtle synthetic tracers)
        self.orb_alpha_min = np.where(
            self.is_reference, 0.19,
            np.where(self.is_measured, 0.068, 0.015),
        ).astype(np.float32)
        self.orb_alpha_max = np.where(
            self.is_reference, 0.56,
            np.where(self.is_measured, 0.27, 0.055),
        ).astype(np.float32)
        self.orb_half_width = np.where(
            self.is_reference, 2.05,
            np.where(self.is_measured, 1.65, 1.30),
        ).astype(np.float32)

        # Live dynamical state arrays
        self.pos_au = np.zeros((self.num_stars, 3), dtype=np.float64)
        self.vel_kms = np.zeros((self.num_stars, 3), dtype=np.float64)
        self.pos_world = np.zeros((self.num_stars, 3), dtype=np.float32)
        self.r_au = np.zeros(self.num_stars, dtype=np.float64)
        self.speed_kms = np.zeros(self.num_stars, dtype=np.float64)
        self.redshift_z = np.zeros(self.num_stars, dtype=np.float64)
        self.beaming_factor = np.ones(self.num_stars, dtype=np.float64)
        self.mean_anomaly = np.zeros(self.num_stars, dtype=np.float64)
        self.true_anomaly = np.zeros(self.num_stars, dtype=np.float64)
        self.P_curr = self.P0.copy()
        self.Q_curr = self.Q0.copy()

        # Reusable dynamic star VBO (3f pos, 3f color, 1f size, 1f brightness, 1f highlight)
        self._star_vbo_data = np.zeros((self.num_stars, 9), dtype=np.float32)
        self._star_vbo_data[:, 6] = self.star_size

        # Preallocated orbit uniform buffers (updated CPU-side without VBO writes!)
        self._u_P = np.zeros((36, 3), dtype=np.float32)
        self._u_Q = np.zeros((36, 3), dtype=np.float32)
        self._u_orbit_color = np.zeros((36, 4), dtype=np.float32)
        self._u_orbit_params = np.zeros((36, 4), dtype=np.float32)

        # Preallocated overlay dynamic buffer (cards + reticles + leaders + badges + live text glyphs)
        self._overlay_max_verts = 640 * 6
        self._overlay_vbo_data = np.zeros((self._overlay_max_verts, 8), dtype=np.float32)

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
            [(
                self.star_vbo,
                "3f 3f 1f 1f 1f",
                "in_position", "in_color", "in_size", "in_brightness", "in_highlight",
            )],
        )

        # Static VBO + IBO uploaded once at init (zero per-frame orbit vertex upload!)
        self.orbit_vbo = self.ctx.buffer(self._static_orbit_verts.tobytes())
        self.orbit_ibo = self.ctx.buffer(self._static_orbit_indices.tobytes())
        self.orbit_vao = self.ctx.vertex_array(
            self.orbit_program,
            [(
                self.orbit_vbo,
                "1f 2f 2f 2f 1f 1f",
                "in_starIdx", "in_currUV", "in_prevUV", "in_nextUV", "in_orbM", "in_side",
            )],
            index_buffer=self.orbit_ibo,
        )

        self.overlay_vbo = self.ctx.buffer(reserve=self._overlay_vbo_data.nbytes, dynamic=True)
        self.overlay_vao = self.ctx.vertex_array(
            self.overlay_program,
            [(self.overlay_vbo, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
        )

    def _build_overlay_atlas(self) -> None:
        """
        Pre-render one unified 1024x1024 RGBA texture atlas containing:
          - Identification name pill badges + rp/ra apse tags
          - Pure-white alpha-masked UI vector sprites (reticle ring, directional chevron, diamond)
          - Bottom-left Legend & Clock dark-glass card base
          - Compact Selected Star Telemetry panel template + action button pills
          - Crisp kerned ASCII glyph sheets for zero-allocation dynamic text
        """
        ATLAS_W, ATLAS_H = 1024, 1024
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
        semi_fonts = [
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        reg_fonts = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

        font_badge = load_font(bold_fonts, 13)
        font_title = load_font(bold_fonts, 14)
        font_sub = load_font(reg_fonts, 11)
        font_body = load_font(reg_fonts, 12)
        font_hint = load_font(semi_fonts, 11)
        font_dyn_lg = load_font(bold_fonts, 15)
        font_dyn_val = load_font(semi_fonts, 12)
        font_dyn_sm = load_font(bold_fonts, 11)

        # ------------------------------------------------------------------
        # 1. Identified S-Star Badges + Apse Tags (y = 8..44)
        # ------------------------------------------------------------------
        self._badge_meta: dict[int, dict] = {}
        cur_x, cur_y, row_h = 8, 8, 0

        for idx in self.identified_indices:
            spec = self.specs[idx]
            text = spec.name
            bbox = draw.textbbox((0, 0), text, font=font_badge)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            dot_pad = 13 if spec.is_reference else 11
            w_px = dot_pad + text_w + 10
            h_px = max(20, text_h + 9)

            if cur_x + w_px + 8 > ATLAS_W:
                cur_x = 8
                cur_y += row_h + 8
                row_h = 0

            x0, y0 = cur_x, cur_y
            x1, y1 = x0 + w_px - 1, y0 + h_px - 1

            if spec.is_reference:
                bg = (14, 18, 30, 224)
                border = (245, 192, 74, 240)
                text_col = (255, 242, 206, 255)
                dot_col = (255, 200, 74, 255)
            else:
                bg = (10, 16, 28, 212)
                border = (102, 184, 248, 215)
                text_col = (232, 245, 255, 255)
                dot_col = (114, 198, 255, 255)

            draw.rounded_rectangle([x0, y0, x1, y1], radius=5, fill=bg, outline=border, width=1)
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

        # Tiny subtle apse pills ("Periapsis" / "Apoapsis" compact "q" / "Q")
        self._tag_meta: dict[str, dict] = {}
        for tag_key, tag_txt, fg, bd in [
            ("peri", "rp", (255, 224, 148, 255), (225, 178, 72, 210)),
            ("apo",  "ra", (184, 218, 250, 245), (110, 165, 220, 195)),
        ]:
            tb = draw.textbbox((0, 0), tag_txt, font=font_dyn_sm)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            bw, bh_t = tw + 8, max(16, th + 6)
            bx0, by0 = cur_x, cur_y
            bx1, by1 = bx0 + bw - 1, by0 + bh_t - 1
            draw.rounded_rectangle([bx0, by0, bx1, by1], radius=3, fill=(10, 15, 26, 215), outline=bd, width=1)
            draw.text((bx0 + 4 - tb[0], by0 + (bh_t - th) // 2 - tb[1]), tag_txt, font=font_dyn_sm, fill=fg)
            self._tag_meta[tag_key] = {
                "uv": (bx0 / ATLAS_W, by0 / ATLAS_H, (bx1 + 1) / ATLAS_W, (by1 + 1) / ATLAS_H),
                "w": float(bw),
                "h": float(bh_t),
            }
            cur_x += bw + 8

        # ------------------------------------------------------------------
        # 2. Pure-White Alpha-Masked SDF UI Vector Sprites (y = 52..100)
        #    RGB=(255,255,255) everywhere prevents bilinear dark fringes.
        # ------------------------------------------------------------------
        self._sprite_meta: dict[str, dict] = {}

        def pack_alpha_sprite(key: str, ax: int, ay: int, alpha_mask: np.ndarray):
            sh, sw = alpha_mask.shape
            rgba = np.empty((sh, sw, 4), dtype=np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = alpha_mask
            img.paste(Image.fromarray(rgba, "RGBA"), (ax, ay))
            self._sprite_meta[key] = {
                "uv": (ax / ATLAS_W, ay / ATLAS_H, (ax + sw) / ATLAS_W, (ay + sh) / ATLAS_H),
                "w": float(sw),
                "h": float(sh),
            }

        # Selection ring reticle (32x32)
        S = 32
        yy, xx = np.mgrid[0:S, 0:S]
        rr = np.hypot(xx - (S - 1) * 0.5, yy - (S - 1) * 0.5)
        ring_a = np.clip(1.0 - np.abs(rr - 11.2) / 1.35, 0.0, 1.0)
        pack_alpha_sprite("ring", 8, 52, (ring_a * 255.0).astype(np.uint8))

        # Orbital direction chevron (18x18, pointing right +X)
        C_SZ = 18
        cy_g, cx_g = np.mgrid[0:C_SZ, 0:C_SZ]
        px_c = cx_g - (C_SZ - 1) * 0.5
        py_c = cy_g - (C_SZ - 1) * 0.5
        # Chevron arms: y = +- 0.9 * (3.6 - x) for x in [-2.6, 3.8]
        tip_x = 3.6
        d_arm = np.abs(np.abs(py_c) - 0.92 * (tip_x - px_c)) / math.hypot(1.0, 0.92)
        in_x = np.clip((px_c + 2.8) * 1.4, 0.0, 1.0) * np.clip((tip_x + 0.8 - px_c) * 1.6, 0.0, 1.0)
        chev_a = np.clip(1.0 - (d_arm - 0.55) / 0.95, 0.0, 1.0) * in_x
        pack_alpha_sprite("chevron", 52, 52, (chev_a * 255.0).astype(np.uint8))

        # Hollow diamond for periapsis (16x16)
        D_SZ = 16
        dy_g, dx_g = np.mgrid[0:D_SZ, 0:D_SZ]
        l1 = np.abs(dx_g - (D_SZ - 1) * 0.5) + np.abs(dy_g - (D_SZ - 1) * 0.5)
        diam_a = np.clip(1.0 - np.abs(l1 - 4.4) / 1.15, 0.0, 1.0)
        pack_alpha_sprite("diamond", 82, 52, (diam_a * 255.0).astype(np.uint8))

        # Subtle hollow square for apoapsis (16x16)
        sq = np.maximum(np.abs(dx_g - (D_SZ - 1) * 0.5), np.abs(dy_g - (D_SZ - 1) * 0.5))
        sq_a = np.clip(1.0 - np.abs(sq - 3.8) / 1.10, 0.0, 1.0)
        pack_alpha_sprite("square", 108, 52, (sq_a * 255.0).astype(np.uint8))

        # Solid soft-AA pill / filled round swatch (16x16)
        dot_r = np.hypot(dx_g - (D_SZ - 1) * 0.5, dy_g - (D_SZ - 1) * 0.5)
        dot_a = np.clip(3.8 - dot_r, 0.0, 1.0)
        pack_alpha_sprite("dot", 134, 52, (dot_a * 255.0).astype(np.uint8))

        # Rounded solid pill button background (16x16 sliceable / uniform tint)
        px_b = np.maximum(np.abs(dx_g - (D_SZ - 1) * 0.5) - 4.0, 0.0)
        py_b = np.maximum(np.abs(dy_g - (D_SZ - 1) * 0.5) - 4.0, 0.0)
        box_a = np.clip(3.2 - np.hypot(px_b, py_b), 0.0, 1.0)
        pack_alpha_sprite("pill", 160, 52, (box_a * 255.0).astype(np.uint8))

        def draw_dark_glass_card(cx: int, cy: int, cw: int, ch: int, hdr_h: int, border_rgba: tuple[int, int, int, int]) -> None:
            draw.rounded_rectangle(
                [cx, cy, cx + cw - 1, cy + ch - 1],
                radius=8,
                fill=(10, 14, 25, 236),
            )
            hdr_mask = Image.new("L", (cw, hdr_h), 0)
            hdr_draw = ImageDraw.Draw(hdr_mask)
            hdr_draw.rounded_rectangle([1, 1, cw - 2, hdr_h + 12], radius=7, fill=255)
            hdr_fill = Image.new("RGBA", (cw, hdr_h), (18, 28, 48, 242))
            img.paste(hdr_fill, (cx, cy), hdr_mask)
            draw.rounded_rectangle(
                [cx, cy, cx + cw - 1, cy + ch - 1],
                radius=8,
                outline=border_rgba,
                width=1,
            )

        # ------------------------------------------------------------------
        # 3. Bottom-Left Legend & Clock Card (y = 104..294)
        # ------------------------------------------------------------------
        leg_x, leg_y = 8, 104
        leg_w, leg_h = 398, 184

        draw_dark_glass_card(leg_x, leg_y, leg_w, leg_h, 47, (92, 148, 220, 186))
        mass_m = self.bh_mass / 1e6
        draw.text((leg_x + 13, leg_y + 8), "SGR A* NUCLEAR S-STAR CLUSTER", font=font_title, fill=(255, 224, 140, 255))
        draw.text(
            (leg_x + 13, leg_y + 27),
            f"M_BH = {mass_m:.2f} x 10^6 Msun   |   Keplerian + 1PN Relativistic",
            font=font_sub,
            fill=(192, 216, 244, 255),
        )
        draw.line([(leg_x + 12, leg_y + 47), (leg_x + leg_w - 12, leg_y + 47)], fill=(86, 126, 186, 125), width=1)

        # Row 53..74 reserved for dynamic Year / Speed / RUNNING-PAUSED status strip
        draw.line([(leg_x + 12, leg_y + 76), (leg_x + leg_w - 12, leg_y + 76)], fill=(86, 126, 186, 118), width=1)

        # Population hierarchy swatches
        r1_y = leg_y + 84
        draw.line([(leg_x + 13, r1_y + 7), (leg_x + 29, r1_y + 7)], fill=(242, 190, 82, 245), width=2)
        draw.polygon([(leg_x + 21, r1_y + 3), (leg_x + 25, r1_y + 7), (leg_x + 21, r1_y + 11), (leg_x + 17, r1_y + 7)], fill=(255, 226, 130, 255))
        draw.text((leg_x + 37, r1_y - 1), "S2 Reference (GRAVITY/Keck: P=16.05y, e=0.884)", font=font_body, fill=(250, 236, 200, 255))

        r2_y = leg_y + 104
        draw.line([(leg_x + 13, r2_y + 7), (leg_x + 29, r2_y + 7)], fill=(102, 186, 248, 235), width=2)
        draw.ellipse([leg_x + 18, r2_y + 4, leg_x + 24, r2_y + 10], fill=(176, 226, 255, 255))
        draw.text((leg_x + 37, r2_y - 1), "Identified S-Stars (S1, S9, S13, S38, S55, S62, S4714)", font=font_body, fill=(222, 240, 255, 255))

        r3_y = leg_y + 124
        draw.line([(leg_x + 13, r3_y + 7), (leg_x + 29, r3_y + 7)], fill=(104, 126, 162, 175), width=1)
        draw.ellipse([leg_x + 19, r3_y + 5, leg_x + 23, r3_y + 9], fill=(166, 186, 214, 215))
        draw.text((leg_x + 37, r3_y - 1), "Synthetic Model Tracers (28 Keplerian 3D; [G] orbits)", font=font_body, fill=(174, 194, 222, 238))

        draw.line([(leg_x + 12, leg_y + 144), (leg_x + leg_w - 12, leg_y + 144)], fill=(86, 126, 186, 100), width=1)
        draw.text(
            (leg_x + 13, leg_y + 149),
            "[O] Orbits  [Y] Badges  [H] HUD  [G] Synth  [SPACE] Pause",
            font=font_hint,
            fill=(166, 202, 242, 240),
        )
        draw.text(
            (leg_x + 13, leg_y + 165),
            "Click Star: Inspect   Dbl-Click / [F]: Focus   [C]: Center Sgr A*",
            font=font_hint,
            fill=(152, 186, 226, 228),
        )

        self._legend_meta = {
            "uv": (leg_x / ATLAS_W, leg_y / ATLAS_H, (leg_x + leg_w) / ATLAS_W, (leg_y + leg_h) / ATLAS_H),
            "w": float(leg_w),
            "h": float(leg_h),
        }

        # ------------------------------------------------------------------
        # 4. Compact Selected Star Info Panel Template (x = 424..750, y = 104..358)
        # ------------------------------------------------------------------
        sel_x, sel_y = 424, 104
        sel_w, sel_h = 326, 254

        draw_dark_glass_card(sel_x, sel_y, sel_w, sel_h, 50, (104, 174, 245, 204))
        draw.line([(sel_x + 12, sel_y + 50), (sel_x + sel_w - 12, sel_y + 50)], fill=(92, 140, 205, 135), width=1)

        # Close [X] micro-button top-right
        cx_btn = sel_x + sel_w - 22
        cy_btn = sel_y + 16
        draw.rounded_rectangle([cx_btn - 8, cy_btn - 8, cx_btn + 8, cy_btn + 8], radius=4, fill=(28, 38, 60, 185), outline=(120, 165, 220, 160), width=1)
        draw.line([(cx_btn - 4, cy_btn - 4), (cx_btn + 4, cy_btn + 4)], fill=(215, 232, 250, 235), width=1)
        draw.line([(cx_btn - 4, cy_btn + 4), (cx_btn + 4, cy_btn - 4)], fill=(215, 232, 250, 235), width=1)

        # Static left-column labels
        metric_labels = [
            "Semi-Major Axis (a)",
            "Eccentricity (e)",
            "Orbital Period (P)",
            "Inclination (i)",
            "Current Radius (r)",
            "Orbital Speed (v)",
            "Peri / Apo (rp / ra)",
            "Orbital Phase (M)",
        ]
        for row_i, lbl in enumerate(metric_labels):
            ry = sel_y + 58 + row_i * 19
            draw.text((sel_x + 14, ry), lbl, font=font_body, fill=(144, 174, 212, 238))

        draw.line([(sel_x + 12, sel_y + 215), (sel_x + sel_w - 12, sel_y + 215)], fill=(86, 126, 186, 108), width=1)

        # Action button templates at bottom of Selected Panel
        # Left button: Focus Star [12..166], Right button: Center Sgr A* [174..314]
        draw.rounded_rectangle(
            [sel_x + 12, sel_y + 223, sel_x + 166, sel_y + 245],
            radius=5,
            fill=(24, 42, 72, 215),
            outline=(104, 180, 245, 190),
            width=1,
        )
        draw.rounded_rectangle(
            [sel_x + 174, sel_y + 223, sel_x + sel_w - 12, sel_y + 245],
            radius=5,
            fill=(20, 30, 52, 210),
            outline=(104, 148, 206, 170),
            width=1,
        )

        self._sel_panel_meta = {
            "uv": (sel_x / ATLAS_W, sel_y / ATLAS_H, (sel_x + sel_w) / ATLAS_W, (sel_y + sel_h) / ATLAS_H),
            "w": float(sel_w),
            "h": float(sel_h),
        }

        # ------------------------------------------------------------------
        # 5. White Alpha-Masked ASCII + Degree Glyph Sheets (y = 380..580)
        # ------------------------------------------------------------------
        charset = [chr(c) for c in range(32, 127)] + ["\u00b0"]
        self._fonts_meta: dict[str, dict] = {}

        sheet_cursor_y = 380
        for font_key, fnt, cell_pad in [
            ("lg", font_dyn_lg, 2),
            ("val", font_dyn_val, 2),
            ("sm", font_dyn_sm, 2),
        ]:
            # Render onto a single-channel L image then paste as white + alpha
            ascent, descent = fnt.getmetrics()
            line_h = ascent + descent + 4
            sheet_x = 8
            sheet_y = sheet_cursor_y
            gx = sheet_x
            gy = sheet_y
            glyphs: dict[str, dict] = {}

            for ch in charset:
                adv = float(draw.textlength(ch, font=fnt))
                bb = fnt.getbbox(ch)
                if bb is None:
                    bb = (0, 0, max(1, int(math.ceil(adv))), 1)
                gw = max(1, bb[2] - bb[0] + cell_pad * 2)
                gh = max(1, line_h)
                if gx + gw + 4 > ATLAS_W:
                    gx = 8
                    gy += line_h + 4

                mask_img = Image.new("L", (gw, gh), 0)
                mdraw = ImageDraw.Draw(mask_img)
                mdraw.text((cell_pad - bb[0], 1), ch, font=fnt, fill=255)
                mask_arr = np.asarray(mask_img, dtype=np.uint8)
                rgba_arr = np.empty((gh, gw, 4), dtype=np.uint8)
                rgba_arr[..., :3] = 255
                rgba_arr[..., 3] = mask_arr
                img.paste(Image.fromarray(rgba_arr, "RGBA"), (gx, gy))

                glyphs[ch] = {
                    "uv": (gx / ATLAS_W, gy / ATLAS_H, (gx + gw) / ATLAS_W, (gy + gh) / ATLAS_H),
                    "w": float(gw),
                    "h": float(gh),
                    "xoff": float(bb[0] - cell_pad),
                    "adv": float(adv),
                }
                gx += gw + 3

            self._fonts_meta[font_key] = {
                "glyphs": glyphs,
                "line_h": float(line_h),
            }
            sheet_cursor_y = gy + line_h + 10

        self.atlas_tex = self.ctx.texture((ATLAS_W, ATLAS_H), 4, img.tobytes())
        self.atlas_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.atlas_tex.repeat_x = False
        self.atlas_tex.repeat_y = False

    # ==================================================================
    # Independent State Control Helpers & Selection API
    # ==================================================================

    def toggle_orbits(self) -> bool:
        self.show_orbits = not self.show_orbits
        return self.show_orbits

    def toggle_labels(self) -> bool:
        self.show_labels = not self.show_labels
        return self.show_labels

    def toggle_hud_card(self) -> bool:
        self.show_hud_card = not self.show_hud_card
        return self.show_hud_card

    def toggle_synthetic_orbits(self) -> bool:
        self.show_synthetic_orbits = not self.show_synthetic_orbits
        return self.show_synthetic_orbits

    def set_clock_status(self, sim_years: float, sim_speed: float, sim_paused: bool) -> None:
        self._sim_years = float(sim_years)
        self._sim_speed = float(sim_speed)
        self._sim_paused = bool(sim_paused)

    def set_focus_active(self, active: bool) -> None:
        self._focus_active = bool(active)

    def select_star(self, idx: int | None) -> None:
        if idx is None or not (0 <= idx < self.num_stars):
            self.selected_idx = None
        else:
            self.selected_idx = int(idx)
            self._display_selected_idx = int(idx)

    def get_selected_position(self) -> tuple[float, float, float] | None:
        if self.selected_idx is None:
            return None
        p = self.pos_world[self.selected_idx]
        return (float(p[0]), float(p[1]), float(p[2]))

    def get_selected_info(self) -> dict | None:
        if self.selected_idx is None:
            return None
        idx = self.selected_idx
        spec = self.specs[idx]
        return {
            "index": idx,
            "name": spec.name,
            "catalog_label": spec.catalog_label,
            "is_measured": spec.is_measured,
            "is_reference": spec.is_reference,
            "a_arcsec": spec.a_arcsec,
            "a_au": float(self.a_au[idx]),
            "e": float(self.e[idx]),
            "period_yr": float(self.period_yr[idx]),
            "inc_deg": float(spec.inc_deg),
            "r_au": float(self.r_au[idx]),
            "speed_kms": float(self.speed_kms[idx]),
            "rp_au": float(self.rp_au[idx]),
            "ra_au": float(self.ra_au[idx]),
            "mean_anom_deg": float(math.degrees(self.mean_anomaly[idx]) % 360.0),
            "true_anom_deg": float(math.degrees(self.true_anomaly[idx]) % 360.0),
        }

    def pick_star(
        self,
        mx: float,
        my: float,
        camera,
        aspect_ratio: float,
        window_size: tuple[int, int],
        hit_radius_px: float = 18.0,
    ) -> int | None:
        """
        Screen-space click picking of individual S-stars. Only picks stars
        that are visible and not eclipsed behind the opaque Sgr A* shadow.
        """
        vis = self.get_visibility(camera.absolute_distance)
        if vis <= 0.02 or not self.show_stars:
            return None

        w, h = float(window_size[0]), float(window_size[1])
        if w < 10.0 or h < 10.0:
            return None

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect_ratio)
        vp = proj_mat * view_mat

        bh_view = view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
        bh_view_depth = float(-bh_view.z)
        bh_clip = vp * glm.vec4(0.0, 0.0, 0.0, 1.0)
        if bh_clip.w > 1e-5:
            bh_sx = (bh_clip.x / bh_clip.w * 0.5 + 0.5) * w
            bh_sy = (0.5 - bh_clip.y / bh_clip.w * 0.5) * h
        else:
            bh_sx, bh_sy = w * 0.5, h * 0.5

        cam_local_r = max(float(np.linalg.norm(camera.get_position())) * 12.0, 3.0)
        bh_shadow_px = (2.0 * 2.6 / cam_local_r) * 0.5 * h

        best_idx = None
        best_score = 1e9

        for idx in range(self.num_stars):
            px, py, pz = map(float, self.pos_world[idx])
            vpos = view_mat * glm.vec4(px, py, pz, 1.0)
            if vpos.z >= -0.02:
                continue
            clip = proj_mat * vpos
            if clip.w <= 1e-5:
                continue
            sx = (clip.x / clip.w * 0.5 + 0.5) * w
            sy = (0.5 - clip.y / clip.w * 0.5) * h
            if sx < -20.0 or sx > w + 20.0 or sy < -20.0 or sy > h + 20.0:
                continue

            rel_depth = float(-vpos.z) - bh_view_depth
            d_bh = math.hypot(sx - bh_sx, sy - bh_sy)
            if rel_depth > 0.0 and d_bh < bh_shadow_px * 0.90:
                continue

            dist_px = math.hypot(sx - mx, sy - my)
            allowed = hit_radius_px if self.is_measured[idx] else hit_radius_px * 0.78
            if dist_px <= allowed:
                # Gently prefer observationally measured / closer candidates
                score = dist_px - (4.0 if self.is_measured[idx] else 0.0)
                if score < best_score:
                    best_score = score
                    best_idx = idx

        return best_idx

    def handle_overlay_click(self, mx: float, my: float) -> str | None:
        """
        Hit-test HUD card & action pills rendered on the previous frame.
        Returns:
          - 'focus'  : clicked Focus button on Selected Star panel
          - 'center' : clicked Center Sgr A* button on Selected Star panel
          - 'close'  : clicked [X] close icon on Selected Star panel
          - 'consume': clicked somewhere inside a visible HUD panel
          - None     : click landed outside HUD cards
        """
        def in_box(bx: tuple[float, float, float, float] | None) -> bool:
            if bx is None:
                return False
            return bx[0] <= mx <= bx[2] and bx[1] <= my <= bx[3]

        if in_box(self._last_close_btn):
            return "close"
        if in_box(self._last_focus_btn):
            return "focus"
        if in_box(self._last_center_btn):
            return "center"
        if in_box(self._last_sel_box) or in_box(self._last_leg_box):
            return "consume"
        return None

    # ==================================================================
    # Simulation Physics & GPU Buffer Update
    # ==================================================================

    def compute_positions(self, sim_years: float) -> None:
        """
        Deterministically solve Kepler's equation and Schwarzschild 1PN precession
        for elapsed simulation time `sim_years`. Called prior to camera follow updates.
        """
        t_yr = float(sim_years)
        self._sim_years = t_yr

        M = self.m0 + self.mean_motion * t_yr
        self.mean_anomaly[:] = np.mod(M, 2.0 * math.pi)
        E = _solve_kepler_array(M, self.e)
        cos_E = np.cos(E)
        sin_E = np.sin(E)

        # True anomaly nu from E
        self.true_anomaly[:] = np.mod(
            2.0 * np.arctan2(
                np.sqrt(1.0 + self.e) * np.sin(0.5 * E),
                np.sqrt(1.0 - self.e) * np.cos(0.5 * E),
            ),
            2.0 * math.pi,
        )

        # Leading-order Schwarzschild 1PN periapse precession
        d_omega = self.d_omega_dt * t_yr
        cos_dw = np.cos(d_omega)[:, None]
        sin_dw = np.sin(d_omega)[:, None]
        P = cos_dw * self.P0 + sin_dw * self.Q0
        Q = -sin_dw * self.P0 + cos_dw * self.Q0
        self.P_curr[:] = P
        self.Q_curr[:] = Q

        # Physical position (AU) and orbital velocity (km/s)
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

        # Exact uniform linear spatial transformation (zero radial distortion)
        self.pos_world[:] = au_to_world(self.pos_au).astype(np.float32)

    def update(
        self,
        sim_years: float,
        cam_pos: tuple[float, float, float],
        dt: float = 0.0,
        recompute_orbits: bool = True,
    ) -> None:
        """
        Update dynamical orbital state, smooth HUD/selection transitions,
        relativistic beaming/redshift tints, and lightweight GPU buffers.
        """
        if recompute_orbits:
            self.compute_positions(sim_years)

        # Smooth fade transitions
        if dt > 0.0:
            blend_step = min(10.0 * dt, 1.0)
            sel_goal = 1.0 if self.selected_idx is not None else 0.0
            self._selection_blend += (sel_goal - self._selection_blend) * blend_step
            if self._selection_blend < 0.005 and self.selected_idx is None:
                self._selection_blend = 0.0
                self._display_selected_idx = None

            hud_goal = 1.0 if self.show_hud_card else 0.0
            self._hud_blend += (hud_goal - self._hud_blend) * blend_step

            synth_goal = 1.0 if self.show_synthetic_orbits else 0.0
            self._synth_blend += (synth_goal - self._synth_blend) * min(8.0 * dt, 1.0)
        else:
            self._selection_blend = 1.0 if self.selected_idx is not None else 0.0
            if self.selected_idx is None:
                self._display_selected_idx = None
            self._hud_blend = 1.0 if self.show_hud_card else 0.0
            self._synth_blend = 1.0 if self.show_synthetic_orbits else 0.0

        # Relativistic gravitational redshift + transverse Doppler + radial Doppler
        pos_w = self.pos_au * WORLD_PER_AU
        cam_arr = np.array(cam_pos, dtype=np.float64)[None, :]
        los_vec = pos_w - cam_arr
        los_dist = np.maximum(np.linalg.norm(los_vec, axis=1, keepdims=True), 1e-6)
        los_hat = los_vec / los_dist

        v_los_kms = np.sum(self.vel_kms * los_hat, axis=1)
        beta = np.clip(self.speed_kms / C_KMS, 0.0, 0.65)
        beta_los = np.clip(v_los_kms / C_KMS, -0.65, 0.65)

        gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta ** 2, 1e-4))
        grav_factor = 1.0 / np.sqrt(np.maximum(1.0 - self.rs_au / self.r_au, 1e-4))
        one_plus_z = grav_factor * gamma * (1.0 + beta_los)
        z_rel = one_plus_z - 1.0
        self.redshift_z[:] = z_rel

        doppler_factor = 1.0 / np.maximum(one_plus_z, 0.35)
        beaming = np.clip(np.power(doppler_factor, 2.2), 0.80, 1.26)
        self.beaming_factor[:] = beaming

        shift = np.clip(-z_rel * 1.8, -0.14, 0.14)
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

        # Populate compact dynamic star VBO
        hi_arr = np.zeros(self.num_stars, dtype=np.float32)
        if self._display_selected_idx is not None:
            hi_arr[self._display_selected_idx] = float(self._selection_blend)

        self._star_vbo_data[:, 0:3] = self.pos_world
        self._star_vbo_data[:, 3:6] = mod_col
        self._star_vbo_data[:, 7] = self.star_brightness * beaming.astype(np.float32)
        self._star_vbo_data[:, 8] = hi_arr
        self.star_vbo.orphan(self._star_vbo_data.nbytes)
        self.star_vbo.write(self._star_vbo_data.tobytes())

        # Populate compact per-orbit uniform arrays (36 entries; zero vertex rebuild!)
        self._u_P[:self.num_stars] = self.P_curr.astype(np.float32)
        self._u_Q[:self.num_stars] = self.Q_curr.astype(np.float32)

        sel_w = float(self._selection_blend)
        active_sel = self._display_selected_idx
        dim_others = 1.0 - 0.58 * sel_w  # gentle fade on non-selected orbits

        for i in range(self.num_stars):
            is_sel = (active_sel == i)
            base_vis = 1.0 if self.is_measured[i] else float(self._synth_blend)
            if is_sel:
                tier_scale = max(base_vis, sel_w) * (1.0 + 0.30 * sel_w)
            else:
                tier_scale = base_vis * dim_others

            rgb = self.orbit_rgb[i]
            if is_sel:
                # Crisp restrained lift on selected trajectory
                hi_rgb = np.array([1.00, 0.88, 0.54], dtype=np.float32) if self.is_reference[i] else np.array([0.62, 0.88, 1.00], dtype=np.float32)
                use_rgb = rgb * (1.0 - sel_w) + hi_rgb * sel_w
            else:
                use_rgb = rgb

            self._u_orbit_color[i, 0:3] = use_rgb
            self._u_orbit_color[i, 3] = float(tier_scale)

            hw_px = float(self.orb_half_width[i])
            amin = float(self.orb_alpha_min[i])
            amax = float(self.orb_alpha_max[i])
            if is_sel:
                hw_px = hw_px * (1.0 - sel_w) + 2.45 * sel_w
                amin = amin * (1.0 - sel_w) + 0.34 * sel_w
                amax = amax * (1.0 - sel_w) + 0.78 * sel_w

            self._u_orbit_params[i] = (
                float(self.mean_anomaly[i]),
                hw_px,
                amin,
                amax,
            )

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

    def get_hud_visibility(self, camera) -> float:
        """
        Show the bottom-left HUD Legend Card only in the viewing range where
        S-stars are clearly visible around Sagittarius A*:
          - Hidden during ultra-close BH accretion-disk inspection (dist <= 1.28,
            including startup dist = 1.0)
          - Full visibility across dist = 1.58..9.0 near Sgr A*
          - Smoothly hidden by dist >= 11.5 (when 3D bulge drowns the cluster)
            or when panned far away from Sgr A*
        """
        if not self.show_stars:
            return 0.0
        d = float(camera.absolute_distance)
        if d <= 1.28 or d >= 11.5:
            return 0.0

        def smoothstep(e0: float, e1: float, x: float) -> float:
            t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
            return t * t * (3.0 - 2.0 * t)

        near_in = smoothstep(1.28, 1.58, d)
        far_out = 1.0 - smoothstep(9.0, 11.5, d)
        cam_pos = camera.get_position()
        cam_to_bh = math.hypot(cam_pos[0], math.hypot(cam_pos[1], cam_pos[2]))
        bh_anchor = 1.0 if self.selected_idx is not None else (1.0 - smoothstep(11.0, 14.0, cam_to_bh))
        return float(near_in * far_out * bh_anchor)

    # ==================================================================
    # 3D Scene Rendering (Independent Star & Orbit Passes, Depth Split)
    # ==================================================================

    def render_side(
        self,
        camera,
        aspect_ratio: float,
        window_size: tuple[int, int],
        side_filter: int,
    ) -> None:
        """
        Render S-stars (`show_stars`) and/or orbital paths (`show_orbits`) in one
        depth half-space relative to Sgr A* (`-1` behind BH before raymarching;
        `+1` in front of BH after raymarching).

        Stars and orbits are completely independent: `O` only toggles orbits.
        """
        if not self.show_stars and not self.show_orbits:
            return

        vis = self.get_visibility(camera.absolute_distance)
        if vis <= 0.001:
            return

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect_ratio)
        bh_view_pos = view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
        bh_view_depth = float(-bh_view_pos.z)

        w_px = max(float(window_size[0]), 1.0)
        h_px = max(float(window_size[1]), 1.0)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # 1. Independent Orbital Trajectories (`show_orbits`)
        if self.show_orbits:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.orbit_program["u_view"].write(view_mat)
            self.orbit_program["u_projection"].write(proj_mat)
            self.orbit_program["u_viewport"].value = (w_px, h_px)
            self.orbit_program["u_bhViewDepth"].value = bh_view_depth
            self.orbit_program["u_visibility"].value = vis
            self.orbit_program["u_sideFilter"].value = int(side_filter)
            self.orbit_program["u_camPos"].value = camera.get_position()
            self.orbit_program["u_P"].write(self._u_P.tobytes())
            self.orbit_program["u_Q"].write(self._u_Q.tobytes())
            self.orbit_program["u_orbitColor"].write(self._u_orbit_color.tobytes())
            self.orbit_program["u_orbitParams"].write(self._u_orbit_params.tobytes())
            self.orbit_vao.render(moderngl.TRIANGLES)

        # 2. Independent Moving S-Stars (`show_stars`, always active in range)
        if self.show_stars:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            self.star_program["u_view"].write(view_mat)
            self.star_program["u_projection"].write(proj_mat)
            self.star_program["u_bhViewDepth"].value = bh_view_depth
            self.star_program["u_visibility"].value = vis
            self.star_program["u_fovHeightPx"].value = h_px
            self.star_program["u_sideFilter"].value = int(side_filter)
            self.star_vao.render(moderngl.POINTS)

        # Restore BH premultiplied transmittance blend state
        self.ctx.blend_func = (moderngl.ONE, moderngl.SRC_ALPHA)

    # ==================================================================
    # 2D Overlay Geometry & Zero-Allocation Atlas Text Helpers
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
        if cursor + 6 > len(buf):
            return cursor
        r, g, b, a = rgba
        buf[cursor + 0] = (x0, y0, u0, v0, r, g, b, a)
        buf[cursor + 1] = (x0, y1, u0, v1, r, g, b, a)
        buf[cursor + 2] = (x1, y0, u1, v0, r, g, b, a)
        buf[cursor + 3] = (x1, y0, u1, v0, r, g, b, a)
        buf[cursor + 4] = (x0, y1, u0, v1, r, g, b, a)
        buf[cursor + 5] = (x1, y1, u1, v1, r, g, b, a)
        return cursor + 6

    @staticmethod
    def _append_rotated_sprite(
        buf: np.ndarray,
        cursor: int,
        cx: float,
        cy: float,
        half_w: float,
        half_h: float,
        angle_rad: float,
        uv: tuple[float, float, float, float],
        rgba: tuple[float, float, float, float],
    ) -> int:
        if cursor + 6 > len(buf):
            return cursor
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        u0, v0, u1, v1 = uv
        r, g, b, a = rgba

        def rot(lx: float, ly: float) -> tuple[float, float]:
            return (cx + lx * c - ly * s, cy + lx * s + ly * c)

        p00 = rot(-half_w, -half_h)
        p01 = rot(-half_w,  half_h)
        p10 = rot( half_w, -half_h)
        p11 = rot( half_w,  half_h)

        buf[cursor + 0] = (p00[0], p00[1], u0, v0, r, g, b, a)
        buf[cursor + 1] = (p01[0], p01[1], u0, v1, r, g, b, a)
        buf[cursor + 2] = (p10[0], p10[1], u1, v0, r, g, b, a)
        buf[cursor + 3] = (p10[0], p10[1], u1, v0, r, g, b, a)
        buf[cursor + 4] = (p01[0], p01[1], u0, v1, r, g, b, a)
        buf[cursor + 5] = (p11[0], p11[1], u1, v1, r, g, b, a)
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
        if cursor + 6 > len(buf):
            return cursor
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

    def _measure_text(self, text: str, font_key: str, scale: float = 1.0) -> float:
        glyphs = self._fonts_meta[font_key]["glyphs"]
        space_adv = glyphs.get(" ", {}).get("adv", 4.0)
        return sum(glyphs.get(ch, {}).get("adv", space_adv) for ch in text) * scale

    def _append_text(
        self,
        buf: np.ndarray,
        cursor: int,
        x: float,
        y: float,
        text: str,
        font_key: str,
        rgba: tuple[float, float, float, float],
        scale: float = 1.0,
        align: str = "left",
    ) -> int:
        glyphs = self._fonts_meta[font_key]["glyphs"]
        if align == "right":
            x -= self._measure_text(text, font_key, scale)
        elif align == "center":
            x -= 0.5 * self._measure_text(text, font_key, scale)

        space_adv = glyphs.get(" ", {}).get("adv", 4.0)
        cx = x
        for ch in text:
            g = glyphs.get(ch)
            if g is None:
                cx += space_adv * scale
                continue
            if ch != " ":
                gx0 = cx + g["xoff"] * scale
                gy0 = y
                gx1 = gx0 + g["w"] * scale
                gy1 = gy0 + g["h"] * scale
                u0, v0, u1, v1 = g["uv"]
                cursor = self._append_quad(buf, cursor, gx0, gy0, gx1, gy1, u0, v0, u1, v1, rgba)
            cx += g["adv"] * scale
        return cursor

    @staticmethod
    def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], pad: float) -> bool:
        return not (
            a[2] + pad <= b[0]
            or b[2] + pad <= a[0]
            or a[3] + pad <= b[1]
            or b[3] + pad <= a[1]
        )

    @staticmethod
    def _closest_point_on_box(px: float, py: float, box: tuple[float, float, float, float]) -> tuple[float, float]:
        return min(max(px, box[0]), box[2]), min(max(py, box[1]), box[3])

    # ==================================================================
    # 2D Screen-Space HUD, Badges, Indicators & Telemetry Overlay
    # ==================================================================

    def render_overlay(
        self,
        camera,
        aspect_ratio: float,
        window_size: tuple[int, int],
    ) -> None:
        """
        Render independent S-star HUD components cleanly scaled & anchored:
          1. Bottom-left Cluster Legend & Simulation Clock Card (`show_hud_card`, `H` key)
          2. Bottom-right Compact Selected Star Telemetry Panel (`selected_idx`)
          3. Orbital direction chevrons & periapsis/apoapsis markers (`show_orbits`, `O` key)
          4. Individual S-star name badges & anti-overlap leaders (`show_labels`, `Y` key)
          5. Subtle tracking reticle around selected S-star
        """
        self._last_leg_box = None
        self._last_sel_box = None
        self._last_focus_btn = None
        self._last_center_btn = None
        self._last_close_btn = None

        vis = self.get_visibility(camera.absolute_distance)
        if vis <= 0.02:
            return

        w, h = float(window_size[0]), float(window_size[1])
        if w < 10.0 or h < 10.0:
            return

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect_ratio)
        vp = proj_mat * view_mat

        def project_world(pt3: np.ndarray | tuple[float, float, float]) -> tuple[float, float, float] | None:
            vpos = view_mat * glm.vec4(float(pt3[0]), float(pt3[1]), float(pt3[2]), 1.0)
            if vpos.z >= -0.02:
                return None
            clip = proj_mat * vpos
            if clip.w <= 1e-5:
                return None
            sx = (clip.x / clip.w * 0.5 + 0.5) * w
            sy = (0.5 - clip.y / clip.w * 0.5) * h
            rel_depth = float(-vpos.z) - bh_view_depth
            return sx, sy, rel_depth

        # Apparent Sgr A* screen position and shadow radius
        bh_view = view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
        bh_view_depth = float(-bh_view.z)
        bh_clip = vp * glm.vec4(0.0, 0.0, 0.0, 1.0)
        if bh_clip.w > 1e-5:
            bh_sx = (bh_clip.x / bh_clip.w * 0.5 + 0.5) * w
            bh_sy = (0.5 - bh_clip.y / bh_clip.w * 0.5) * h
        else:
            bh_sx, bh_sy = w * 0.5, h * 0.5

        cam_local_r = max(float(np.linalg.norm(camera.get_position())) * 12.0, 3.0)
        bh_shadow_px = (2.0 * 2.6 / cam_local_r) * 0.5 * h
        bh_shadow_excl = max(bh_shadow_px + 10.0, 24.0)

        # Responsive margins and gentle downscale only if window becomes very narrow
        margin = max(14.0, min(22.0, 0.02 * min(w, h)))
        ui_scale = min(1.0, max(0.68, min((w - 3.0 * margin) / 760.0, (h - 2.0 * margin) / 360.0)))

        buf = self._overlay_vbo_data
        cur = 0
        occupied_boxes: list[tuple[float, float, float, float]] = []

        # ------------------------------------------------------------------
        # 1. Bottom-Left S-Star HUD Legend & Live Clock Card (`show_hud_card`)
        #    Shown automatically only where S-stars are clearly visible near BH
        # ------------------------------------------------------------------
        hud_auto = self.get_hud_visibility(camera)
        ha = float(self._hud_blend) * hud_auto
        if ha > 0.01:
            lw = self._legend_meta["w"] * ui_scale
            lh = self._legend_meta["h"] * ui_scale
            lx0 = margin
            ly0 = h - margin - lh
            lx1 = lx0 + lw
            ly1 = ly0 + lh
            self._last_leg_box = (lx0, ly0, lx1, ly1)
            occupied_boxes.append(self._last_leg_box)

            u0, v0, u1, v1 = self._legend_meta["uv"]
            cur = self._append_quad(buf, cur, lx0, ly0, lx1, ly1, u0, v0, u1, v1, (1.0, 1.0, 1.0, ha))

            # Dynamic Simulation Status row inside Bottom-Left Card (y+54..70)
            row_y = ly0 + 54.0 * ui_scale
            yr_str = f"Year: {self._sim_years:+7.2f} yr    Speed: {self._sim_speed:.2g}x"
            cur = self._append_text(
                buf, cur, lx0 + 13.0 * ui_scale, row_y, yr_str, "val",
                (0.84, 0.92, 1.00, 0.95 * ha), scale=ui_scale, align="left",
            )

            # Live [RUNNING] / [PAUSED] badge right-aligned inside card status row
            pill_txt = "PAUSED" if self._sim_paused else "RUNNING"
            pill_col = (0.36, 0.22, 0.06, 0.88 * ha) if self._sim_paused else (0.08, 0.30, 0.24, 0.88 * ha)
            pill_fg = (1.00, 0.82, 0.38, 0.98 * ha) if self._sim_paused else (0.48, 0.94, 0.74, 0.98 * ha)
            pw = (self._measure_text(pill_txt, "sm", ui_scale) + 14.0 * ui_scale)
            ph = 17.0 * ui_scale
            px1 = lx1 - 12.0 * ui_scale
            px0 = px1 - pw
            py0 = ly0 + 52.5 * ui_scale
            pu0, pv0, pu1, pv1 = self._sprite_meta["pill"]["uv"]
            cur = self._append_quad(buf, cur, px0, py0, px1, py0 + ph, pu0, pv0, pu1, pv1, pill_col)
            cur = self._append_text(
                buf, cur, 0.5 * (px0 + px1), py0 + 1.5 * ui_scale, pill_txt, "sm",
                pill_fg, scale=ui_scale, align="center",
            )

        # ------------------------------------------------------------------
        # 2. Compact Bottom-Right Selected Star Telemetry Panel
        # ------------------------------------------------------------------
        disp_idx = self._display_selected_idx
        if disp_idx is not None and self._selection_blend > 0.01:
            sa = float(self._selection_blend)
            sw = self._sel_panel_meta["w"] * ui_scale
            sh = self._sel_panel_meta["h"] * ui_scale
            sx0 = w - margin - sw
            sy0 = h - margin - sh
            sx1 = sx0 + sw
            sy1 = sy0 + sh
            self._last_sel_box = (sx0, sy0, sx1, sy1)
            occupied_boxes.append(self._last_sel_box)

            u0, v0, u1, v1 = self._sel_panel_meta["uv"]
            cur = self._append_quad(buf, cur, sx0, sy0, sx1, sy1, u0, v0, u1, v1, (1.0, 1.0, 1.0, sa))

            # Store exact interactive button hit-rects only while selection is active
            if self.selected_idx is not None:
                self._last_close_btn = (
                    sx1 - 32.0 * ui_scale, sy0 + 6.0 * ui_scale,
                    sx1 - 12.0 * ui_scale, sy0 + 26.0 * ui_scale,
                )
                self._last_focus_btn = (
                    sx0 + 12.0 * ui_scale, sy0 + 222.0 * ui_scale,
                    sx0 + 166.0 * ui_scale, sy0 + 246.0 * ui_scale,
                )
                self._last_center_btn = (
                    sx0 + 174.0 * ui_scale, sy0 + 222.0 * ui_scale,
                    sx1 - 12.0 * ui_scale, sy0 + 246.0 * ui_scale,
                )

            spec = self.specs[disp_idx]
            title_col = (
                (1.00, 0.86, 0.50, sa) if spec.is_reference
                else ((0.54, 0.86, 1.00, sa) if spec.is_measured else (0.78, 0.86, 0.96, sa))
            )
            sub_txt = (
                "S2 Reference Orbit (GRAVITY/Keck)" if spec.is_reference
                else ("Measured S-Star Orbit (GRAVITY/Keck)" if spec.is_measured else "Representative Synthetic Tracer Model")
            )
            cur = self._append_text(
                buf, cur, sx0 + 14.0 * ui_scale, sy0 + 8.0 * ui_scale,
                spec.catalog_label.split("  |  ")[0], "lg", title_col, scale=ui_scale,
            )
            cur = self._append_text(
                buf, cur, sx0 + 14.0 * ui_scale, sy0 + 29.0 * ui_scale,
                sub_txt, "sm", (0.72, 0.84, 0.96, 0.92 * sa), scale=ui_scale,
            )

            # Dynamic right-aligned metrics from live catalog & Kepler state
            a_au = float(self.a_au[disp_idx])
            ecc = float(self.e[disp_idx])
            per = float(self.period_yr[disp_idx])
            inc_d = float(spec.inc_deg)
            r_cur = float(self.r_au[disp_idx])
            v_cur = float(self.speed_kms[disp_idx])
            rp_au = float(self.rp_au[disp_idx])
            ra_au = float(self.ra_au[disp_idx])
            m_deg = float(math.degrees(self.mean_anomaly[disp_idx]) % 360.0)
            pct = int(round((m_deg / 360.0) * 100.0)) % 100

            rows_val = [
                f"{a_au:,.0f} AU ({spec.a_arcsec:.3f}\")",
                f"{ecc:.4f}",
                f"{per:.2f} yr",
                f"{inc_d:.1f}\u00b0",
                f"{r_cur:,.1f} AU",
                f"{v_cur:,.0f} km/s",
                f"{rp_au:,.0f} / {ra_au:,.0f} AU",
                f"{m_deg:5.1f}\u00b0 ({pct}%)",
            ]
            val_col = (0.94, 0.97, 1.00, 0.98 * sa)
            for row_i, val_txt in enumerate(rows_val):
                ry = sy0 + (58.0 + row_i * 19.0) * ui_scale
                cur = self._append_text(
                    buf, cur, sx1 - 14.0 * ui_scale, ry, val_txt, "val",
                    val_col, scale=ui_scale, align="right",
                )

            # Action bar labels
            focus_lbl = "[F] Tracking Star" if self._focus_active else "[F] Focus Star"
            focus_col = (1.00, 0.88, 0.52, sa) if self._focus_active else (0.78, 0.92, 1.00, 0.96 * sa)
            cur = self._append_text(
                buf, cur, sx0 + 89.0 * ui_scale, sy0 + 226.0 * ui_scale,
                focus_lbl, "sm", focus_col, scale=ui_scale, align="center",
            )
            cur = self._append_text(
                buf, cur, sx0 + (174.0 + (self._sel_panel_meta["w"] - 186.0) * 0.5) * ui_scale, sy0 + 226.0 * ui_scale,
                "[C] Center Sgr A*", "sm", (0.76, 0.87, 0.98, 0.92 * sa), scale=ui_scale, align="center",
            )

        # ------------------------------------------------------------------
        # 3. Orbital Direction Chevrons (`show_orbits`: NEVER hidden by Y)
        # ------------------------------------------------------------------
        chev_uv = self._sprite_meta["chevron"]["uv"]
        diam_uv = self._sprite_meta["diamond"]["uv"]
        sq_uv = self._sprite_meta["square"]["uv"]

        if self.show_orbits:
            dir_indices = list(self.identified_indices)
            if disp_idx is not None and disp_idx not in dir_indices and self._selection_blend > 0.05:
                dir_indices.append(disp_idx)

            for idx in dir_indices:
                proj = project_world(self.pos_world[idx])
                if proj is None:
                    continue
                sx, sy, rel_depth = proj
                if sx < 6.0 or sx > w - 6.0 or sy < 6.0 or sy > h - 6.0:
                    continue
                d_bh = math.hypot(sx - bh_sx, sy - bh_sy)
                if rel_depth > 0.0 and d_bh < bh_shadow_px * 0.90:
                    continue

                # Compute true 3D projected velocity tangent direction
                vnorm = max(float(self.speed_kms[idx]), 1e-6)
                vhat = self.vel_kms[idx] / vnorm
                step_w = max(0.06, float(self.r_au[idx] * WORLD_PER_AU) * 0.14)
                ahead_proj = project_world(self.pos_world[idx] + vhat.astype(np.float32) * step_w)
                if ahead_proj is None:
                    continue
                ax, ay, _ = ahead_proj
                tdx, tdy = ax - sx, ay - sy
                tlen = math.hypot(tdx, tdy)
                if tlen < 2.0:
                    continue
                ux, uy = tdx / tlen, tdy / tlen
                ang = math.atan2(uy, ux)

                # Place small delicate chevron slightly ahead of star center
                cx = sx + ux * 11.5
                cy = sy + uy * 11.5
                is_sel = (idx == disp_idx)
                col_c = (
                    (1.00, 0.84, 0.42, 0.86) if self.specs[idx].is_reference
                    else ((0.72, 0.92, 1.00, 0.90) if is_sel else (0.50, 0.80, 0.98, 0.62))
                )
                if rel_depth > 0.0:
                    col_c = (col_c[0], col_c[1], col_c[2], col_c[3] * 0.72)
                if not is_sel and disp_idx is not None:
                    col_c = (col_c[0], col_c[1], col_c[2], col_c[3] * (1.0 - 0.50 * self._selection_blend))
                sz = 7.2 if (self.specs[idx].is_reference or is_sel) else 6.0
                cur = self._append_rotated_sprite(buf, cur, cx, cy, sz, sz, ang, chev_uv, col_c)

        # ------------------------------------------------------------------
        # 4. Subtle Selected Star Tracking Ring (independent of O & Y!)
        # ------------------------------------------------------------------
        if disp_idx is not None and self._selection_blend > 0.02:
            p_sel = project_world(self.pos_world[disp_idx])
            if p_sel is not None:
                sx, sy, pd = p_sel
                if -20.0 <= sx <= w + 20.0 and -20.0 <= sy <= h + 20.0:
                    if not (pd > 0.0 and math.hypot(sx - bh_sx, sy - bh_sy) < bh_shadow_px * 0.90):
                        ring_uv = self._sprite_meta["ring"]["uv"]
                        ring_col = (1.00, 0.86, 0.46, 0.92 * self._selection_blend) if self.specs[disp_idx].is_reference else (0.58, 0.88, 1.00, 0.92 * self._selection_blend)
                        cur = self._append_quad(buf, cur, sx - 13.0, sy - 13.0, sx + 13.0, sy + 13.0, *ring_uv, ring_col)

        # ------------------------------------------------------------------
        # 5. Individual S-Star Name Badges & Reticles (`show_labels`, `Y` key ONLY)
        #    Placed BEFORE apse tags so apse tags cleanly yield to name pills.
        # ------------------------------------------------------------------
        if self.show_labels:
            label_indices = list(self.identified_indices)
            projected: list[tuple[int, float, float, float]] = []
            star_points_px: list[tuple[float, float]] = []

            for idx in label_indices:
                p_pr = project_world(self.pos_world[idx])
                if p_pr is None:
                    continue
                sx, sy, rel_depth = p_pr
                if sx < -40.0 or sx > w + 40.0 or sy < -40.0 or sy > h + 40.0:
                    continue
                if rel_depth > 0.0 and math.hypot(sx - bh_sx, sy - bh_sy) < bh_shadow_px * 0.90:
                    continue
                projected.append((idx, sx, sy, rel_depth))
                if 0.0 <= sx <= w and 0.0 <= sy <= h:
                    star_points_px.append((sx, sy))

            base_dirs = [
                (math.cos(math.radians(a)), math.sin(math.radians(a)))
                for a in (0, 45, -45, 90, -90, 135, -135, 180)
            ]
            placed_labels = []

            for idx, sx, sy, rel_depth in projected:
                spec = self.specs[idx]
                meta = self._badge_meta[idx]
                bw, bh_px = meta["w"], meta["h"]

                col_line = (1.00, 0.82, 0.36, 0.92) if spec.is_reference else (0.50, 0.82, 1.00, 0.86)
                if rel_depth > 0.0:
                    col_line = (col_line[0], col_line[1], col_line[2], col_line[3] * 0.76)

                rx, ry = sx - bh_sx, sy - bh_sy
                rnorm = math.hypot(rx, ry)
                ux, uy = (rx / rnorm, ry / rnorm) if rnorm > 1.0 else (0.707, -0.707)

                dirs_sorted = sorted(base_dirs, key=lambda d: -(d[0] * ux + d[1] * uy))
                radii = (22.0, 38.0, 54.0) if spec.is_reference else (20.0, 34.0)

                chosen_box = None
                for r_off in radii:
                    for dx, dy in dirs_sorted:
                        cx = sx + dx * r_off
                        cy = sy + dy * r_off
                        bx0, by0 = cx - 0.5 * bw, cy - 0.5 * bh_px
                        bx1, by1 = bx0 + bw, by0 + bh_px
                        if bx0 < 8.0 or by0 < 8.0 or bx1 > w - 8.0 or by1 > h - 8.0:
                            continue
                        cand = (bx0, by0, bx1, by1)
                        cbx, cby = self._closest_point_on_box(bh_sx, bh_sy, cand)
                        if math.hypot(cbx - bh_sx, cby - bh_sy) < bh_shadow_excl:
                            continue
                        if any(self._boxes_overlap(cand, ob, pad=6.0) for ob in occupied_boxes):
                            continue
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

                if 4.0 <= sx <= w - 4.0 and 4.0 <= sy <= h - 4.0:
                    r_tick = 7.0 if spec.is_reference else 5.8
                    cur = self._append_segment_quad(buf, cur, sx - r_tick, sy, sx - 3.2, sy, 1.2, col_line)
                    cur = self._append_segment_quad(buf, cur, sx + 3.2, sy, sx + r_tick, sy, 1.2, col_line)
                    cur = self._append_segment_quad(buf, cur, sx, sy - r_tick, sx, sy - 3.2, 1.2, col_line)
                    cur = self._append_segment_quad(buf, cur, sx, sy + 3.2, sx, sy + r_tick, 1.2, col_line)

                if chosen_box is not None:
                    occupied_boxes.append(chosen_box)
                    placed_labels.append((idx, sx, sy, chosen_box, col_line))

            for idx, sx, sy, box, col_line in placed_labels:
                lx, ly = self._closest_point_on_box(sx, sy, box)
                dist_l = math.hypot(lx - sx, ly - sy)
                if dist_l > 8.5:
                    gap = 5.5
                    ax = sx + ((lx - sx) / dist_l) * gap
                    ay = sy + ((ly - sy) / dist_l) * gap
                    cur = self._append_segment_quad(
                        buf, cur, ax, ay, lx, ly, 1.05,
                        (col_line[0], col_line[1], col_line[2], col_line[3] * 0.72),
                    )
                meta = self._badge_meta[idx]
                cur = self._append_quad(
                    buf, cur, box[0], box[1], box[2], box[3], *meta["uv"],
                    (1.0, 1.0, 1.0, 1.0),
                )

        # ------------------------------------------------------------------
        # 6. Subtle Periapsis & Apoapsis Indicators (`show_orbits`)
        # ------------------------------------------------------------------
        if self.show_orbits:
            apse_targets: list[tuple[int, float]] = []
            if disp_idx is not None and self._selection_blend > 0.05:
                apse_targets.append((disp_idx, float(self._selection_blend)))
            if disp_idx != 0:
                ref_a = 0.75 * (1.0 - 0.45 * self._selection_blend)
                apse_targets.append((0, ref_a))

            for idx, alpha_w in apse_targets:
                p_vec = self.P_curr[idx]
                peri_world = self.rp_world[idx] * p_vec
                apo_world = -self.ra_world[idx] * p_vec
                is_sel = (idx == disp_idx)
                base_tint = (1.00, 0.84, 0.42) if self.specs[idx].is_reference else (0.56, 0.86, 1.00)

                for pt_w, kind_uv, tag_key in [
                    (peri_world, diam_uv, "peri"),
                    (apo_world,  sq_uv,   "apo"),
                ]:
                    p_pr = project_world(pt_w)
                    if p_pr is None:
                        continue
                    px, py, pd = p_pr
                    if px < 10.0 or px > w - 10.0 or py < 10.0 or py > h - 10.0:
                        continue
                    d_bh = math.hypot(px - bh_sx, py - bh_sy)
                    if pd > 0.0 and d_bh < bh_shadow_px * 0.90:
                        continue

                    m_alpha = alpha_w * (0.72 if pd <= 0.0 else 0.50) * (0.95 if tag_key == "peri" else 0.75)
                    col_m = (base_tint[0], base_tint[1], base_tint[2], m_alpha)
                    cur = self._append_quad(buf, cur, px - 6.5, py - 6.5, px + 6.5, py + 6.5, *kind_uv, col_m)

                    # Show concise "rp" / "ra" tag only on selected orbit (or when Y labels on S2)
                    # and keep the black hole shadow interior clean
                    if ((is_sel and self._selection_blend > 0.25) or (idx == 0 and self.show_labels)) and d_bh >= bh_shadow_px * 0.95:
                        tm = self._tag_meta[tag_key]
                        # Try right, left, top, bottom offsets so tag avoids star badge/shadow
                        placed_tag = None
                        for ox, oy in [
                            (9.0, -tm["h"] * 0.5),
                            (-tm["w"] - 9.0, -tm["h"] * 0.5),
                            (-tm["w"] * 0.5, -tm["h"] - 9.0),
                            (-tm["w"] * 0.5, 9.0),
                        ]:
                            tx0 = px + ox
                            ty0 = py + oy
                            tbox = (tx0, ty0, tx0 + tm["w"], ty0 + tm["h"])
                            if tx0 < 8.0 or ty0 < 8.0 or tbox[2] > w - 8.0 or tbox[3] > h - 8.0:
                                continue
                            cbx, cby = self._closest_point_on_box(bh_sx, bh_sy, tbox)
                            if math.hypot(cbx - bh_sx, cby - bh_sy) < bh_shadow_px * 0.92:
                                continue
                            if any(self._boxes_overlap(tbox, ob, pad=4.0) for ob in occupied_boxes):
                                continue
                            placed_tag = tbox
                            break
                        if placed_tag is not None:
                            occupied_boxes.append(placed_tag)
                            cur = self._append_quad(buf, cur, *placed_tag, *tm["uv"], (1.0, 1.0, 1.0, alpha_w * 0.88))

        self._overlay_vert_count = cur
        if cur == 0:
            return

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
