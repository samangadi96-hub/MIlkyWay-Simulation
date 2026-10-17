import numpy as np


# Spectral class indices: 0: O, 1: B, 2: A, 3: F, 4: G, 5: K, 6: M
CLASS_NAMES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

# Diagnostic colors for population verification
DIAGNOSTIC_COLORS = np.array([
    [1.00, 0.20, 0.80],  # O: Magenta
    [0.20, 0.70, 1.00],  # B: Cyan
    [0.95, 0.95, 1.00],  # A: White
    [0.85, 1.00, 0.50],  # F: Light yellow-green
    [1.00, 0.90, 0.20],  # G: Yellow
    [1.00, 0.55, 0.10],  # K: Orange
    [1.00, 0.20, 0.10],  # M: Red
], dtype=np.float32)


def temperature_to_rgb(temps, desat=0.04):
    """
    Vectorized continuous blackbody color approximation.
    Maps effective temperature (Kelvin) to normalized linear RGB.
    Calibrated to Planckian locus and Mitchell Charity stellar color calculations.
    
    A subtle desaturation factor ensures colors remain understated,
    avoiding artificial or saturated hues on additive GPU point rendering.
    """
    t = np.clip(temps, 2200.0, 45000.0) / 100.0
    r = np.zeros_like(t, dtype=np.float32)
    g = np.zeros_like(t, dtype=np.float32)
    b = np.zeros_like(t, dtype=np.float32)

    # Red component
    m1 = t <= 66.0
    r[m1] = 1.0
    r[~m1] = np.clip(1.292936 * ((t[~m1] - 60.0) ** -0.133205), 0.0, 1.0)

    # Green component
    m2 = t <= 66.0
    g[m2] = np.clip(0.390082 * np.log(t[m2]) - 0.631841, 0.0, 1.0)
    g[~m2] = np.clip(1.129891 * ((t[~m2] - 60.0) ** -0.075515), 0.0, 1.0)

    # Blue component
    m3 = t <= 19.0
    m4 = (t > 19.0) & (t < 66.0)
    m5 = t >= 66.0
    b[m3] = 0.0
    b[m4] = np.clip(0.543207 * np.log(t[m4] - 10.0) - 1.196254, 0.0, 1.0)
    b[m5] = 1.0

    rgb = np.column_stack((r, g, b))
    # Soften towards white to maintain restrained, realistic stellar appearance
    rgb = rgb * (1.0 - desat) + desat * 1.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def generate_stellar_population(
    num_stars,
    population_type='disk',
    rng=None,
    arm_factor=None,
    radius=None,
    diagnostic_mode=False
):
    """
    Generate realistic astrophysical stellar populations:
    - Spectral classes (O, B, A, F, G, K, M) based on galactic component and environment
    - Effective surface temperatures
    - Temperature-derived RGB colors
    - Luminosity / brightness following IMF (dwarf vs giant/supergiant)
    - Corresponding rendered point sizes
    
    Parameters:
        num_stars: int, number of stars to generate
        population_type: str, 'bulge', 'bar', 'disk', or 'nebula_cluster'
        rng: np.random.Generator (optional, deterministic if provided)
        arm_factor: np.ndarray, float values in [0, 1] for disk stars (0 = inter-arm, 1 = arm core)
        radius: np.ndarray, float values representing galactic radius
        diagnostic_mode: bool, if True, assigns distinct debug colors per spectral class
        
    Returns:
        colors: (num_stars, 3) float32 array of RGB colors
        sizes: (num_stars,) float32 array of rendered point sizes
        brightness: (num_stars,) float32 array of star brightness values
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if num_stars == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32)
        )

    # ==============================================================
    # 1. Determine Spectral Class Probabilities
    # ==============================================================
    # Probabilities in order [O, B, A, F, G, K, M]
    if population_type == 'bulge':
        # Ancient Population II & dense nuclear core:
        # Dominated by G solar-type stars, K red clump giants, and warm F/G stars
        # creating a luminous warm-white and pale-gold nuclear core around Sgr A*
        probs = np.array([0.000, 0.000, 0.005, 0.095, 0.380, 0.380, 0.140], dtype=np.float64)
        classes = rng.choice(7, size=num_stars, p=probs)

    elif population_type == 'bar':
        # Primarily older & intermediate-age stars (age ~ 4-8 Gyr)
        # Warm golden population: rich in G dwarfs/subgiants and K golden giants,
        # with subtle warm-white F stars and amber M dwarfs.
        probs = np.array([0.000, 0.000, 0.005, 0.075, 0.340, 0.430, 0.150], dtype=np.float64)
        classes = rng.choice(7, size=num_stars, p=probs)

    elif population_type == 'nebula_cluster':
        # Embedded young OB association inside an active star-forming H-II complex
        # Rich in ionizing hot O and B stars, A-type giants, and young stellar objects.
        probs = np.array([0.050, 0.220, 0.220, 0.160, 0.140, 0.120, 0.090], dtype=np.float64)
        classes = rng.choice(7, size=num_stars, p=probs)

    else:
        # General Galactic Disk (continuous arm vs inter-arm & radial variation)
        # Inter-arm base: mixed population with low young star fraction
        p_inter = np.array([0.000, 0.006, 0.040, 0.100, 0.260, 0.350, 0.244], dtype=np.float64)
        # Spiral arm core: enhanced young, hot, luminous star fraction (OB associations)
        p_arm   = np.array([0.014, 0.086, 0.110, 0.140, 0.230, 0.240, 0.180], dtype=np.float64)

        if arm_factor is None:
            arm_factor = np.zeros(num_stars, dtype=np.float64)
        else:
            arm_factor = np.clip(arm_factor, 0.0, 1.0)

        # Smooth interpolation of probability vector per star
        p_matrix = (1.0 - arm_factor[:, None]) * p_inter + arm_factor[:, None] * p_arm

        # Apply slight radial modulation: inner disk is older, outer disk has slightly fewer metals
        if radius is not None:
            r_norm = np.clip((radius - 4.0) / 25.0, 0.0, 1.0)
            # Subtle boost to young stars in star-forming ring (r ~ 8 - 25)
            ring_factor = np.exp(-0.5 * ((radius - 16.0) / 9.0) ** 2) * 0.25
            p_matrix[:, 0] += ring_factor * 0.005
            p_matrix[:, 1] += ring_factor * 0.025

        # Normalize cumulative probabilities
        cum_probs = np.cumsum(p_matrix, axis=1)
        cum_probs /= cum_probs[:, -1:]

        rand_vals = rng.random((num_stars, 1))
        classes = np.argmax(rand_vals < cum_probs, axis=1)

    # ==============================================================
    # 2. Sample Temperatures (Kelvin) within Spectral Class
    # ==============================================================
    temps = np.zeros(num_stars, dtype=np.float32)
    sizes = np.zeros(num_stars, dtype=np.float32)
    brightness = np.zeros(num_stars, dtype=np.float32)

    # Giant / evolutionary status roll (IMF & HR diagram proxy)
    evo_roll = rng.random(num_stars)

    # Class 0: O-type (T = 30,000 - 42,000 K) - Brilliant celestial blue-white
    mask_o = classes == 0
    if np.any(mask_o):
        n_o = np.sum(mask_o)
        temps[mask_o] = rng.uniform(30000.0, 42000.0, n_o)
        is_super = evo_roll[mask_o] < 0.22
        # Main sequence O
        sizes[mask_o] = np.where(is_super, rng.uniform(0.12, 0.18, n_o), rng.uniform(0.075, 0.12, n_o))
        brightness[mask_o] = np.where(is_super, rng.uniform(2.4, 3.4, n_o), rng.uniform(1.6, 2.4, n_o))

    # Class 1: B-type (T = 11,000 - 29,000 K) - Ice blue-white
    mask_b = classes == 1
    if np.any(mask_b):
        n_b = np.sum(mask_b)
        temps[mask_b] = rng.uniform(11000.0, 29000.0, n_b)
        is_giant_b = evo_roll[mask_b] < 0.15
        sizes[mask_b] = np.where(is_giant_b, rng.uniform(0.08, 0.13, n_b), rng.uniform(0.040, 0.075, n_b))
        brightness[mask_b] = np.where(is_giant_b, rng.uniform(1.6, 2.3, n_b), rng.uniform(1.0, 1.6, n_b))

    # Class 2: A-type (T = 7,500 - 10,500 K) - Crisp pure white
    mask_a = classes == 2
    if np.any(mask_a):
        n_a = np.sum(mask_a)
        temps[mask_a] = rng.uniform(7500.0, 10500.0, n_a)
        is_giant_a = evo_roll[mask_a] < 0.12
        sizes[mask_a] = np.where(is_giant_a, rng.uniform(0.055, 0.090, n_a), rng.uniform(0.026, 0.048, n_a))
        brightness[mask_a] = np.where(is_giant_a, rng.uniform(1.2, 1.7, n_a), rng.uniform(0.70, 1.10, n_a))

    # Class 3: F-type (T = 6,000 - 7,400 K) - Warm neutral white
    mask_f = classes == 3
    if np.any(mask_f):
        n_f = np.sum(mask_f)
        temps[mask_f] = rng.uniform(6000.0, 7400.0, n_f)
        is_giant_f = evo_roll[mask_f] < 0.10
        sizes[mask_f] = np.where(is_giant_f, rng.uniform(0.045, 0.075, n_f), rng.uniform(0.020, 0.038, n_f))
        brightness[mask_f] = np.where(is_giant_f, rng.uniform(1.0, 1.45, n_f), rng.uniform(0.60, 0.90, n_f))

    # Class 4: G-type (T = 5,200 - 5,950 K) - Pale solar yellow-white
    mask_g = classes == 4
    if np.any(mask_g):
        n_g = np.sum(mask_g)
        temps[mask_g] = rng.uniform(5200.0, 5950.0, n_g)
        is_giant_g = evo_roll[mask_g] < 0.10
        sizes[mask_g] = np.where(is_giant_g, rng.uniform(0.045, 0.075, n_g), rng.uniform(0.020, 0.036, n_g))
        brightness[mask_g] = np.where(is_giant_g, rng.uniform(1.0, 1.45, n_g), rng.uniform(0.55, 0.85, n_g))

    # Class 5: K-type (T = 3,800 - 5,100 K) - Warm golden yellow to orange
    mask_k = classes == 5
    if np.any(mask_k):
        n_k = np.sum(mask_k)
        temps[mask_k] = rng.uniform(3800.0, 5100.0, n_k)
        is_giant_k = evo_roll[mask_k] < 0.14  # Red clump giants
        sizes[mask_k] = np.where(is_giant_k, rng.uniform(0.052, 0.090, n_k), rng.uniform(0.018, 0.032, n_k))
        brightness[mask_k] = np.where(is_giant_k, rng.uniform(1.05, 1.60, n_k), rng.uniform(0.50, 0.78, n_k))

    # Class 6: M-type (T = 2,600 - 3,750 K) - Warm amber to orange-red (understated)
    mask_m = classes == 6
    if np.any(mask_m):
        n_m = np.sum(mask_m)
        temps[mask_m] = rng.uniform(2600.0, 3750.0, n_m)
        is_giant_m = evo_roll[mask_m] < 0.08  # Asymptotic giant branch / red giants
        sizes[mask_m] = np.where(is_giant_m, rng.uniform(0.048, 0.085, n_m), rng.uniform(0.014, 0.026, n_m))
        brightness[mask_m] = np.where(is_giant_m, rng.uniform(0.95, 1.45, n_m), rng.uniform(0.35, 0.55, n_m))

    # Core luminosity boost for bulge and central bar
    if population_type in ('bulge', 'bar'):
        brightness *= 1.25

    # ==============================================================
    # 3. Derive Linear RGB Colors
    # ==============================================================
    if diagnostic_mode:
        colors = DIAGNOSTIC_COLORS[classes]
    else:
        colors = temperature_to_rgb(temps, desat=0.04)
        # Subtle per-star chromatic variance prevents sterile uniformity
        color_noise = rng.normal(0.0, 0.015, (num_stars, 3)).astype(np.float32)
        colors = np.clip(colors + color_noise, 0.0, 1.0)

    return (
        colors.astype(np.float32),
        sizes.astype(np.float32),
        brightness.astype(np.float32)
    )
