import numpy as np

class NebulaManager:
    def __init__(self, num_nebulae=60, seed=1337, diagnostic_mode=False):
        self.num_nebulae = num_nebulae
        self.seed = seed
        self.diagnostic_mode = diagnostic_mode
        rng = np.random.default_rng(self.seed)
        
        self.nebulae_data = [] # List of dicts for shader uniforms
        self.nebulae_meta = [] # Physical spiral-arm metadata per nebula (for ISM parent-child coupling)
        self.stars_data = []   # List of stars to add to galactic disk
        
        # Spiral Parameters (must match galactic_disk.py exactly)
        ARM_COUNT = 4
        SPIRAL_TIGHTNESS = 2.5
        INNER_RADIUS = 3.5
        ARM_BASE_OFFSET = (np.pi / 2.0 - 0.488) - 2.5 * np.log(7.0 / 3.5)
        
        # We want nebulae mostly in the active star-forming ring
        MIN_R = 6.0
        MAX_R = 40.0
        
        # Emission colors (Realistic Astrophotography H-II palette)
        # H-alpha (656 nm, deep red) + H-beta (486 nm, blue-cyan) + O-III (500 nm, teal)
        color_hii_crimson = np.array([0.92, 0.06, 0.16]) # Pure deep H-alpha red
        color_hii_magenta = np.array([0.96, 0.16, 0.36]) # Standard H-II magenta/pink
        color_hii_salmon  = np.array([0.98, 0.28, 0.38]) # Warmer pink
        color_hii_pale    = np.array([1.00, 0.42, 0.52]) # Pale pink (mixed with O-III)
        
        # Blue/white for reflection nebulae (dust scattering hot young star light)
        color_ref_bright = np.array([0.35, 0.68, 1.0])
        color_ref_dusty  = np.array([0.22, 0.42, 0.82])
        
        for i in range(num_nebulae):
            # 1. Position along spiral arms
            # Preferentially place in the denser parts (r ~ 10-35)
            r = rng.gamma(shape=4.0, scale=4.0) + MIN_R
            r = np.clip(r, MIN_R, MAX_R)
            
            # Select arm with authentic Milky Way 4-arm asymmetry
            # (Perseus & Scutum-Centaurus major; Sagittarius & Norma minor)
            arm_index = rng.choice([0, 1, 2, 3], p=[0.31, 0.20, 0.30, 0.19])
            arm_base_angle = arm_index * (2.0 * np.pi / ARM_COUNT)
            
            # Ideal log spiral centerline angle (matching galactic_disk.py exactly)
            ideal_theta = arm_base_angle + ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * np.log(r / INNER_RADIUS)
            
            # Physical perpendicular distance offset (1 unit ~ 700 pc):
            # H-II star-forming complexes lie within the active star-forming envelope
            # (100 - 280 pc, ~0.14 - 0.40 units), positioned slightly downstream of the
            # inner-edge dust shock front along the gas compression ridge.
            # Convert physical offset to angular offset via: delta_theta = perp_offset / r
            for _ in range(10):
                perp_offset = rng.normal(0.12, 0.12) # ~84 pc mean downstream offset, ~84 pc std
                if abs(perp_offset) <= 0.38: # Enforce strict star-forming arm envelope (~266 pc)
                    break
            else:
                perp_offset = np.clip(perp_offset, -0.38, 0.38)
                
            delta_theta = perp_offset / r
            theta = ideal_theta + delta_theta
            
            # 3D Position
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            
            # Nebulae are born in cold molecular gas in the ultra-thin disk (scale height ~ 50 pc ~ 0.07 units)
            y = rng.normal(0.0, 0.07)
            
            # 2. Size and Type (Astrophysical H-II complexes: 100 - 350 pc)
            size_roll = rng.random()
            if size_roll < 0.10: # ~6 hero complexes (like 30 Doradus, NGC 604)
                radius = rng.uniform(0.65, 0.95)
                density = rng.uniform(1.0, 1.4)
            elif size_roll < 0.35: # ~15 medium star-forming complexes
                radius = rng.uniform(0.38, 0.65)
                density = rng.uniform(0.8, 1.1)
            else: # ~39 small localized H-II knots
                radius = rng.uniform(0.20, 0.38)
                density = rng.uniform(0.6, 0.9)
                
            # Reflection fraction (minority are blue)
            ref_roll = rng.random()
            if ref_roll < 0.05: # 5% strong reflection
                c = color_ref_bright
                emission_strength = rng.uniform(1.2, 2.0)
            elif ref_roll < 0.2: # 15% weak reflection / mixed
                c = color_ref_dusty * 0.4 + color_hii_magenta * 0.6
                emission_strength = rng.uniform(1.0, 1.5)
            else: # 80% H-II
                c_roll = rng.random()
                if c_roll < 0.30: c = color_hii_crimson
                elif c_roll < 0.70: c = color_hii_magenta
                elif c_roll < 0.90: c = color_hii_salmon
                else: c = color_hii_pale
                emission_strength = rng.uniform(1.5, 2.5)
                
            # Add some variance to color
            c = c + rng.normal(0.0, 0.05, 3)
            c = np.clip(c, 0.0, 1.0)
            
            # Pack metadata into the shader
            # w of position = radius
            # w of color = combined density/emission parameter
            self.nebulae_data.append({
                'position': [x, y, z, radius],
                'color': [c[0], c[1], c[2], density * emission_strength]
            })
            self.nebulae_meta.append({
                'index': i,
                'arm_index': int(arm_index),
                'r': float(r),
                'theta': float(theta),
                'ideal_theta': float(ideal_theta),
                'perp_offset': float(perp_offset),
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'radius': float(radius),
                'color': [float(c[0]), float(c[1]), float(c[2])],
                'strength': float(density * emission_strength),
                'is_reflection': bool(ref_roll < 0.20),
                'size_tier': 'hero' if size_roll < 0.10 else ('medium' if size_roll < 0.35 else 'knot'),
            })
            
            # 3. Generate Young Bright Stars associated with this nebula
            # Create dense star clusters (aggregates) that stand out from the surrounding disk
            if size_roll < 0.4: # Top 40% of nebulae get star clusters
                if size_roll < 0.1: # Hero regions get massive clusters
                    cluster_count = rng.integers(150, 320)
                elif size_roll < 0.25: # Medium regions get dense clusters
                    cluster_count = rng.integers(50, 110)
                else: # Smaller regions get sparse clusters
                    cluster_count = rng.integers(15, 35)
                    
                dist_scale = rng.gamma(shape=1.5, scale=0.3, size=cluster_count)
                theta_star = rng.uniform(0.0, 2.0 * np.pi, size=cluster_count)
                phi_star = rng.uniform(0.0, np.pi, size=cluster_count)
                
                # 3D offset (flattened slightly in the Y axis)
                dx = radius * dist_scale * np.sin(phi_star) * np.cos(theta_star)
                dy = (radius * dist_scale * np.cos(phi_star)) * 0.4
                dz = radius * dist_scale * np.sin(phi_star) * np.sin(theta_star)
                
                sx = x + dx
                sy = y + dy
                sz = z + dz
                
                from stellar_population import generate_stellar_population
                c_stars, s_stars, b_stars = generate_stellar_population(
                    cluster_count,
                    population_type='nebula_cluster',
                    rng=rng,
                    diagnostic_mode=self.diagnostic_mode
                )
                
                cluster_data = np.column_stack((sx, sy, sz, c_stars, s_stars, b_stars)).astype('f4')
                self.stars_data.extend(cluster_data)

    def get_stars_array(self):
        if not self.stars_data:
            return np.zeros((0, 8), dtype='f4')
        return np.array(self.stars_data, dtype='f4')

    def get_nebulae_regions(self):
        """Reusable interface exposing positions and radii of all star-forming regions."""
        return [{'center': n['position'][:3], 'radius': n['position'][3]} for n in self.nebulae_data]

    def get_nebulae_metadata(self):
        """Returns physical arm and morphology metadata for all catalogued nebulae."""
        return self.nebulae_meta

    def get_shader_uniforms(self):
        """Returns two flat lists of floats for the shader arrays."""
        positions = []
        colors = []
        for n in self.nebulae_data:
            positions.extend(n['position'])
            colors.extend(n['color'])
            
        # Pad up to a maximum number (e.g. 150) if needed
        MAX_NEBULAE = 150
        while len(positions) // 4 < MAX_NEBULAE:
            positions.extend([0.0, 0.0, 0.0, 0.0])
            colors.extend([0.0, 0.0, 0.0, 0.0])
            
        return positions, colors
