import numpy as np

class NebulaManager:
    def __init__(self, num_nebulae=100, seed=1337, diagnostic_mode=False):
        self.num_nebulae = num_nebulae
        self.seed = seed
        self.diagnostic_mode = diagnostic_mode
        rng = np.random.default_rng(self.seed)
        
        self.nebulae_data = [] # List of dicts for shader uniforms
        self.stars_data = []   # List of stars to add to galactic disk
        
        # Spiral Parameters (must match galactic_disk.py)
        ARM_COUNT = 4
        SPIRAL_TIGHTNESS = 2.5
        INNER_RADIUS = 2.0
        
        # We want nebulae mostly in the active star-forming ring
        MIN_R = 6.0
        MAX_R = 40.0
        
        # Emission colors (Realistic Astrophotography H-II palette)
        # H-alpha (656 nm, deep red) + H-beta (486 nm, blue-cyan) + O-III (500 nm, teal)
        color_hii_crimson = np.array([0.95, 0.05, 0.15]) # Pure deep H-alpha red
        color_hii_magenta = np.array([1.00, 0.15, 0.35]) # Standard H-II magenta/pink
        color_hii_salmon  = np.array([1.00, 0.30, 0.40]) # Warmer pink
        color_hii_pale    = np.array([1.00, 0.45, 0.55]) # Pale pink (mixed with O-III)
        
        # Blue/white for reflection nebulae (dust scattering hot young star light)
        color_ref_bright = np.array([0.4, 0.7, 1.0])
        color_ref_dusty  = np.array([0.2, 0.4, 0.8])
        
        for i in range(num_nebulae):
            # 1. Position along spiral arms
            # Preferentially place in the denser parts (r ~ 10-35)
            r = rng.gamma(shape=4.0, scale=4.0) + MIN_R
            r = np.clip(r, MIN_R, MAX_R)
            
            # Select arm
            arm_index = rng.integers(0, ARM_COUNT)
            arm_base_angle = arm_index * (2 * np.pi / ARM_COUNT)
            
            # Ideal log spiral angle
            ideal_theta = arm_base_angle + SPIRAL_TIGHTNESS * np.log(r / INNER_RADIUS)
            
            # Slight offset inside/outside the arm
            theta_offset = rng.normal(0.0, 0.15)
            theta = ideal_theta + theta_offset
            
            # 3D Position
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            
            # Nebulae are very close to the galactic plane (thin disk)
            y = rng.normal(0.0, 0.1)
            
            # 2. Size and Type
            size_roll = rng.random()
            if size_roll < 0.1: # 10% hero regions
                radius = rng.uniform(3.0, 5.0)
                density = rng.uniform(1.0, 1.5)
            elif size_roll < 0.3: # 20% medium regions
                radius = rng.uniform(1.5, 3.0)
                density = rng.uniform(0.7, 1.1)
            else: # 70% small faint regions
                radius = rng.uniform(0.6, 1.5)
                density = rng.uniform(0.4, 0.7)
                
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
