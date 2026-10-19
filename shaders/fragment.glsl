#version 330

uniform float aspectRatio;
uniform float u_time;
uniform float u_innerDisk;
uniform float u_outerDisk;
uniform float u_camDistance;
uniform float u_lodDetail;
uniform float u_volumetricDetail;
uniform float u_diskThickness;
uniform float u_proceduralStarWeight;
uniform float u_lodWeight;

// Real camera basis from Python (position is in BH-local space)
uniform vec3 u_cameraPosition;
uniform vec3 u_cameraForward;
uniform vec3 u_cameraRight;
uniform vec3 u_cameraUp;

in vec2 frag_pos;
out vec4 fragColor;

// ==========================================
// UTILITIES & NOISE
// ==========================================
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D Hash for Volumetric Gas
float hash3D(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

// 3D Noise function for gas turbulence
float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    
    float n = mix(
        mix(mix(hash3D(i + vec3(0,0,0)), hash3D(i + vec3(1,0,0)), u.x),
            mix(hash3D(i + vec3(0,1,0)), hash3D(i + vec3(1,1,0)), u.x), u.y),
        mix(mix(hash3D(i + vec3(0,0,1)), hash3D(i + vec3(1,0,1)), u.x),
            mix(hash3D(i + vec3(0,1,1)), hash3D(i + vec3(1,1,1)), u.x), u.y), u.z);
    return n;
}

// 4-Octave 3D fBm
float fbm3D(vec3 p) {
    float f = 0.0;
    float amp = 0.5;
    for(int i = 0; i < 4; i++) {
        f += amp * noise3D(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return f;
}

// --- PROCEDURAL STARFIELD ---
vec3 get_stars(vec2 p) {
    p += vec2(12.34, 56.78); 
    vec2 id = floor(p * 90.0);
    vec2 f = fract(p * 90.0);
    float star = smoothstep(0.96, 1.0, hash(id));
    star *= 1.0 - smoothstep(0.1, 0.5, length(f - 0.5));
    vec3 color = mix(vec3(0.6, 0.8, 1.0), vec3(1.0, 0.7, 0.4), hash(id + 1.0));
    return color * star * 4.0; 
}

void main()
{
    vec2 uv = vec2(frag_pos.x * aspectRatio, frag_pos.y);

    // 1. 3D CAMERA SETUP — uses real camera from Python
    float Rs = 1.0;

    vec3 ray_origin = u_cameraPosition;
    float cam_radius = length(ray_origin);

    float fov_zoom = 2.0; 
    vec3 ray_dir = normalize(u_cameraForward * fov_zoom + uv.x * u_cameraRight + uv.y * u_cameraUp);

    // 2. PHYSICS ENGINE & VOLUMETRIC VARIABLES
    float dt = mix(0.2, 0.1, u_lodDetail);        
    int max_steps = int(mix(50.0, 300.0, u_lodDetail)); 
    
    vec3 p = ray_origin; 
    vec3 v = ray_dir;

    vec3 L = cross(p, v);
    float h2 = dot(L, L); 

    bool hit_black_hole = false;
    vec3 accumulated_gas_color = vec3(0.0);
    float transmittance = 1.0; // How much background light makes it through the gas

    // Gas Color Palette
    vec3 hot_white = vec3(1.0, 0.95, 0.8);
    vec3 bright_orange = vec3(1.0, 0.4, 0.05);
    vec3 dark_red = vec3(0.4, 0.05, 0.0);

    // 3. THE GEODESIC RAY-MARCHING LOOP
    for (int i = 0; i < max_steps; i++) {
        float r2 = dot(p, p);
        float r = sqrt(r2);

        if (r < Rs) {
            hit_black_hole = true;
            break;
        }

        if (r > cam_radius + 50.0) break; 

        // ==========================================
        // VOLUMETRIC ACCRETION DISK SAMPLING
        // ==========================================
        // Only calculate gas if we are near the equator and within the disk radius
        if (abs(p.y) < 0.5 && r > 2.5 && r < 12.0) {
            
            // Keplerian Rotation: inner gas orbits much faster than outer gas
            float orbital_velocity = 8.0 * pow(r, -1.5); 
            float angle = atan(p.z, p.x) + u_time * orbital_velocity;
            
            // Rotate the sampling coordinates to simulate fluid motion
            vec3 sampling_pos = vec3(r * cos(angle), p.y, r * sin(angle));

            // Density calculation: Thicker in middle, falls off at edges
            float p_y_scaled = p.y / max(u_diskThickness, 0.05);
            float vertical_falloff = exp(-(p_y_scaled * p_y_scaled) * 20.0);
            float radial_falloff = smoothstep(2.5, 4.0, r) * (1.0 - smoothstep(8.0, 12.0, r));
            
            // Apply 3D turbulent noise to the rotating volume
            float noise = fbm3D(sampling_pos * 1.5 - vec3(0.0, u_time * 1.0, 0.0));
            // Fade out noise as volumetric detail goes down
            noise = mix(1.0, noise, max(u_volumetricDetail, 0.01));
            
            float density = vertical_falloff * radial_falloff * noise * 2.5;

            if (density > 0.05) {
                // Color Gradient based on distance (Hotter near the black hole)
                float temp = clamp(1.0 - (r - 2.5) / 7.0, 0.0, 1.0);
                vec3 local_color = mix(dark_red, bright_orange, temp);
                local_color = mix(local_color, hot_white, pow(temp, 3.0));

                // Relativistic Doppler Beaming (Approximation)
                // Gas moving towards the camera is on the left (-x side generally)
                float doppler = 1.0 + 0.8 * (p.x / r) * min(1.0, 5.0/r); 
                local_color *= pow(doppler, 3.0); // Doppler boosts brightness exponentially

                // Accumulate the glowing gas
                accumulated_gas_color += local_color * density * dt * transmittance * 5.0;
                
                // The gas absorbs light behind it (makes the disk opaque)
                transmittance *= exp(-density * dt * 2.0);
            }
        }

        // ==========================================
        // GENERAL RELATIVITY (Spacetime Curvature)
        // ==========================================
        
        // LOCAL 3-9 FIX ONLY
        // Distance-aware adaptive step size to bridge the empty space between the camera and the disk.
        // Active only when camera.distance is between 3.0 and 9.0.
        float current_dt = dt;
        float t_fix = smoothstep(2.0, 2.3, u_camDistance) * (1.0 - smoothstep(9.0, 10.5, u_camDistance));
        
        if (t_fix > 0.001 && r > 12.0) {
            float empty_space = max(0.0, cam_radius - 12.0);
            float safe_steps = max(20.0, float(max_steps) - 240.0); // leave 240 steps for the disk
            float max_boost = max(dt, empty_space / safe_steps);
            
            // Smoothly ramp down the boost as we approach the accretion disk (12.0 to 18.0)
            // to ensure we don't overshoot it.
            float approach_factor = smoothstep(12.0, 18.0, r);
            float boosted_dt = mix(dt, max_boost, approach_factor);
            
            // Apply the correction only based on how deep we are in the 3-9 range
            current_dt = mix(dt, boosted_dt, t_fix);
        }
        
        vec3 acceleration = -1.5 * Rs * h2 / (r2 * r2 * r) * p;
        v += acceleration * current_dt;
        v = normalize(v); 
        p += v * current_dt;
        
        // Early exit if the gas becomes completely opaque
        if (transmittance < 0.01) break; 
    }

    // 4. FINAL COMPOSITING
    //
    // Output convention for blend mode (ONE, SRC_ALPHA):
    //   result = fragColor.rgb * 1 + background * fragColor.a
    //
    // fragColor.rgb = emission (gas accumulated along ray)
    // fragColor.a   = transmittance (how much background shows through)

    if (hit_black_hole) {
        // Event horizon: pure black behind accumulated gas.
        // transmittance = 0 ensures NO background stars bleed through.
        fragColor = vec4(accumulated_gas_color * u_lodWeight, mix(1.0, 0.0, u_lodWeight));
    } else {
        // Procedural background stars (only when not fully covered by 3D stars)
        vec2 sky_uv = vec2(atan(v.z, v.x), asin(clamp(v.y, -1.0, 1.0)));
        vec3 background_stars = get_stars(sky_uv * 10.0) * u_proceduralStarWeight;

        // Gas emission + attenuated procedural stars
        vec3 final_color = (background_stars * transmittance) + accumulated_gas_color;

        // Output: emission in RGB, transmittance in alpha
        // The 3D star layer underneath will be multiplied by this transmittance
        fragColor = vec4(final_color * u_lodWeight, mix(1.0, transmittance, u_lodWeight));
    }
}