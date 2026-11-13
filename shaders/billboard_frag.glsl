#version 330

uniform float aspectRatio;
uniform float u_camDistance;
uniform float u_diskSquish;
uniform float u_lodWeight;
uniform float u_diskThickness;
uniform float u_glowIntensity;
uniform float u_innerRingStrength;
uniform float u_dopplerStrength;
uniform float u_diskOuterRadius;
uniform float u_innerDisk;
uniform float u_outerDisk;
uniform float u_time;
uniform float u_volumetricDetail;
uniform float u_proceduralStarWeight;

// Real camera basis from Python (position is in BH-local space)
uniform vec3 u_cameraPosition;
uniform vec3 u_cameraForward;
uniform vec3 u_cameraRight;
uniform vec3 u_cameraUp;

in vec2 frag_pos;
out vec4 fragColor;

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 get_stars(vec2 p) {
    p += vec2(12.34, 56.78); 
    vec2 id = floor(p * 90.0);
    vec2 f = fract(p * 90.0);
    float star = smoothstep(0.96, 1.0, hash(id));
    star *= 1.0 - smoothstep(0.1, 0.5, length(f - 0.5));
    vec3 color = mix(vec3(0.6, 0.8, 1.0), vec3(1.0, 0.7, 0.4), hash(id + 1.0));
    return color * star * 4.0; 
}

void main() {
    vec2 uv = vec2(frag_pos.x * aspectRatio, frag_pos.y);
    
    // Scale from screen space to BH-local space using real camera distance
    float cam_radius = max(length(u_cameraPosition), 0.1);
    float fov_zoom = 2.0;
    
    vec2 p = uv * (cam_radius / fov_zoom);
    float r = length(p);
    
    vec2 disk_p = vec2(p.x, p.y / max(u_diskSquish * u_diskThickness, 0.01));
    float disk_r = length(disk_p);
    float shadow_radius = 2.6;
    
    vec3 color = vec3(0.0);
    float transmittance = 1.0;
    
    float glow_radius = 15.0; 
    if (r < glow_radius && u_glowIntensity > 0.0) {
        float intensity = pow(1.0 - (r / glow_radius), 4.0);
        vec3 core_color = vec3(1.0, 0.9, 0.7);
        float alpha = intensity * 0.8 * u_glowIntensity;
        color += core_color * alpha;
        transmittance *= (1.0 - alpha);
    }
    
    if (r < shadow_radius) {
        // Inside shadow: opaque black — NO background stars
        color = vec3(0.0);
        transmittance = 0.0;
    } else {
        float inner_r = max(u_innerDisk, 2.5);
        float outer_r = max(u_outerDisk, 12.0);
        
        if (disk_r > inner_r && disk_r < outer_r) {
            vec3 final_disk_color = vec3(0.0);
            float final_alpha = 0.0;
            
            float doppler = 1.0 + 0.8 * (disk_p.x / disk_r) * min(1.0, 6.0 / disk_r);
            float doppler_boost = pow(doppler, 3.0);
            
            // 1. Photon Ring / Inner Ring (approx 2.6 to 4.5)
            float ring_mask = smoothstep(inner_r, 3.5, disk_r) * (1.0 - smoothstep(3.5, 4.5, disk_r));
            vec3 ring_color = mix(vec3(1.0, 0.4, 0.0), vec3(1.0, 0.95, 0.8), 1.0 - smoothstep(3.0, 4.0, disk_r));
            final_disk_color += ring_color * ring_mask;
            final_alpha += ring_mask * 0.9;
            
            // 2. Inner Accretion Disk (approx 3.5 to 6.5)
            float inner_mask = smoothstep(3.5, 4.2, disk_r) * (1.0 - smoothstep(4.8, 5.5, disk_r));
            float temp_inner = clamp(1.0 - (disk_r - 3.5) / 1.5, 0.0, 1.0);
            vec3 inner_color = mix(vec3(0.5, 0.05, 0.0), vec3(1.0, 0.8, 0.2), pow(temp_inner, 1.5));
            final_disk_color += inner_color * inner_mask;
            final_alpha += inner_mask * 0.85;
            
            // 3. Subtle Turbulence (approx 4.5 to 7.5)
            float gas_mask = smoothstep(4.5, 5.2, disk_r) * (1.0 - smoothstep(5.8, 6.5, disk_r));
            float angle = atan(disk_p.y, disk_p.x) + u_time * 1.5; 
            float cheap_noise = sin(angle * 4.0 + disk_r * 5.0) * 0.5 + 0.5; 
            cheap_noise = mix(1.0, cheap_noise, u_volumetricDetail); // Fade out turbulence over distance
            vec3 gas_color = mix(vec3(0.4, 0.05, 0.0), vec3(0.8, 0.4, 0.0), cheap_noise);
            gas_color *= mix(1.0, doppler_boost, 0.6);
            final_disk_color += gas_color * gas_mask;
            final_alpha += gas_mask * 0.7;
            
            // 4. Outer Disk with strong Doppler (approx 5.5 to u_outerDisk)
            float outer_mask = smoothstep(5.5, 6.5, disk_r) * (1.0 - smoothstep(outer_r - 4.0, outer_r, disk_r));
            vec3 hot_white = vec3(1.0, 0.95, 0.8);
            vec3 dark_red = vec3(0.3, 0.02, 0.0);
            float temp_outer = clamp(1.0 - (disk_r - 5.5) / (outer_r - 5.5), 0.0, 1.0);
            vec3 outer_color = mix(dark_red, hot_white, pow(temp_outer, 2.0));
            outer_color *= doppler_boost;
            final_disk_color += outer_color * outer_mask;
            final_alpha += outer_mask * 0.8;
            
            final_alpha = clamp(final_alpha, 0.0, 0.95);
            color += final_disk_color;
            transmittance *= (1.0 - final_alpha);
        }
    }
    
    // Background Stars — use real camera basis
    vec3 ray_dir = normalize(u_cameraForward * fov_zoom + uv.x * u_cameraRight + uv.y * u_cameraUp);
    
    vec2 sky_uv = vec2(atan(ray_dir.z, ray_dir.x), asin(clamp(ray_dir.y, -1.0, 1.0)));
    vec3 background_stars = get_stars(sky_uv * 10.0) * u_proceduralStarWeight;
    
    vec3 final_color = (background_stars * transmittance) + color;
    
    fragColor = vec4(final_color * u_lodWeight, mix(1.0, transmittance, u_lodWeight));
}
