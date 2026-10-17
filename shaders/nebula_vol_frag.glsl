#version 330

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;
layout(location = 1) out float fragDepth;

uniform vec3 u_camPos;
uniform mat4 u_invVP;
uniform mat4 u_galaxyModel;

uniform float u_nebulaWeight;
uniform float u_camDist;
uniform int u_numNebulae;

uniform vec4 u_nebulae_pos[150];
uniform vec4 u_nebulae_col[150];

// ============================================================
// Hash and Noise
// ============================================================

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                   mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                   mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

float fbm3(vec3 p) {
    float f = 0.0;
    float w = 0.5;
    for (int i = 0; i < 3; i++) {
        f += w * noise(p);
        p *= 2.07;
        w *= 0.5;
    }
    return f;
}

float interleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

// ============================================================
// Slab intersection
// ============================================================

vec2 intersectSlab(vec3 ro, vec3 rd, float H) {
    if (abs(rd.y) < 1e-6) {
        if (abs(ro.y) < H) return vec2(0.0, 200.0);
        else return vec2(0.0, -1.0);
    }
    float t1 = (-H - ro.y) / rd.y;
    float t2 = (H - ro.y) / rd.y;
    float tmin = min(t1, t2);
    float tmax = max(t1, t2);
    tmin = max(0.0, tmin);
    return vec2(tmin, tmax);
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 target = u_invVP * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(target.xyz / target.w - u_camPos);
    vec3 ro = u_camPos;

    float slab_H = 1.2; // Confined strictly to the thin star-forming galactic plane
    vec2 t_bounds = intersectSlab(ro, rd, slab_H);

    float transmittance = 1.0;
    vec3 accumulated_emission = vec3(0.0);
    float front_depth = -1.0;

    if (t_bounds.y > t_bounds.x && u_nebulaWeight > 0.001) {
        float t = t_bounds.x;
        float t_end = min(t_bounds.y, 160.0);

        int steps = 36;
        float dt = (t_end - t) / float(steps);

        vec2 pixel = gl_FragCoord.xy;
        float jitter = interleavedGradientNoise(pixel);
        t += dt * jitter;

        for (int i = 0; i < 36; i++) {
            if (t > t_end || transmittance < 0.01) break;

            vec3 p = ro + rd * t;
            vec4 p_local_4 = u_galaxyModel * vec4(p, 1.0);
            vec3 p_local = p_local_4.xyz;

            // Performance optimization: nebulae are strictly confined to the thin galactic disk
            // (|y| <= 1.0) and radial star-forming ring (5.5 <= r <= 41.5).
            float r_xz = length(p_local.xz);
            if (abs(p_local.y) > 1.0 || r_xz < 5.5 || r_xz > 41.5) {
                t += dt;
                continue;
            }

            float local_density = 0.0;
            vec3 local_emission = vec3(0.0);

            for(int j = 0; j < u_numNebulae; j++) {
                vec3 n_pos = u_nebulae_pos[j].xyz;
                float n_rad = u_nebulae_pos[j].w;
                float dist = distance(p_local, n_pos);
                
                if (dist < n_rad) {
                    float d_norm = dist / n_rad;
                    float falloff = 1.0 - smoothstep(0.0, 1.0, d_norm);
                    
                    if (falloff > 0.01) {
                        float ns = 4.2;
                        vec3 p_noise = p_local * ns + vec3(float(j) * 3.17);
                        
                        // Multi-scale FBM with domain warping for filamentary clumps
                        float n_base = fbm3(p_noise);
                        vec3 p_warp = p_noise + vec3(n_base * 1.6);
                        float n_detail = fbm3(p_warp * 2.2);
                        
                        float shape = (n_base * 0.55 + n_detail * 0.45);
                        // Natural irregular boundary: threshold with falloff so edges are wispy filaments
                        float cloud_density = max(0.0, (shape - 0.28) / 0.72) * pow(falloff, 1.4);
                        
                        if (cloud_density > 0.01) {
                            vec4 c_data = u_nebulae_col[j];
                            vec3 color = c_data.rgb;
                            float strength = c_data.a;
                            
                            // Emission: deep crimson base with pink highlights in the densest clumps
                            vec3 core_color = mix(color, vec3(1.0, 0.45, 0.65), clamp(cloud_density * 1.2, 0.0, 0.85));
                            float em = cloud_density * strength * 2.2;
                            
                            local_density += cloud_density * strength * 0.75;
                            local_emission += core_color * em;
                        }
                    }
                }
            }

            if (local_density > 0.001) {
                if (front_depth < 0.0) front_depth = t;
                
                float step_opacity = local_density * dt * 0.8;
                float step_transmittance = exp(-step_opacity);
                
                accumulated_emission += local_emission * dt * u_nebulaWeight * transmittance;
                transmittance *= step_transmittance;
            }

            t += dt;
        }
    }
    
    transmittance = mix(1.0, transmittance, u_nebulaWeight);

    // Write depth. If front_depth < 0, output 1.0 (no hit), else normalize.
    float encoded_depth = (front_depth < 0.0) ? 1.0 : front_depth / 200.0;

    fragColor = vec4(accumulated_emission, transmittance);
    fragDepth = encoded_depth;
}
