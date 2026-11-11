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

// Optional 3D noise volume (smooth fallback hash noise when unbound by legacy callers)
uniform sampler3D u_noiseVol3D;
uniform int u_hasISMTextures;

const float SPIRAL_TIGHTNESS = 2.5;

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

vec2 intersectSlab(vec3 ro, vec3 rd, float H) {
    if (abs(rd.y) < 1e-6) {
        if (abs(ro.y) < H) return vec2(0.0, 200.0);
        else return vec2(0.0, -1.0);
    }
    float t1 = (-H - ro.y) / rd.y;
    float t2 = (H - ro.y) / rd.y;
    return vec2(max(0.0, min(t1, t2)), max(t1, t2));
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 target = u_invVP * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(target.xyz / target.w - u_camPos);
    vec3 ro = u_camPos;

    vec3 ro_local = (u_galaxyModel * vec4(ro, 1.0)).xyz;
    vec3 rd_local = normalize((u_galaxyModel * vec4(rd, 0.0)).xyz);

    float slab_H = 0.72; // Strictly confined to thin star-forming galactic plane
    vec2 t_bounds = intersectSlab(ro_local, rd_local, slab_H);

    float transmittance = 1.0;
    vec3 accumulated_emission = vec3(0.0);
    float front_depth = -1.0;

    if (t_bounds.y > t_bounds.x && u_nebulaWeight > 0.001) {
        // Collect up to 8 closest intersecting nebulae sorted front-to-back
        const int MAX_HITS = 8;
        int hit_count = 0;
        int hit_idx[8];
        float hit_tin[8];
        float hit_tout[8];

        int count_max = min(u_numNebulae, 150);
        for (int j = 0; j < count_max; j++) {
            vec3 n_pos = u_nebulae_pos[j].xyz;
            float n_rad = u_nebulae_pos[j].w;
            if (n_rad < 0.05) continue;

            float bound_r = n_rad * 1.45;
            vec3 oc = n_pos - ro_local;
            float t_proj = dot(oc, rd_local);
            float d2 = dot(oc, oc) - t_proj * t_proj;
            float r2 = bound_r * bound_r;

            if (d2 < r2) {
                float h = sqrt(r2 - d2);
                float tin = max(t_bounds.x, t_proj - h);
                float tout = min(t_bounds.y, t_proj + h);

                if (tout > tin && tout > 0.0) {
                    if (hit_count < MAX_HITS) {
                        hit_idx[hit_count] = j;
                        hit_tin[hit_count] = tin;
                        hit_tout[hit_count] = tout;
                        hit_count++;
                    } else {
                        int far_k = 0;
                        for (int m = 1; m < MAX_HITS; m++) {
                            if (hit_tin[m] > hit_tin[far_k]) far_k = m;
                        }
                        if (tin < hit_tin[far_k]) {
                            hit_idx[far_k] = j;
                            hit_tin[far_k] = tin;
                            hit_tout[far_k] = tout;
                        }
                    }
                }
            }
        }

        // Sort front-to-back
        if (hit_count > 1) {
            for (int a = 0; a < hit_count - 1; a++) {
                for (int b = a + 1; b < hit_count; b++) {
                    if (hit_tin[b] < hit_tin[a]) {
                        float tmp_tin = hit_tin[a]; hit_tin[a] = hit_tin[b]; hit_tin[b] = tmp_tin;
                        float tmp_tout = hit_tout[a]; hit_tout[a] = hit_tout[b]; hit_tout[b] = tmp_tout;
                        int tmp_idx = hit_idx[a]; hit_idx[a] = hit_idx[b]; hit_idx[b] = tmp_idx;
                    }
                }
            }
        }

        vec2 pixel = gl_FragCoord.xy;
        float jitter = interleavedGradientNoise(pixel);

        for (int k = 0; k < hit_count; k++) {
            if (transmittance < 0.01) break;

            int j = hit_idx[k];
            float tin = hit_tin[k];
            float tout = hit_tout[k];
            vec3 n_pos = u_nebulae_pos[j].xyz;
            float n_rad = u_nebulae_pos[j].w;
            vec4 c_data = u_nebulae_col[j];
            vec3 color = c_data.rgb;
            float strength = c_data.a;

            float seed_a = float(j) * 2.39996;
            float yaw = atan(n_pos.z, n_pos.x) + atan(SPIRAL_TIGHTNESS, 1.0) + 0.45 * sin(seed_a);
            float cy = cos(yaw);
            float sy = sin(yaw);
            float stretch_x = 1.12 + 0.22 * cos(seed_a * 1.7);
            float stretch_z = 0.70 + 0.16 * sin(seed_a * 2.3);
            float stretch_y = 0.38;

            const int CLOUD_STEPS = 10;
            float dt = (tout - tin) / float(CLOUD_STEPS);
            float t = tin + dt * jitter;

            for (int s = 0; s < CLOUD_STEPS; s++) {
                if (t > tout || transmittance < 0.01) break;

                vec3 p_local = ro_local + rd_local * t;
                vec3 d = p_local - n_pos;
                vec3 d_rot = vec3(d.x * cy + d.z * sy, d.y, -d.x * sy + d.z * cy);
                vec3 q = d_rot / vec3(n_rad * stretch_x, n_rad * stretch_y, n_rad * stretch_z);

                vec4 n_detail = (u_hasISMTextures != 0)
                    ? texture(u_noiseVol3D, p_local * 0.28 + vec3(0.37, 0.61, 0.19))
                    : vec4(fbm3(p_local * 3.8 + vec3(float(j) * 3.17)));
                q += (n_detail.rgb - 0.5) * 0.46;

                float d_norm = length(q);
                if (d_norm < 1.0) {
                    float falloff = 1.0 - smoothstep(0.08, 1.0, d_norm);
                    vec3 knot_off = vec3(0.20 * sin(seed_a * 3.1), 0.0, 0.16 * cos(seed_a * 2.7));
                    float core_r = length(q - knot_off);
                    float core_focus = exp(-core_r * core_r * 7.5);

                    float fil = (u_hasISMTextures != 0) ? mix(n_detail.g, n_detail.a, 0.45) : n_detail.r;
                    float gas_density = clamp(falloff * mix(0.28, 1.40, fil) * (0.48 + 0.95 * core_focus), 0.0, 2.2);
                    gas_density = max(0.0, gas_density - 0.08) / 0.92;

                    if (gas_density > 0.005) {
                        bool is_ref = (color.b > color.r * 1.05);
                        vec3 final_col = is_ref
                            ? mix(color, vec3(0.76, 0.90, 1.05), clamp(core_focus * 0.55, 0.0, 0.60)) * mix(0.75, 1.15, gas_density)
                            : mix(color * vec3(1.06, 0.54, 0.64),
                                  mix(color * vec3(1.10, 0.85, 0.90), vec3(1.00, 0.60, 0.68), 0.45),
                                  clamp(core_focus * gas_density * 0.95, 0.0, 0.85));

                        float em = (0.38 * gas_density + 1.05 * core_focus * gas_density) * strength * 1.38;
                        float local_sigma = gas_density * strength * 0.34 * u_nebulaWeight;

                        if (front_depth < 0.0) front_depth = t;

                        float d_tau = local_sigma * dt;
                        float T_step = exp(-d_tau);
                        float w_ds = (d_tau > 1e-3) ? ((1.0 - T_step) / d_tau) : (1.0 - 0.5 * d_tau);
                        accumulated_emission += final_col * em * dt * w_ds * u_nebulaWeight * transmittance;
                        transmittance *= T_step;
                    }
                }
                t += dt;
            }
        }
    }

    float encoded_depth = (front_depth < 0.0) ? 1.0 : front_depth / 200.0;

    fragColor = vec4(accumulated_emission, transmittance);
    fragDepth = encoded_depth;
}
