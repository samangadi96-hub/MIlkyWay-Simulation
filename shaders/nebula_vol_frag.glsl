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

    // Transform ray to local galaxy space
    vec3 ro_local = (u_galaxyModel * vec4(ro, 1.0)).xyz;
    vec3 rd_local = normalize((u_galaxyModel * vec4(rd, 0.0)).xyz);

    float slab_H = 1.4; // Confined strictly to the thin star-forming galactic plane
    vec2 t_bounds = intersectSlab(ro_local, rd_local, slab_H);

    float transmittance = 1.0;
    vec3 accumulated_emission = vec3(0.0);
    float front_depth = -1.0;

    if (t_bounds.y > t_bounds.x && u_nebulaWeight > 0.001) {
        // Collect clouds intersecting this ray (per-cloud ray-bounding sphere intersection)
        int hit_count = 0;
        int hit_idx[4];
        float hit_tin[4];
        float hit_tout[4];

        for (int j = 0; j < u_numNebulae; j++) {
            vec3 n_pos = u_nebulae_pos[j].xyz;
            float n_rad = u_nebulae_pos[j].w;
            if (n_rad < 0.05) continue;

            vec3 oc = n_pos - ro_local;
            float t_proj = dot(oc, rd_local);
            float d2 = dot(oc, oc) - t_proj * t_proj;
            float r2 = n_rad * n_rad;

            if (d2 < r2) {
                float h = sqrt(r2 - d2);
                float tin = max(t_bounds.x, t_proj - h);
                float tout = min(t_bounds.y, t_proj + h);

                if (tout > tin && tout > 0.0) {
                    if (hit_count < 4) {
                        hit_idx[hit_count] = j;
                        hit_tin[hit_count] = tin;
                        hit_tout[hit_count] = tout;
                        hit_count++;
                    }
                }
            }
        }

        // Sort hit clouds front-to-back
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

        // Localized sampling through each intersected cloud
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

            int cloud_steps = 8;
            float dt = (tout - tin) / float(cloud_steps);
            float t = tin + dt * jitter;

            for (int s = 0; s < 8; s++) {
                if (t > tout || transmittance < 0.01) break;

                vec3 p_local = ro_local + rd_local * t;
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
                        float cloud_density = max(0.0, (shape - 0.28) / 0.72) * pow(falloff, 1.4);

                        if (cloud_density > 0.01) {
                            vec3 core_color = mix(color, vec3(1.0, 0.45, 0.65), clamp(cloud_density * 1.2, 0.0, 0.85));
                            float em = cloud_density * strength * 2.2;
                            float local_density = cloud_density * strength * 0.75;

                            if (front_depth < 0.0) front_depth = t;

                            float step_opacity = local_density * dt * 0.8;
                            float step_transmittance = exp(-step_opacity);

                            accumulated_emission += core_color * em * dt * u_nebulaWeight * transmittance;
                            transmittance *= step_transmittance;
                        }
                    }
                }

                t += dt;
            }
        }
    }

    transmittance = mix(1.0, transmittance, u_nebulaWeight);

    // Write depth. If front_depth < 0, output 1.0 (no hit), else normalize.
    float encoded_depth = (front_depth < 0.0) ? 1.0 : front_depth / 200.0;

    fragColor = vec4(accumulated_emission, transmittance);
    fragDepth = encoded_depth;
}
