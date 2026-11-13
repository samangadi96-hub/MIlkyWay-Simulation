#version 330

in vec2 v_uv;
layout(location = 0) out vec4 fragDustTrans;   // RGB = CCM transmittance, A = encoded dust front depth
layout(location = 1) out vec4 fragTauBins;     // Cumulative green-band optical depth across 4 ray bins
layout(location = 2) out vec4 fragNebulaColor; // RGB = dust-coupled nebula + scattered glow, A = nebula transmittance
layout(location = 3) out float fragNebulaDepth;// Normalized nebula front depth

uniform vec3 u_camPos;
uniform mat4 u_invVP;
uniform mat4 u_galaxyModel; // inverse rotation of galaxy

uniform float u_dustWeight;
uniform float u_nebulaWeight;
uniform float u_camDist;

// ISM tunables (with backward-compatible uniform names preserved)
uniform float u_dustOpacity;
uniform float u_dustWidth;
uniform float u_dustArmOffset;
uniform float u_dustNoiseScale;
uniform float u_dustDensity;
uniform float u_dustHeightScale;
uniform float u_dustLaneWidth;
uniform float u_dustDiffuseWidth;

// Precomputed 3D ISM textures
uniform sampler2D u_ismSpiralMap;  // R=thin_lane, G=thick_diff, B=mol_ridge, A=radial_mask
uniform sampler3D u_molCloudTex;   // R=mol_env, G=mol_core, B=illum_warm, A=illum_cool
uniform sampler3D u_noiseVol3D;    // R=fbm, G=ridged_filament, B=worley_clump, A=fine_wisp
uniform int u_hasISMTextures;
uniform int u_unifiedISM;

// Nebula catalogue uniforms
uniform int u_numNebulae;
uniform vec4 u_nebulae_pos[150];
uniform vec4 u_nebulae_col[150];

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
const float SPIRAL_TIGHTNESS = 2.5;

// Verified physical unit conversion: 1 simulation unit ~= 700 pc
const float H_DUST_THIN_BASE  = 0.1157; // ~81 pc
const float H_DUST_THICK_BASE = 0.2171; // ~152 pc
const float SLAB_H = 0.72;              // Tight vertical dust + molecular gas slab (>6x thin scale height)
const float CYL_R  = 47.2;

// ============================================================
// Interleaved gradient noise (temporally & spatially stable dither)
// ============================================================

float interleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

// ============================================================
// Analytical Slab & Cylinder Intersections
// ============================================================

vec2 intersectSlab(vec3 ro, vec3 rd, float H) {
    if (abs(rd.y) < 1e-6) {
        if (abs(ro.y) < H) return vec2(0.0, 200.0);
        return vec2(0.0, -1.0);
    }
    float t1 = (-H - ro.y) / rd.y;
    float t2 = (H - ro.y) / rd.y;
    return vec2(max(0.0, min(t1, t2)), max(t1, t2));
}

vec2 intersectCylinder(vec3 ro, vec3 rd, float R) {
    float a = rd.x * rd.x + rd.z * rd.z;
    if (a < 1e-7) {
        float r2 = ro.x * ro.x + ro.z * ro.z;
        return (r2 < R * R) ? vec2(0.0, 200.0) : vec2(0.0, -1.0);
    }
    float b = 2.0 * (ro.x * rd.x + ro.z * rd.z);
    float c = ro.x * ro.x + ro.z * ro.z - R * R;
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return vec2(0.0, -1.0);
    float s = sqrt(disc);
    float t1 = (-b - s) / (2.0 * a);
    float t2 = (-b + s) / (2.0 * a);
    return vec2(min(t1, t2), max(t1, t2));
}

// Evaluate single nebula volume emission & dust self-opacity at p_local
void evalNebulaSample(
    int idx, vec3 p_local, vec4 n_broad, vec4 n_detail,
    out vec3 out_emit, out float out_sigma
) {
    out_emit = vec3(0.0);
    out_sigma = 0.0;

    vec4 pos_r = u_nebulae_pos[idx];
    vec3 n_pos = pos_r.xyz;
    float n_rad = pos_r.w;
    if (n_rad < 0.05) return;

    // Seeded local spiral-sheared orientation frame per nebula
    float seed_a = float(idx) * 2.39996;
    float yaw = atan(n_pos.z, n_pos.x) + atan(SPIRAL_TIGHTNESS, 1.0) + 0.45 * sin(seed_a);
    float cy = cos(yaw);
    float sy = sin(yaw);

    vec3 d = p_local - n_pos;
    // Rotate into anisotropic elliptical nebula frame stretched slightly along arm shear
    vec3 d_rot = vec3(
        d.x * cy + d.z * sy,
        d.y,
        -d.x * sy + d.z * cy
    );

    float stretch_x = 1.12 + 0.22 * cos(seed_a * 1.7);
    float stretch_z = 0.70 + 0.16 * sin(seed_a * 2.3);
    float stretch_y = 0.38; // Nebulae tightly confined near star-forming thin disk

    vec3 q = d_rot / vec3(n_rad * stretch_x, n_rad * stretch_y, n_rad * stretch_z);
    // Domain warp with 3D turbulence so boundaries are jagged & ragged, never round discs
    q += (n_detail.rgb - 0.5) * 0.46 + (n_broad.rgb - 0.5) * 0.20;

    float d_norm = length(q);
    if (d_norm >= 1.0) return;

    float radial_env = 1.0 - smoothstep(0.08, 1.0, d_norm);
    // Asymmetric ionized cavity/knot center slightly offset inside parent complex
    vec3 knot_off = vec3(0.20 * sin(seed_a * 3.1), 0.0, 0.16 * cos(seed_a * 2.7));
    float core_r = length(q - knot_off);
    float core_focus = exp(-core_r * core_r * 7.5);

    // Multi-scale filamentary & clumpy gas density inside nebula envelope
    float fil = mix(n_detail.g, n_broad.g, 0.35);
    float clump = smoothstep(0.20, 0.82, n_detail.b * 0.65 + n_broad.r * 0.35);
    float wisp = mix(0.28, 1.40, fil) * mix(0.42, 1.30, clump);

    float gas_density = clamp(radial_env * wisp * (0.48 + 0.95 * core_focus), 0.0, 2.2);
    // Crisp zero outside carved filaments
    gas_density = max(0.0, gas_density - 0.08) / 0.92;
    if (gas_density <= 0.005) return;

    vec4 c_data = u_nebulae_col[idx];
    vec3 base_col = c_data.rgb;
    float strength = c_data.a;

    bool is_reflection = (base_col.b > base_col.r * 1.05);
    vec3 final_col;
    if (is_reflection) {
        // Blue reflection nebulae scatter hot OB light -> cool sapphire outer wisps, pale blue-white core
        vec3 core_tint = mix(base_col, vec3(0.76, 0.90, 1.05), clamp(core_focus * 0.55, 0.0, 0.60));
        final_col = core_tint * mix(0.75, 1.15, gas_density);
    } else {
        // Deep H-alpha crimson in outer ragged filaments, rich ruby/rose in ionized blister cores
        vec3 h_deep = base_col * vec3(1.06, 0.54, 0.64);
        vec3 h_core = mix(base_col * vec3(1.10, 0.85, 0.90), vec3(1.00, 0.60, 0.68), 0.45);
        final_col = mix(h_deep, h_core, clamp(core_focus * gas_density * 0.95, 0.0, 0.85));
    }

    // Natural luminosity profile: crisp bright ionized knots inside softer crimson skirts
    float em_intensity = (0.38 * gas_density + 1.05 * core_focus * gas_density) * strength * 1.38;
    out_emit = final_col * em_intensity;
    out_sigma = gas_density * strength * 0.38;
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 target = u_invVP * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(target.xyz / target.w - u_camPos);
    vec3 ro = u_camPos;

    vec3 ro_local = (u_galaxyModel * vec4(ro, 1.0)).xyz;
    vec3 rd_local = normalize((u_galaxyModel * vec4(rd, 0.0)).xyz);

    vec2 t_slab = intersectSlab(ro_local, rd_local, SLAB_H);
    vec2 t_cyl  = intersectCylinder(ro_local, rd_local, CYL_R);

    vec3  T_ccm = vec3(1.0);
    vec4  bin_tau = vec4(0.0);
    vec3  accum_emission = vec3(0.0);
    float neb_transmittance = 1.0;
    float dust_front_depth = -1.0;
    float neb_front_depth = -1.0;

    bool active_ism = (u_hasISMTextures != 0) && (u_dustWeight > 0.001 || (u_unifiedISM != 0 && u_nebulaWeight > 0.001));

    if (active_ism && t_slab.y > t_slab.x && t_cyl.y > t_cyl.x) {
        float t_start = max(max(0.0, t_slab.x), t_cyl.x);
        float t_end   = min(min(t_slab.y, t_cyl.y), 160.0);

        if (t_end > t_start) {
            float path_len = t_end - t_start;

            // Collect closest up to 8 intersecting nebulae sorted front-to-back by tin
            const int MAX_HITS = 8;
            int   hit_count = 0;
            int   hit_idx[8];
            float hit_tin[8];
            float hit_tout[8];
            float min_neb_rad = 10.0;
            float neb_span_min = 1e9;
            float neb_span_max = -1.0;

            if (u_unifiedISM != 0 && u_nebulaWeight > 0.001) {
                int count_max = min(u_numNebulae, 150);
                for (int j = 0; j < count_max; j++) {
                    vec3 n_pos = u_nebulae_pos[j].xyz;
                    float n_rad = u_nebulae_pos[j].w;
                    if (n_rad < 0.05) continue;

                    // Generous spherical bound to encompass anisotropic stretch & domain warp
                    float bound_r = n_rad * 1.48;
                    vec3 oc = n_pos - ro_local;
                    float t_proj = dot(oc, rd_local);
                    float d2 = dot(oc, oc) - t_proj * t_proj;
                    float r2 = bound_r * bound_r;

                    if (d2 < r2) {
                        float h = sqrt(r2 - d2);
                        float tin  = max(t_start, t_proj - h);
                        float tout = min(t_end,   t_proj + h);
                        if (tout > tin && tout > 0.0) {
                            neb_span_min = min(neb_span_min, tin);
                            neb_span_max = max(neb_span_max, tout);
                            min_neb_rad = min(min_neb_rad, n_rad);

                            if (hit_count < MAX_HITS) {
                                hit_idx[hit_count] = j;
                                hit_tin[hit_count] = tin;
                                hit_tout[hit_count] = tout;
                                hit_count++;
                            } else {
                                // Replace farthest entry if this nebula starts closer along ray
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
            }

            // Adaptive march step: fast vertically face-on, dense along edge-on paths without banding
            float target_dt = clamp(path_len / 52.0, 0.085, 0.58);
            int base_steps = clamp(int(ceil(path_len / target_dt)), 14, 56);
            float dt_base = path_len / float(base_steps);

            vec2 pixel = gl_FragCoord.xy;
            float jitter = interleavedGradientNoise(pixel);
            float t = t_start + dt_base * (jitter * 0.85 + 0.075);

            // CCM (1989) wavelength-dependent extinction ratios:
            // Diffuse warm amber-brown dust vs neutral dark dense molecular cloud cores
            const vec3 K_DIFFUSE = vec3(0.70, 1.00, 1.44);
            const vec3 K_DENSE   = vec3(0.91, 1.00, 1.11);

            for (int iter = 0; iter < 92; iter++) {
                if (t > t_end) break;
                if (dot(T_ccm, vec3(0.333)) < 0.004) break;

                // Determine adaptive sub-step if inside active nebula span
                float ds = dt_base;
                bool in_neb_span = (hit_count > 0) && (t >= neb_span_min - 0.08) && (t <= neb_span_max + 0.08);
                if (in_neb_span) {
                    ds = min(dt_base, clamp(min_neb_rad * 0.32, 0.09, 0.24));
                }
                ds = min(ds, max(0.02, t_end - t + ds * 0.5));

                vec3 p_local = ro_local + rd_local * t;
                float r = length(p_local.xz);

                if (r >= 0.68 && r <= 47.0 && abs(p_local.y) <= SLAB_H) {
                    // 1. Two-tier tileable 3D multi-scale noise (domain warp + filaments + clumps + fine wisps)
                    float ns = max(u_dustNoiseScale, 0.2) * 1.25;
                    vec4 n_broad  = texture(u_noiseVol3D, p_local * (0.048 * ns) + vec3(0.11, 0.27, 0.43));
                    vec3 p_warp   = p_local + (n_broad.rgb - 0.5) * vec3(1.20, 0.16, 1.20);
                    vec4 n_detail = texture(u_noiseVol3D, p_warp * (0.24 * ns) + vec3(0.37, 0.61, 0.19));

                    // 2. Sample spiral macro ISM structure
                    vec2 uv_sp = (p_local.xz + (n_broad.rb - 0.5) * 0.46 + 47.5) / 95.0;
                    vec4 sp = texture(u_ismSpiralMap, clamp(uv_sp, 0.001, 0.999));
                    float thin_map  = sp.r;
                    float diff_map  = sp.g;
                    float mol_ridge = sp.b;
                    float rad_mask  = sp.a;

                    // 3. Stratified vertical profiles (81 pc thin dust vs 152 pc diffuse dust)
                    float flare = 1.0 + 0.15 * smoothstep(18.0, 44.0, r);
                    float h_thin  = max(u_dustHeightScale, 0.2) * H_DUST_THIN_BASE  * flare;
                    float h_thick = max(u_dustHeightScale, 0.2) * H_DUST_THICK_BASE * flare;
                    float y2 = p_local.y * p_local.y;
                    float vert_thin  = exp(-0.5 * y2 / (h_thin  * h_thin));
                    float vert_thick = exp(-0.5 * y2 / (h_thick * h_thick));

                    // 4. Thin filamentary + clumpy arm/spur/bar dust
                    float fil = mix(n_detail.g, n_detail.a, 0.35);
                    float clumps = smoothstep(0.16, 0.80, n_broad.r * 0.42 + n_detail.b * 0.58);
                    float edge_wisp = 0.42 + 0.82 * n_detail.a;
                    float sigma_thin = thin_map * vert_thin * (0.26 + 1.28 * fil * edge_wisp) * (0.52 + 0.96 * clumps);

                    // 5. Weaker, smoother thicker diffuse cirrus component
                    float sigma_thick = diff_map * vert_thick * (0.50 + 0.72 * n_broad.r) * (0.72 + 0.42 * n_detail.a);

                    // 6. Sample precomputed 3D Molecular Clouds + Local Nebula Illumination (skip far off-plane empty regions)
                    vec4 mol = vec4(0.0);
                    if (abs(p_local.y) <= 0.65 && (diff_map > 0.045 || in_neb_span)) {
                        vec3 uvw_base = vec3(
                            (p_local.x + 47.0) / 94.0,
                            (p_local.z + 47.0) / 94.0,
                            (p_local.y + 0.90) / 1.80
                        );
                        vec3 uvw_warp = (n_detail.rgb - 0.5) * vec3(0.011, 0.011, 0.045);
                        mol = texture(u_molCloudTex, clamp(uvw_base + uvw_warp, 0.001, 0.999));
                    }

                    // Sculpt irregular fractal-like molecular cloud clumps & cores
                    float mol_erode = smoothstep(0.18, 0.76, n_detail.b * 0.56 + n_detail.g * 0.28 + n_detail.a * 0.16);
                    float ridge_knot = max(0.0, mol_ridge * vert_thin * (n_detail.b * n_detail.g - 0.14) * 2.4) * rad_mask;
                    float sigma_mol_env  = mol.r * mix(0.32, 1.45, mol_erode) * rad_mask + ridge_knot * 0.60;
                    float sigma_mol_core = mol.g * smoothstep(0.22, 0.82, n_detail.b * 0.65 + n_broad.b * 0.35) * rad_mask + ridge_knot * 0.40;

                    float w_dust = u_dustWeight * u_dustOpacity * u_dustDensity;
                    float coeff_diff = (sigma_thin * 0.56 + sigma_thick * 0.21) * w_dust;
                    float coeff_mol  = (sigma_mol_env * 0.92 + sigma_mol_core * 2.10) * w_dust;

                    // 7. Nebula Emission & Coupled 3D Self-Absorption
                    vec3  S_neb = vec3(0.0);
                    float coeff_neb = 0.0;
                    if (in_neb_span) {
                        for (int k = 0; k < MAX_HITS; k++) {
                            if (k >= hit_count) break;
                            if (t >= hit_tin[k] - 0.04 && t <= hit_tout[k] + 0.04) {
                                vec3 n_em;
                                float n_sig;
                                evalNebulaSample(hit_idx[k], p_local, n_broad, n_detail, n_em, n_sig);
                                S_neb += n_em;
                                coeff_neb += n_sig;
                            }
                        }
                        S_neb *= u_nebulaWeight;
                        coeff_neb *= u_nebulaWeight;
                    }

                    // 8. Subtle Illumination of Nearby Dust Filaments by H-II / OB Photons
                    float dust_scat_carrier = sigma_thin * 0.42 + sigma_mol_env * 0.50 + sigma_thick * 0.12;
                    vec3 illum_col = mol.b * vec3(0.95, 0.24, 0.30) + mol.a * vec3(0.36, 0.72, 1.00);
                    vec3 S_scatter = illum_col * dust_scat_carrier * u_nebulaWeight * 0.24;

                    // Track front depths
                    if (dust_front_depth < 0.0 && (coeff_diff + coeff_mol) > 0.008) {
                        dust_front_depth = t;
                    }
                    if (neb_front_depth < 0.0 && (dot(S_neb, vec3(1.0)) > 0.015 || coeff_neb > 0.008)) {
                        neb_front_depth = t;
                    }

                    // 9. Exact Front-to-Back Radiative Transfer Step:
                    //    L_out = L_in * exp(-d_tau) + S_source * (1 - exp(-d_tau)) / alpha
                    vec3 d_tau = coeff_diff * K_DIFFUSE * ds + coeff_mol * K_DENSE * ds + vec3(coeff_neb * ds);
                    vec3 T_step = exp(-d_tau);
                    vec3 slab_weight = mix(
                        1.0 - 0.5 * d_tau,
                        (1.0 - T_step) / max(d_tau, vec3(1e-4)),
                        greaterThan(d_tau, vec3(1e-3))
                    );

                    vec3 S_total = S_neb + S_scatter;
                    vec3 step_emitted = S_total * ds * slab_weight;

                    accum_emission += step_emitted * T_ccm;
                    T_ccm *= T_step;
                    if (coeff_neb > 0.0) {
                        neb_transmittance *= exp(-coeff_neb * ds);
                    }

                    // Accumulate optical depth into 4 equal bins along [t_start, t_end]
                    // for exact per-pixel star depth splitting in dust_comp_frag.glsl
                    float bin_f = clamp((t - t_start) / max(path_len, 1e-4), 0.0, 0.9999) * 4.0;
                    int b_idx = int(floor(bin_f));
                    float d_tau_v = d_tau.g;
                    if (b_idx == 0) bin_tau.x += d_tau_v;
                    else if (b_idx == 1) bin_tau.y += d_tau_v;
                    else if (b_idx == 2) bin_tau.z += d_tau_v;
                    else bin_tau.w += d_tau_v;
                }

                t += ds;
            }
        }
    }

    float encoded_dust_depth = (dust_front_depth < 0.0) ? 1.0 : clamp(dust_front_depth / 200.0, 0.0002, 0.998);
    float encoded_neb_depth  = (neb_front_depth  < 0.0) ? 1.0 : clamp(neb_front_depth  / 200.0, 0.0002, 0.998);

    if (u_unifiedISM != 0) {
        fragDustTrans   = vec4(T_ccm, encoded_dust_depth);
        fragTauBins     = bin_tau;
        fragNebulaColor = vec4(accum_emission, neb_transmittance);
        fragNebulaDepth = encoded_neb_depth;
    } else {
        // Legacy fallback signature (RG = scalar transmittance, front depth)
        fragDustTrans   = vec4(T_ccm.g, encoded_dust_depth, 0.0, 1.0);
        fragTauBins     = bin_tau;
        fragNebulaColor = vec4(0.0, 0.0, 0.0, 1.0);
        fragNebulaDepth = 1.0;
    }
}
