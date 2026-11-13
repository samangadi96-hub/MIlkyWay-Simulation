#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform sampler2D u_dust;        // Unified: RGB=T_ccm, A=front_depth | Legacy: R=T_scalar, G=front_depth
uniform sampler2D u_ismBins;     // Cumulative green-band optical depth across 4 slab bins
uniform sampler2D u_nebulaColor; // RGB=dust-coupled emission, A=nebula transmittance
uniform sampler2D u_nebulaDepth; // R=nebula front depth
uniform sampler2D u_sceneDepth;

uniform bool u_debugDust;
uniform bool u_debugNebula;
uniform int u_useUnifiedISM;
uniform float u_nearPlane;
uniform float u_farPlane;

uniform mat4 u_invVP;
uniform mat4 u_galaxyModel;
uniform vec3 u_camPos;
uniform vec3 u_camForward;

const float SLAB_H = 0.72;
const float CYL_R  = 47.2;

float linearizeDepth(float d) {
    float ndc_z = d * 2.0 - 1.0;
    return (2.0 * u_nearPlane * u_farPlane) / (u_farPlane + u_nearPlane - ndc_z * (u_farPlane - u_nearPlane));
}

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
    return vec2((-b - s) / (2.0 * a), (-b + s) / (2.0 * a));
}

void main() {
    vec4 scene = texture(u_scene, uv);
    vec4 dust  = texture(u_dust, uv);
    vec4 nebula_col = texture(u_nebulaColor, uv);

    // Wavelength-dependent CCM transmittance from unified pass (or legacy scalar fallback)
    vec3 total_T = (u_useUnifiedISM != 0)
        ? clamp(dust.rgb, vec3(0.0005), vec3(1.0))
        : vec3(pow(clamp(dust.r, 0.001, 1.0), 0.75),
               pow(clamp(dust.r, 0.001, 1.0), 1.00),
               pow(clamp(dust.r, 0.001, 1.0), 1.35));

    if (u_debugDust) {
        fragColor = vec4(total_T, 1.0);
        return;
    }

    if (u_debugNebula) {
        fragColor = vec4(nebula_col.rgb + vec3(1.0 - nebula_col.a) * 0.25, 1.0);
        return;
    }

    float rawDepth = texture(u_sceneDepth, uv).r;
    float sceneLinearDepth = linearizeDepth(rawDepth);

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 target = u_invVP * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(target.xyz / target.w - u_camPos);
    float cosTheta = dot(rd, u_camForward);
    if (cosTheta <= 0.001) cosTheta = 1.0;
    float sceneRayDistance = (rawDepth > 0.9999) ? 10000.0 : (sceneLinearDepth / cosTheta);

    if (u_useUnifiedISM != 0) {
        vec4 bins = texture(u_ismBins, uv);
        float tau_sum = bins.x + bins.y + bins.z + bins.w;
        float total_lum_T = dot(total_T, vec3(0.2126, 0.7152, 0.0722));
        bool has_dust = (tau_sum > 1e-4) || ((1.0 - total_lum_T) > 0.001);

        // Compute depth-resolved foreground transmittance T_star using 4-bin optical depth curve
        vec3 T_star = total_T;
        float f_depth = 1.0;
        if (has_dust && rawDepth <= 0.9999 && tau_sum > 1e-4) {
            vec3 ro_local = (u_galaxyModel * vec4(u_camPos, 1.0)).xyz;
            vec3 rd_local = normalize((u_galaxyModel * vec4(rd, 0.0)).xyz);
            vec2 t_slab = intersectSlab(ro_local, rd_local, SLAB_H);
            vec2 t_cyl  = intersectCylinder(ro_local, rd_local, CYL_R);
            float t_start = max(max(0.0, t_slab.x), t_cyl.x);
            float t_end   = min(min(t_slab.y, t_cyl.y), 160.0);
            if (t_end > t_start) {
                float pos4 = clamp((sceneRayDistance - t_start) / (t_end - t_start), 0.0, 1.0) * 4.0;
                float tau_prefix = 0.0;
                tau_prefix += bins.x * clamp(pos4, 0.0, 1.0);
                tau_prefix += bins.y * clamp(pos4 - 1.0, 0.0, 1.0);
                tau_prefix += bins.z * clamp(pos4 - 2.0, 0.0, 1.0);
                tau_prefix += bins.w * clamp(pos4 - 3.0, 0.0, 1.0);
                f_depth = clamp(tau_prefix / tau_sum, 0.0, 1.0);
                T_star = pow(clamp(total_T, vec3(0.0005), vec3(1.0)), vec3(f_depth));
            }
        }

        // Preserve foreground visibility of close nebulae relative to scene stars cleanly
        float neb_d_norm = texture(u_nebulaDepth, uv).r;
        bool has_nebula = (neb_d_norm < 0.999) && ((1.0 - nebula_col.a) > 0.001 || dot(nebula_col.rgb, vec3(1.0)) > 0.001);
        float eff_neb_trans = 1.0;
        if (has_nebula) {
            float nebFrontDepth = neb_d_norm * 200.0;
            float n_tw = nebFrontDepth * 0.15 + 0.5;
            float nebDepthFactor = smoothstep(nebFrontDepth - n_tw, nebFrontDepth + n_tw * 0.3, sceneRayDistance);
            eff_neb_trans = mix(1.0, nebula_col.a, nebDepthFactor);
        }

        // Unified emission is already dust-attenuated in 3D along the front-to-back ray;
        // never attenuate nebula_col.rgb a second time!
        vec3 final_color = scene.rgb * T_star * eff_neb_trans + nebula_col.rgb;

        float star_lum_T = clamp(dot(T_star, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        float eff_bloom_trans = has_dust ? star_lum_T : 1.0;
        if (has_nebula) {
            float neb_glow = clamp((1.0 - nebula_col.a) * 0.4 + dot(nebula_col.rgb, vec3(0.333)), 0.0, 1.0);
            eff_bloom_trans = max(eff_bloom_trans, neb_glow);
        }
        fragColor = vec4(final_color, eff_bloom_trans);
        return;
    }

    // =========================================================================
    // Legacy Fallback Path (for older single-attachment scripts using u_dust/u_nebula separately)
    // =========================================================================
    float t_v = clamp(dust.r, 0.001, 1.0);
    float dustFrontDepth = dust.g * 200.0;
    bool has_dust = (dust.g > 0.0001 && dust.g < 0.999) && ((1.0 - t_v) > 0.001);

    float neb_d = texture(u_nebulaDepth, uv).r;
    float nebFrontDepth = neb_d * 200.0;
    bool has_nebula = (neb_d < 0.999) && (1.0 - nebula_col.a > 0.001 || dot(nebula_col.rgb, vec3(1.0)) > 0.001);

    float nebDepthFactor = 1.0;
    if (has_nebula) {
        float n_transitionWidth = nebFrontDepth * 0.15 + 0.5;
        nebDepthFactor = smoothstep(nebFrontDepth - n_transitionWidth, nebFrontDepth + n_transitionWidth * 0.3, sceneRayDistance);
    }

    vec3 extinct_scene = scene.rgb * total_T;
    float fore_factor = 0.0;
    if (has_dust && sceneRayDistance < dustFrontDepth) {
        fore_factor = clamp((dustFrontDepth - sceneRayDistance) / (dustFrontDepth * 0.25 + 0.5), 0.0, 1.0);
    }
    vec3 comp_scene = mix(extinct_scene, scene.rgb, fore_factor * 0.75);

    float eff_neb_trans = mix(1.0, nebula_col.a, nebDepthFactor);
    vec3 final_color = scene.rgb;
    if (has_dust && has_nebula) {
        final_color = (dustFrontDepth < nebFrontDepth)
            ? (comp_scene * eff_neb_trans + nebula_col.rgb * total_T)
            : (comp_scene * eff_neb_trans + nebula_col.rgb);
    } else if (has_nebula) {
        final_color = scene.rgb * eff_neb_trans + nebula_col.rgb;
    } else if (has_dust) {
        final_color = comp_scene;
    }

    float eff_bloom_trans = has_dust ? mix(t_v, 1.0, fore_factor * 0.75) : 1.0;
    fragColor = vec4(final_color, eff_bloom_trans);
}
