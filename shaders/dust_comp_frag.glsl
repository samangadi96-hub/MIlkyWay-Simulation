#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform sampler2D u_dust;
uniform sampler2D u_nebulaColor;
uniform sampler2D u_nebulaDepth;
uniform sampler2D u_sceneDepth;
uniform bool u_debugDust;
uniform bool u_debugNebula;
uniform float u_nearPlane;
uniform float u_farPlane;

uniform mat4 u_invVP;
uniform vec3 u_camPos;
uniform vec3 u_camForward;

float linearizeDepth(float d) {
    float ndc_z = d * 2.0 - 1.0;
    return (2.0 * u_nearPlane * u_farPlane) / (u_farPlane + u_nearPlane - ndc_z * (u_farPlane - u_nearPlane));
}

void main() {
    vec4 scene = texture(u_scene, uv);
    vec4 dust = texture(u_dust, uv);
    vec4 nebula_col = texture(u_nebulaColor, uv);
    float nebula_depth_norm = texture(u_nebulaDepth, uv).r;

    float dust_trans = dust.r;
    float dust_depth_norm = dust.g;

    if (u_debugDust) {
        fragColor = vec4(dust_trans, dust_trans, dust_trans, 1.0);
        return;
    }
    
    if (u_debugNebula) {
        // Show emission + density (1 - transmittance)
        fragColor = vec4(nebula_col.rgb + vec3(1.0 - nebula_col.a), 1.0);
        return;
    }

    float rawDepth = texture(u_sceneDepth, uv).r;
    float sceneLinearDepth = linearizeDepth(rawDepth);

    // Compute ray direction for consistent Euclidean distance convention
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 target = u_invVP * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(target.xyz / target.w - u_camPos);
    
    // View ray cosine with camera optical axis
    float cosTheta = dot(rd, u_camForward);
    if (cosTheta <= 0.001) cosTheta = 1.0;
    
    // Convert scene planar view depth into Euclidean ray distance from camera,
    // exactly matching the volume ray distances of dust and nebulae.
    float sceneRayDistance = (rawDepth > 0.9999) ? 10000.0 : (sceneLinearDepth / cosTheta);

    float dustFrontDepth = dust_depth_norm * 200.0;
    // We clear to 1.0, so if < 0.999 we hit a nebula
    float nebFrontDepth = nebula_depth_norm * 200.0;
    
    bool has_dust = dust_depth_norm > 0.001;
    bool has_nebula = nebula_depth_norm < 0.999 && (1.0 - nebula_col.a > 0.001 || dot(nebula_col.rgb, vec3(1.0)) > 0.001);
    
    // Scene occlusion factors using consistent Euclidean ray distance
    float dustDepthFactor = 1.0;
    if (has_dust) {
        float transitionWidth = dustFrontDepth * 0.15 + 0.5;
        dustDepthFactor = smoothstep(dustFrontDepth - transitionWidth, dustFrontDepth + transitionWidth * 0.3, sceneRayDistance);
    }
    
    float nebDepthFactor = 1.0;
    if (has_nebula) {
        float n_transitionWidth = nebFrontDepth * 0.15 + 0.5;
        nebDepthFactor = smoothstep(nebFrontDepth - n_transitionWidth, nebFrontDepth + n_transitionWidth * 0.3, sceneRayDistance);
    }
    
    // Cardelli, Clayton & Mathis (CCM 1989) astrophysical interstellar extinction law:
    // Wavelength-dependent optical depth produces natural reddening of background starlight:
    // Red transmits most (tau ~ 0.75), Green nominal (tau ~ 1.00), Blue absorbed most (tau ~ 1.35)
    float t_v = clamp(dust_trans, 0.001, 1.0);
    vec3 ccm_trans = vec3(
        pow(t_v, 0.75),
        pow(t_v, 1.00),
        pow(t_v, 1.35)
    );

    // Depth-aware starlight extinction:
    // Starlight passing through the dust is attenuated and reddened.
    // If an individual foreground star is closer than the dust layer,
    // it preserves its pinpoint brightness in front of the dark lane,
    // while the dust lane STILL attenuates the background disk.
    vec3 extinct_scene = scene.rgb * ccm_trans;
    float fore_factor = 0.0;
    if (has_dust && sceneRayDistance < dustFrontDepth) {
        fore_factor = clamp((dustFrontDepth - sceneRayDistance) / (dustFrontDepth * 0.25 + 0.5), 0.0, 1.0);
    }
    vec3 comp_scene = mix(extinct_scene, scene.rgb, fore_factor * 0.75);

    float eff_neb_trans = mix(1.0, nebula_col.a, nebDepthFactor);
    vec3 eff_neb_emission = nebula_col.rgb;

    vec3 final_color = scene.rgb;

    if (has_dust && has_nebula) {
        if (dustFrontDepth < nebFrontDepth) {
            // Dust is in front of Nebula: attenuate background nebula emission
            vec3 extinct_neb = eff_neb_emission * ccm_trans;
            final_color = comp_scene * eff_neb_trans + extinct_neb;
        } else {
            // Nebula is in front of Dust
            final_color = comp_scene * eff_neb_trans + eff_neb_emission;
        }
    } else if (has_nebula) {
        final_color = scene.rgb * eff_neb_trans + eff_neb_emission;
    } else if (has_dust) {
        final_color = comp_scene;
    }

    // Output composited HDR color in RGB, and effective dust transmittance in Alpha
    // (used by post-processing to protect dark dust silhouettes from bloom wash-out)
    fragColor = vec4(final_color, t_v);
}
