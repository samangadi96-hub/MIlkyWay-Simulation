#version 330

in vec4 v_color;
in vec3 v_worldPos;
in float v_relDepth;

out vec4 fragColor;

uniform int u_sideFilter; // -1 = far side (before BH), +1 = near side (after BH), 0 = both
uniform vec3 u_camPos;
uniform float u_shadowRadius;

void main() {
    if (u_sideFilter < 0 && v_relDepth < 0.0) discard;
    if (u_sideFilter > 0 && v_relDepth >= 0.0) discard;

    vec3 to_pt = v_worldPos - u_camPos;
    float cam_pt_dist = length(to_pt);
    // Avoid near-camera chord artifacts when camera is inside outer orbital halo
    float near_cam_fade = smoothstep(0.28, 0.78, cam_pt_dist);

    // Keep foreground conceptual orbit guide lines cleanly outside the opaque black-hole silhouette
    float shadow_clear = 1.0;
    if (v_relDepth < 0.0 && cam_pt_dist > 1e-4) {
        vec3 ray = to_pt / cam_pt_dist;
        float b_screen = length(cross(u_camPos, ray));
        shadow_clear = smoothstep(u_shadowRadius * 0.98, u_shadowRadius * 1.32, b_screen);
    }

    float alpha = v_color.a * near_cam_fade * shadow_clear;
    if (alpha <= 0.002) discard;
    fragColor = vec4(v_color.rgb, alpha);
}
