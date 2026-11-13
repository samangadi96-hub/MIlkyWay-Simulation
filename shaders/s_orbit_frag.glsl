#version 330

in vec4 v_color;
in vec3 v_worldPos;
in float v_relDepth;
in float v_edgeDist;
in float v_coreFrac;

out vec4 fragColor;

uniform int u_sideFilter; // -1 = far side (before BH), +1 = near side (after BH), 0 = both
uniform vec3 u_camPos;

void main() {
    if (u_sideFilter < 0 && v_relDepth < 0.0) discard;
    if (u_sideFilter > 0 && v_relDepth >= 0.0) discard;
    if (v_color.a <= 0.001) discard;

    float cam_pt_dist = length(v_worldPos - u_camPos);
    // Prevent screen-spanning chord artifacts only right inside the near lens plane
    float near_cam_fade = smoothstep(0.14, 0.38, cam_pt_dist);

    // Smoothly attenuate only the 3D interior of the enlarged visual BH photon-capture sphere
    // (never affects S2 whose rp > 0.25, nor any foreground arc passing in front of Sgr A*)
    float bh_vol_fade = smoothstep(0.115, 0.205, length(v_worldPos));

    // Analytical subpixel anti-aliased edge falloff across ribbon width
    float aa = 1.0 - smoothstep(v_coreFrac, 1.0, abs(v_edgeDist));

    float alpha = v_color.a * aa * near_cam_fade * bh_vol_fade;
    if (alpha <= 0.0015) discard;
    fragColor = vec4(v_color.rgb, alpha);
}
