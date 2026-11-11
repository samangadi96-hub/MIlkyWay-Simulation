#version 330

in vec3 v_color;
in float v_relDepth;
in float v_projSize;

out vec4 fragColor;

uniform int u_sideFilter; // -1 = far side (before BH), +1 = near side (after BH), 0 = both

void main() {
    if (u_sideFilter < 0 && v_relDepth < 0.0) discard;
    if (u_sideFilter > 0 && v_relDepth >= 0.0) discard;

    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    // Crisp hot core + gentle optical PSF halo (additive HDR compatible)
    float r = sqrt(r2);
    float core = exp(-r2 * 10.5);
    float halo = exp(-r2 * 3.2) * 0.35;
    float edge_fade = 1.0 - smoothstep(0.75, 1.0, r);
    float profile = (core + halo) * edge_fade;

    // Normalize intensity gently at large point sizes so sprites never turn into flat opaque blobs
    float norm = mix(1.0, 0.65, smoothstep(5.0, 14.0, v_projSize));
    fragColor = vec4(v_color * profile * norm, profile);
}

