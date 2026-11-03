#version 330

in vec3 v_color;
out vec4 fragColor;

void main() {
    // Distance from point sprite center [-1, 1]
    vec2 coord = gl_PointCoord * 2.0 - vec2(1.0);
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }
    float r = sqrt(r2);
    
    // Dual-component astrophysical stellar Point Spread Function (PSF):
    // 1. Sharp central stellar core: crisp pinpoint star
    float core = exp(-r2 * 10.0);
    // 2. Restrained diffuse halo skirt: gentle natural stellar glow fading smoothly to edge
    float halo = 0.12 * (1.0 - r) * (1.0 - r);
    
    float alpha = clamp(core + halo, 0.0, 1.0);
    fragColor = vec4(v_color, alpha);
}
