#version 330

in vec3 v_color;
out vec4 fragColor;

void main() {
    // Distance from point sprite center [-1, 1]
    vec2 coord = gl_PointCoord * 2.0 - vec2(1.0);
    float r = length(coord);
    
    if (r > 1.0) {
        discard;
    }
    
    // Dual-component astrophysical stellar Point Spread Function (PSF):
    // 1. Sharp central stellar core: preserves crisp individual star pinpoints
    float core = exp(-r * r * 8.0);
    // 2. Restrained diffuse halo skirt: gentle natural stellar glow
    float halo = 0.15 * exp(-r * 3.2) * (1.0 - r);
    
    // Ensure alpha is strictly in [0, 1] to prevent accidental additive amplification
    float alpha = clamp(core + halo, 0.0, 1.0);
    
    fragColor = vec4(v_color, alpha);
}
