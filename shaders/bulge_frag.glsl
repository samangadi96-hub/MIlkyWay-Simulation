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
    float core = exp(-r * r * 6.5);
    // 2. Soft diffuse halo skirt: allows adjacent stars in dense clusters, the central bar,
    //    bulge, and spiral arms to blend smoothly into authentic luminous stellar glow
    float halo = 0.28 * exp(-r * 2.4) * (1.0 - r);
    
    float alpha = core + halo;
    
    fragColor = vec4(v_color, alpha);
}
