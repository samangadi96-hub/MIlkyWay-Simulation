#version 330

in vec3 v_color;
in float v_projSize;
out vec4 fragColor;

uniform int u_depthPass;

// Continuous unit-disk integral of clamp(exp(-10*r2) + 0.12*(1-r)^2, 0, 1):
// integrated pixel flux sum equals (PSF_UNIT_INTEGRAL / 4) * projSize^2.
const float PSF_UNIT_INTEGRAL = 0.375210;

void main() {
    // Distance from point sprite center [-1, 1]
    vec2 coord = gl_PointCoord * 2.0 - vec2(1.0);
    float r2 = dot(coord, coord);

    if (u_depthPass != 0) {
        // Compact core occlusion only in depth pass
        if (r2 > 0.36) {
            discard;
        }
        fragColor = vec4(1.0);
        return;
    }

    // 3 & 4. Projected pixel coverage & normalized fragment PSF:
    // Rendered sprite diameter P is clamped >= 2.0 so subpixel stars filter smoothly
    // across a 2x2 neighborhood without discards or alignment graininess, while
    // coverage_scale = (s / P)^2 preserves exact integrated flux ~ 0.0938 * s^2.
    float P = clamp(v_projSize, 2.0, 24.0);
    float s = min(v_projSize, 24.0);
    float coverage_scale = (s * s) / (P * P);

    // Smooth C1 partition-of-unity subpixel kernel (pixel sum == 1.0 at P = 2.0)
    float wx = smoothstep(1.0, 0.0, abs(coord.x));
    float wy = smoothstep(1.0, 0.0, abs(coord.y));
    float aa_kernel = wx * wy;

    // Dual-component resolved astrophysical Point Spread Function (PSF):
    // 1. Sharp central stellar core: crisp pinpoint star
    // 2. Restrained diffuse halo skirt: gentle natural stellar glow fading smoothly to edge
    float psf_resolved = 0.0;
    if (r2 <= 1.0) {
        float r = sqrt(r2);
        float core = exp(-r2 * 10.0);
        float halo = 0.12 * (1.0 - r) * (1.0 - r);
        psf_resolved = clamp(core + halo, 0.0, 1.0);
    }

    // Continuous blend between flux-conserving subpixel splat and resolved dual PSF
    float resolve_mix = smoothstep(2.0, 4.5, v_projSize);
    float alpha = coverage_scale * mix(PSF_UNIT_INTEGRAL * aa_kernel, psf_resolved, resolve_mix);

    if (alpha <= 1e-8) {
        discard;
    }

    // 5. Final emitted energy under GL_SRC_ALPHA, GL_ONE: v_color * alpha
    fragColor = vec4(v_color, alpha);
}
