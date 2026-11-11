#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_brightness;

out vec3 v_color;
out float v_projSize;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_visibility;
uniform int u_depthPass;

void main() {
    // Transform to world space
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    
    // Transform to view space
    vec4 view_pos = u_view * world_pos;
    
    // Perspective projection
    gl_Position = u_projection * view_pos;
    
    // 1. Intrinsic stellar luminosity (separate from projected screen coverage)
    vec3 lum_rgb = in_color * in_brightness * u_visibility;

    // 2. Projected angular/pixel diameter based on camera distance
    float dist = max(0.1, -view_pos.z);
    float proj_size = in_size * (340.0 / dist);
    v_projSize = proj_size;

    if (u_depthPass != 0) {
        // Depth-only pass: keep sprites tight and cull negligible subpixel specks
        // so dim foreground halo stars do not punch false holes in dust occlusion.
        float lum_scalar = dot(lum_rgb, vec3(0.2126, 0.7152, 0.0722));
        float est_peak = lum_scalar * min(1.0, proj_size * proj_size);
        if (est_peak < 0.025) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            gl_PointSize = 1.0;
        } else {
            gl_PointSize = clamp(proj_size, 1.0, 24.0);
        }
    } else {
        // Clamp point size >= 2.0 px so subpixel stars filter cleanly across a
        // 2x2 neighborhood without discards or alignment sparkle; PSF normalizes flux.
        gl_PointSize = clamp(proj_size, 2.0, 24.0);
    }

    // Pass intrinsic luminosity only; fragment shader applies coverage & PSF once
    v_color = lum_rgb;
}
