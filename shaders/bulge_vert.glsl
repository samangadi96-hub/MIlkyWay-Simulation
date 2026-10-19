#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_brightness;

out vec3 v_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_visibility;

void main() {
    // Transform to world space
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    
    // Transform to view space
    vec4 view_pos = u_view * world_pos;
    
    // Perspective projection
    gl_Position = u_projection * view_pos;
    
    // Size attenuation based on distance
    float dist = max(0.1, -view_pos.z);
    float proj_size = in_size * (340.0 / dist);
    
    // Flux-conserving subpixel attenuation:
    // When a star is smaller than 1 pixel, its point size is clamped to 1.0.
    // To conserve total flux and prevent millions of unresolved stars from
    // overwhelming the detector, attenuate energy by projected subpixel coverage.
    float subpixel_factor = 1.0;
    if (proj_size < 1.0) {
        subpixel_factor = clamp(pow(proj_size, 1.6), 0.002, 1.0);
    }
    
    gl_PointSize = clamp(proj_size, 1.0, 24.0);
    
    // Pass color out (fade in with u_visibility)
    v_color = in_color * in_brightness * u_visibility * subpixel_factor;
}
