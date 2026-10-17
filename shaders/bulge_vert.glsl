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
    
    // Conserve subpixel flux when zooming out:
    // When a star's geometric projection shrinks below 1 pixel,
    // gl_PointSize stays >= 1.0 to rasterize, and the perceived flux
    // is smoothly scaled by pow(proj_size, 0.35) with a floor of 0.62.
    // This preserves structural density contrast between the dense bar and diffuse disk
    // while preventing distant stars from prematurely vanishing into dark grey.
    float subpixel_factor = clamp(pow(proj_size / 1.0, 0.35), 0.62, 1.0);
    
    gl_PointSize = clamp(proj_size, 1.0, 24.0);
    
    // Pass color out (fade in with u_visibility)
    v_color = in_color * in_brightness * u_visibility * subpixel_factor;
}
