#version 330

in vec2 in_offset;
in vec2 in_uv;
in vec3 in_anchor;
in vec4 in_color;

out vec2 v_uv;
out vec4 v_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_alpha;
uniform float u_scale;

void main() {
    // Transform anchor point by galaxy rotation and view matrix
    vec4 anchor_view = u_view * u_model * vec4(in_anchor, 1.0);
    
    // View-space billboard offset ensures labels remain upright and legible
    vec4 view_pos = anchor_view + vec4(in_offset * u_scale, 0.0, 0.0);
    
    gl_Position = u_projection * view_pos;
    v_uv = in_uv;
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
}
