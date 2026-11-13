#version 330

in vec3 in_position;
in vec4 in_color;

out vec4 v_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_alpha;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
}
