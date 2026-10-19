#version 330

in vec3 in_position;
in vec2 in_texcoord;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec2 v_uv;
out vec3 v_worldPos;

void main() {
    v_uv = in_texcoord;
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_worldPos = world_pos.xyz;
    gl_Position = u_projection * u_view * world_pos;
}
