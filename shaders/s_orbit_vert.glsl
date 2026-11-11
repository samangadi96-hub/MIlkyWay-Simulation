#version 330

in vec3 in_position;
in vec4 in_color;

out vec4 v_color;
out vec3 v_worldPos;
out float v_relDepth;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_bhViewDepth;
uniform float u_visibility;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_projection * view_pos;
    v_worldPos = in_position;
    v_relDepth = (-view_pos.z) - u_bhViewDepth;
    v_color = vec4(in_color.rgb, in_color.a * u_visibility);
}
