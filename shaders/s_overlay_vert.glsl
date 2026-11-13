#version 330

in vec2 in_pos;   // screen pixels (x: 0..W right, y: 0..H top-down converted to NDC)
in vec2 in_uv;
in vec4 in_color;

out vec2 v_uv;
out vec4 v_color;

uniform vec2 u_viewport; // (width, height)

void main() {
    vec2 ndc = vec2(
        (in_pos.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (in_pos.y / u_viewport.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}

