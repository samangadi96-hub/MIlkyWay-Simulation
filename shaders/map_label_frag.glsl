#version 330

uniform sampler2D u_atlas;

in vec2 v_uv;
in vec4 v_color;

out vec4 fragColor;

void main() {
    vec4 tex = texture(u_atlas, v_uv);
    // Combine texture alpha with label color and opacity
    fragColor = vec4(v_color.rgb, tex.a * v_color.a);
}
