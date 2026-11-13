#version 330

uniform sampler2D u_tex;
uniform float u_alpha;

in vec2 v_uv;
in vec4 v_color;

out vec4 fragColor;

void main() {
    // Allow solid colored primitives (leader lines) when v_uv.x < 0.0
    if (v_uv.x < -0.5) {
        fragColor = vec4(v_color.rgb, v_color.a * u_alpha);
    } else {
        vec4 tex = texture(u_tex, v_uv);
        fragColor = vec4(tex.rgb * v_color.rgb, tex.a * v_color.a * u_alpha);
    }
}

