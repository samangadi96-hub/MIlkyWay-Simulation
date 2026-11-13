#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_image;
uniform vec2 u_direction;
uniform float u_radius;

// 9-tap normalized Gaussian weights (sigma ~ 2.5)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 offset = u_direction * u_radius;
    vec4 result = texture(u_image, uv) * weights[0];
    
    for (int i = 1; i < 5; i++) {
        vec2 off = offset * float(i);
        result += texture(u_image, uv + off) * weights[i];
        result += texture(u_image, uv - off) * weights[i];
    }
    
    fragColor = result;
}
