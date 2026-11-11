#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_galaxyTex;
uniform float u_knee;
uniform float u_cap;

void main() {
    // Resolve galaxy-only HDR radiance prior to black-hole compositing:
    // Luminance-preserving soft shoulder prevents bright nuclear/bar overlap from
    // blowing out to flat white while leaving pixels at L <= u_knee bit-for-bit intact
    // and preserving spectral hue ratios.
    vec4 galaxy = texture(u_galaxyTex, uv);
    vec3 rgb = max(vec3(0.0), galaxy.rgb);
    float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    if (lum > u_knee) {
        float mapped = u_knee + u_cap * tanh((lum - u_knee) / u_cap);
        rgb *= (mapped / lum);
    }
    fragColor = vec4(rgb, 1.0);
}

