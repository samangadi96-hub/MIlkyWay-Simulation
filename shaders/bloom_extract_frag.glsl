#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_image;
uniform float u_threshold;
uniform float u_knee;

void main() {
    vec4 c = texture(u_image, uv);
    vec3 rgb = c.rgb;
    
    // Perceived luminance (ITU-R BT.709)
    float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    
    // Smooth quadratic threshold curve (Karis / Unreal style)
    float soft = lum - u_threshold + u_knee;
    soft = clamp(soft, 0.0, 2.0 * u_knee);
    soft = (soft * soft) / (4.0 * u_knee + 1e-5);
    
    float contrib = max(soft, lum - u_threshold) / max(lum, 1e-4);
    
    vec3 bright = rgb * clamp(contrib, 0.0, 15.0);
    
    // Pass dust transmittance in alpha
    fragColor = vec4(bright, c.a);
}
