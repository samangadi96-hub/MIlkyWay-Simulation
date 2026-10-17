#version 330

uniform sampler2D scene;
in vec2 uv;
out vec4 fragColor;

const int samples = 48;
const float radius = 0.018;
const float threshold = 0.35;
const float intensity = 1.4;
const float GOLDEN_ANGLE = 2.39996323;

void main() {
    vec3 base_color = texture(scene, uv).rgb;
    float base_lum = dot(base_color, vec3(0.2126, 0.7152, 0.0722));

    // Bloom accumulation (Fibonacci spiral sampling)
    vec3 bloom = vec3(0.0);
    float total_weight = 0.0;

    for (int i = 0; i < samples; i++) {
        float theta = float(i) * GOLDEN_ANGLE;
        float r = radius * sqrt(float(i) / float(samples));
        vec2 offset = vec2(cos(theta), sin(theta)) * r;

        vec3 sc = texture(scene, uv + offset).rgb;
        float brightness = dot(sc, vec3(0.2126, 0.7152, 0.0722));
        float contribution = max(0.0, brightness - threshold);

        if (contribution > 0.0) {
            float weight = exp(-float(i) / float(samples) * 3.0);
            bloom += sc * weight * contribution;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        bloom /= total_weight;
    }

    // CRITICAL: suppress bloom inside the event horizon.
    // If the base pixel is very dark, it is the void - do not add glow.
    float bloom_mask = smoothstep(0.0, 0.06, base_lum);

    vec3 color = base_color + bloom * intensity * bloom_mask;

    // Luminance-guided tonemapping preserves chromatic saturation of stellar populations
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (lum > 0.0001) {
        float mapped_lum = (lum * (2.51 * lum + 0.03)) / (lum * (2.43 * lum + 0.59) + 0.14);
        vec3 per_channel = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
        vec3 lum_scaled = color * (mapped_lum / lum);
        color = mix(per_channel, lum_scaled, 0.42);
        
        // Subtle saturation enhancement to match reference image visual hierarchy
        float final_lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(final_lum), color, 1.28);
    }

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}