#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform sampler2D u_bloomTight;
uniform sampler2D u_bloomBroad;

uniform float u_exposure;
uniform float u_bloomTightIntensity;
uniform float u_bloomBroadIntensity;
uniform float u_dustProtection;
uniform int u_debugPost;

// Reference Narkowicz ACES filmic tone mapping curve
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec4 scene_sample = texture(u_scene, uv);
    vec3 base_color = scene_sample.rgb;
    float dust_trans = scene_sample.a; // Transmittance from dust compositor (1.0 = clear, 0.0 = opaque dust)

    // Sample multi-scale bloom textures (hardware bilinear filtered)
    vec3 bloom_tight = texture(u_bloomTight, uv).rgb;
    vec3 bloom_broad = texture(u_bloomBroad, uv).rgb;

    if (u_debugPost == 1) {
        fragColor = vec4(bloom_tight, 1.0);
        return;
    }
    if (u_debugPost == 2) {
        fragColor = vec4(bloom_broad, 1.0);
        return;
    }

    // Combine multi-tiered bloom
    vec3 total_bloom = bloom_tight * u_bloomTightIntensity + bloom_broad * u_bloomBroadIntensity;

    // 1. Dust Silhouette Protection:
    // Foreground dust physically obscures background bloom as well as stars.
    // Attenuating bloom by dust transmittance keeps dark dust lanes inky black and high-contrast
    // right across the luminous galactic core and star-forming arms.
    float dust_atten = mix(pow(clamp(dust_trans, 0.0, 1.0), 1.6), 1.0, 1.0 - u_dustProtection);
    total_bloom *= dust_atten;

    if (u_debugPost == 3) {
        fragColor = vec4(total_bloom, 1.0);
        return;
    }
    if (u_debugPost == 4) {
        total_bloom = vec3(0.0);
    }

    // 2. Black Hole Event Horizon Protection:
    // Ensure bloom does not wash out the central black hole shadow
    float base_lum = dot(base_color, vec3(0.2126, 0.7152, 0.0722));
    float bh_shadow = smoothstep(0.0001, 0.025, base_lum);
    total_bloom *= bh_shadow;

    // 3. HDR Scene Accumulation:
    // Base stars remain 100% pin-sharp because base_color is sampled at full native resolution
    vec3 hdr = (base_color + total_bloom) * u_exposure;

    // 4. Astrophotography ACES Filmic Tone Mapping:
    // Uses a hybrid luminance-chrominance formulation to compress extreme highlights smoothly
    // while preventing the warm golden core and vibrant H-II regions from bleaching to flat white
    float lum = dot(hdr, vec3(0.2126, 0.7152, 0.0722));
    if (lum > 0.0001) {
        float mapped_lum = ACESFilm(vec3(lum)).r;
        vec3 per_channel = ACESFilm(hdr);
        vec3 lum_scaled = hdr * (mapped_lum / lum);
        // Prioritize luminance-scaled tone mapping to preserve authentic spectral colors
        hdr = mix(per_channel, lum_scaled, 0.75);

        // 5. Subtle Astrophotographical Color Grading (inspired by reference targets):
        // Warm golden core and bar (Ref 1 & Ref 2)
        vec3 warm_gold_tint = vec3(1.05, 0.99, 0.91);
        float core_factor = smoothstep(0.25, 1.2, lum) * 0.40;
        hdr = mix(hdr, hdr * warm_gold_tint, core_factor);

        // Cool blue-white balance in diffuse spiral arms
        vec3 cool_arm_tint = vec3(0.97, 0.99, 1.04);
        float arm_factor = smoothstep(0.04, 0.30, lum) * (1.0 - smoothstep(0.35, 0.75, lum)) * 0.22;
        hdr = mix(hdr, hdr * cool_arm_tint, arm_factor);

        // Rich color saturation preservation for stellar populations and H-II knots
        float final_lum = dot(hdr, vec3(0.2126, 0.7152, 0.0722));
        hdr = mix(vec3(final_lum), hdr, 1.18);
    }

    // 6. Deep Neutral Velvet Black Space:
    // Pure black background without grey milkiness or crushed shadow artifacts
    hdr = max(vec3(0.0), hdr);

    // 7. Single Display Gamma Correction (sRGB gamma 2.2):
    fragColor = vec4(pow(clamp(hdr, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}