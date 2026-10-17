#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform vec3 u_camPos;
uniform mat4 u_invVP;
uniform mat4 u_galaxyModel; // inverse rotation of galaxy

uniform float u_dustWeight;

// Tunable params
uniform float u_dustOpacity;
uniform float u_dustWidth;
uniform float u_dustArmOffset;
uniform float u_dustNoiseScale;
uniform float u_dustDensity;

// New params for upgraded dust
uniform float u_dustHeightScale;   // vertical thickness control
uniform float u_dustLaneWidth;     // narrow dense lane half-width
uniform float u_dustDiffuseWidth;  // broader diffuse envelope half-width
uniform float u_camDist;           // camera distance for LOD

// ============================================================
// Hash and Noise — coherent 3D value noise with smooth interp
// ============================================================

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    // Quintic interpolation for smoother derivatives
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                   mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                   mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

// 5-octave FBM with configurable lacunarity
float fbm5(vec3 p) {
    float f = 0.0;
    float w = 0.5;
    float totalW = 0.0;
    for (int i = 0; i < 5; i++) {
        f += w * noise(p);
        totalW += w;
        p *= 2.03;   // slight lacunarity variation avoids grid artifacts
        w *= 0.48;
    }
    return f / totalW;
}

// 3-octave FBM for cheaper large-scale features
float fbm3(vec3 p) {
    float f = 0.0;
    float w = 0.5;
    float totalW = 0.0;
    for (int i = 0; i < 3; i++) {
        f += w * noise(p);
        totalW += w;
        p *= 2.07;
        w *= 0.5;
    }
    return f / totalW;
}

// Interleaved gradient noise for temporally stable ray jitter
// (no flickering unlike hash-based jitter)
float interleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

// ============================================================
// Slab intersection
// ============================================================

vec2 intersectSlab(vec3 ro, vec3 rd, float H) {
    // Handle rays nearly parallel to the slab
    if (abs(rd.y) < 1e-6) {
        if (abs(ro.y) < H) return vec2(0.0, 200.0);
        else return vec2(0.0, -1.0); // miss
    }
    float t1 = (-H - ro.y) / rd.y;
    float t2 = (H - ro.y) / rd.y;
    float tmin = min(t1, t2);
    float tmax = max(t1, t2);
    tmin = max(0.0, tmin);
    return vec2(tmin, tmax);
}

// ============================================================
// Galaxy parameters — must match galactic_disk.py exactly
// ============================================================

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
const float INNER_ARM_R = 3.5;
const float OUTER_RADIUS = 45.0;
const float ARM_COUNT = 4.0;
const float SPIRAL_TIGHTNESS = 2.5;
// Arm base offset matches galactic_disk.py exactly: (PI/2 - 0.488) - 2.5 * ln(7.0 / 3.5)
const float ARM_BASE_OFFSET = -0.650072;
const float BAR_ANGLE = 0.488;

// ============================================================
// Dust density function — the core of the upgrade
// ============================================================

float getDustDensity(vec3 p) {
    float r = length(p.xz);

    // Hard cutoffs with smooth ramps (preserves clean black-hole event horizon)
    if (r < 0.65 || r > 47.0) return 0.0;

    // ----------------------------------------------------------
    // 1. RADIAL DISTRIBUTION
    //    Prominent across central bar and spiral arms, smooth outer fade
    // ----------------------------------------------------------
    float inner_ramp = smoothstep(0.65, 2.2, r);
    float disk_profile = r * exp(-r / 11.0);
    disk_profile = disk_profile / (11.0 * exp(-1.0));
    float outer_fade = 1.0 - smoothstep(36.0, 46.0, r);

    float radial = inner_ramp * disk_profile * outer_fade;

    // ----------------------------------------------------------
    // 2. VERTICAL PROFILE
    //    Thin Gaussian, height varies smoothly with radius
    // ----------------------------------------------------------
    float scale_height = u_dustHeightScale * (0.22 + 0.18 * smoothstep(4.0, 28.0, r));
    float y_norm = p.y / scale_height;
    float vertical = exp(-0.5 * y_norm * y_norm);

    // ----------------------------------------------------------
    // 3. SPIRAL ARM & BAR ALIGNMENT
    //    Aligned with galactic_disk.py logarithmic arms + bar leading-edge lanes
    // ----------------------------------------------------------
    float r_arm = max(r, INNER_ARM_R);
    float theta = atan(p.z, p.x);

    // Find angular distance to closest spiral arm
    float min_dist = 100.0;
    for (int i = 0; i < int(ARM_COUNT); i++) {
        float arm_base = float(i) * (TWO_PI / ARM_COUNT);
        float ideal_theta = arm_base + ARM_BASE_OFFSET + SPIRAL_TIGHTNESS * log(r_arm / INNER_ARM_R);

        float d_theta = mod(theta - ideal_theta + PI, TWO_PI) - PI;
        float d_arm = abs(d_theta - u_dustArmOffset);
        min_dist = min(min_dist, d_arm);
    }

    float linear_dist = min_dist * r;

    // Narrow dense lane (Gaussian core)
    float lane_w = u_dustLaneWidth * r;
    float dense_lane = exp(-linear_dist * linear_dist / (lane_w * lane_w + 0.001));

    // Broader diffuse envelope
    float diff_w = u_dustDiffuseWidth * r;
    float diffuse_env = exp(-linear_dist * linear_dist / (diff_w * diff_w + 0.001)) * 0.25;

    float arm_factor = (dense_lane + diffuse_env) * smoothstep(2.5, 5.0, r);

    // Bar leading-edge dust lanes (characteristic dark lanes cutting across the central bar)
    if (r < 7.5) {
        float sin_b = sin(BAR_ANGLE);
        float cos_b = cos(BAR_ANGLE);
        float u = p.x * sin_b + p.z * cos_b;
        float v = -p.x * cos_b + p.z * sin_b;
        
        float target_v = sign(u) * 0.38;
        float dv = abs(v - target_v);
        float bar_lane = exp(-dv * dv / 0.035) * smoothstep(0.8, 1.8, abs(u)) * (1.0 - smoothstep(6.5, 7.5, abs(u)));
        arm_factor = max(arm_factor, bar_lane * 1.6);
    }

    // ----------------------------------------------------------
    // 4. MULTI-SCALE NOISE — domain-warped for filamentary structure
    // ----------------------------------------------------------
    float ns = u_dustNoiseScale;

    // Domain warp: offset sample position with noise to create filaments
    vec3 warp_offset = vec3(
        fbm3(p * ns * 0.12 + vec3(5.2, 1.3, 9.7)),
        fbm3(p * ns * 0.12 + vec3(3.7, 8.1, 2.4)),
        fbm3(p * ns * 0.12 + vec3(1.9, 4.6, 7.3))
    ) * 3.0;

    vec3 warped = p + warp_offset;

    // Large-scale: broad clouds and gaps across the galaxy
    float n_large = fbm3(warped * ns * 0.15);
    // Threshold to create distinct cloud/gap regions
    float cloud_mask = smoothstep(0.28, 0.55, n_large);

    // Medium-scale: broken lanes, clumps
    float n_medium = fbm5(warped * ns * 0.5 + vec3(12.3, 4.5, 6.7));
    // Create clumpy structure
    float clumps = smoothstep(0.2, 0.6, n_medium);

    // Fine-scale: wisps and irregular boundaries
    float n_fine = fbm3(p * ns * 1.5 + vec3(7.1, 2.8, 5.3));
    float wisps = 0.6 + 0.4 * n_fine;  // subtle edge variation

    float noise_combined = cloud_mask * clumps * wisps;

    // ----------------------------------------------------------
    // 5. COMBINE all factors
    // ----------------------------------------------------------
    float density = radial * vertical * arm_factor * noise_combined * u_dustDensity;

    return max(density, 0.0);
}


void main() {
    // 1. Reconstruct ray direction
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 target = u_invVP * vec4(ndc, 1.0, 1.0);
    vec3 rd = normalize(target.xyz / target.w - u_camPos);
    vec3 ro = u_camPos;

    // 2. Intersect with galactic dust slab
    // Height varies: thicker slab to encompass the full dust volume
    float slab_H = 2.0;
    vec2 t_bounds = intersectSlab(ro, rd, slab_H);

    float optical_depth = 0.0;
    float front_depth = -1.0;   // linear distance to first significant dust

    if (t_bounds.y > t_bounds.x && u_dustWeight > 0.001) {
        float t = t_bounds.x;
        float t_end = min(t_bounds.y, 150.0);

        // Adaptive step count: use more steps when the path through dust is short
        // (which means we're close and need detail) but stay within budget
        int steps = 32;
        float dt = (t_end - t) / float(steps);

        // Stable jitter using interleaved gradient noise
        vec2 pixel = gl_FragCoord.xy;
        float jitter = interleavedGradientNoise(pixel);
        t += dt * jitter;

        // Ray march with early termination
        float transmittance = 1.0;

        for (int i = 0; i < 32; i++) {
            if (t > t_end) break;
            if (transmittance < 0.01) break;  // early termination

            vec3 p = ro + rd * t;

            // Transform to galaxy local coordinates (undo galaxy rotation)
            vec4 p_local = u_galaxyModel * vec4(p, 1.0);

            float dens = getDustDensity(p_local.xyz);

            if (dens > 0.001) {
                // Record front depth: first significant dust distance
                if (front_depth < 0.0) {
                    front_depth = t;
                }
                optical_depth += dens * dt;
                transmittance = exp(-optical_depth * u_dustOpacity * u_dustWeight);
            }

            t += dt;
        }
    }

    // Transmittance: exp(-optical_depth * opacity * weight)
    float final_transmittance = exp(-optical_depth * u_dustOpacity * u_dustWeight);

    // Encode front depth as linear distance, -1 means no dust hit
    // Normalize to a reasonable range for the float16 texture
    float encoded_depth = (front_depth < 0.0) ? 0.0 : front_depth / 200.0;

    fragColor = vec4(final_transmittance, encoded_depth, 0.0, 1.0);
}
