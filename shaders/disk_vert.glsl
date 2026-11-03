#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_brightness;

out vec3 v_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_visibility;

uniform float u_simTime;
uniform float u_vFlat;
uniform float u_rCore;
uniform float u_patternSpeed;
uniform float u_armTightness;
uniform float u_armOffset;
uniform float u_innerArmR;
uniform float u_compression;
uniform float u_driftScale;
uniform float u_barAnglePolar;

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;

void main() {
    float r = length(in_position.xz);
    float y = in_position.y;
    
    // Smooth galactic rotation curve:
    // v(r) = v_flat * r / sqrt(r*r + r_core*r_core)
    // omega(r) = v(r) / r = v_flat / sqrt(r*r + r_core*r_core)
    // Safe central limit: omega(0) = v_flat / r_core
    float omega = u_vFlat / sqrt(r * r + u_rCore * u_rCore);
    
    // Initial polar angle in galactic plane
    float theta0 = atan(in_position.z, in_position.x);
    
    // Relative angular drift in the pattern frame (rotating at u_patternSpeed):
    // In OpenGL / GLM, a positive rotation angle around +Y rotates +X toward -Z (decreasing theta).
    // The relative angular velocity is (omega - u_patternSpeed).
    // Wrap to [-PI, PI] to avoid any floating-point precision loss over long runs.
    float delta_omega = omega - u_patternSpeed;
    float delta_theta_drift = mod(delta_omega * u_simTime + PI, TWO_PI) - PI;
    
    // Separate stellar orbital motion from arm pattern motion:
    // For differentially orbiting stars (u_driftScale = 1.0), stars drift relative to the pattern.
    // For arm-tracer stars (u_driftScale = 0.0), stars maintain their authentic star-forming alignment.
    float theta_rel = theta0 - delta_theta_drift * u_driftScale;
    
    // Lin-Shu density wave epicyclic modulation in pattern frame:
    // Ideal 4-arm spiral spine angle at radius r
    float r_clamped = max(r, u_innerArmR);
    float arm_spine = u_armOffset + u_armTightness * log(r_clamped / u_innerArmR);
    
    // Phase relative to the 4 spiral arms: psi = 4 * (theta_rel - arm_spine)
    float psi = 4.0 * (theta_rel - arm_spine);
    float psi_wrap = mod(psi + PI, TWO_PI) - PI;
    
    // Radial ramp for wave compression:
    // 0 inside the bar (r < 2.8), full across the disk, soft fade at outer edge (r > 42.0)
    float r_mask = smoothstep(2.8, 4.5, r) * (1.0 - smoothstep(42.0, 46.0, r));
    float compression_amp = u_compression * r_mask;
    
    // Azimuthal epicyclic displacement toward arm spine
    float delta_theta = -compression_amp * sin(psi_wrap);
    float theta_pat = theta_rel + delta_theta;
    
    // Position in pattern frame
    vec3 pos_pat = vec3(r * cos(theta_pat), y, r * sin(theta_pat));
    
    // Transform by u_model (which rotates pattern at u_patternSpeed * u_simTime)
    vec4 world_pos = u_model * vec4(pos_pat, 1.0);
    vec4 view_pos = u_view * world_pos;
    gl_Position = u_projection * view_pos;
    
    // Size attenuation based on distance
    float dist = max(0.1, -view_pos.z);
    float proj_size = in_size * (340.0 / dist);
    
    // Flux-conserving subpixel attenuation:
    // Scale unresolved star contribution with projected coverage to prevent
    // millions of background stars from accumulating excessive energy.
    float subpixel_factor = 1.0;
    if (proj_size < 1.0) {
        subpixel_factor = clamp(proj_size * proj_size, 0.0005, 1.0);
    }
    gl_PointSize = clamp(proj_size, 1.0, 24.0);
    
    // Wave proximity factor for astrophysical luminosity enhancement
    // (young OB stars and emission peaks along density wave shock)
    float arm_prox = 0.5 + 0.5 * cos(psi_wrap);
    float lum_boost = mix(0.88, 1.15, pow(arm_prox, 1.8) * r_mask);
    
    // Color modulation: subtle blue starburst enrichment in density wave shock
    vec3 star_color = in_color;
    vec3 shock_blue = vec3(0.85, 0.94, 1.05);
    star_color = mix(star_color, star_color * shock_blue, pow(arm_prox, 1.5) * 0.25 * r_mask);
    
    // Inter-bar resonance cavity in the bar pattern frame (r < 7.5):
    // Smoothly attenuates diffuse stars along the bar minor axis in the rotating pattern frame,
    // keeping the 3D bar prominent without frozen spatial rejections that shear into spiral grooves.
    float cavity_att = 1.0;
    if (u_driftScale > 0.5 && r < 7.5) {
        float minor_axis_prox = abs(sin(theta_pat - u_barAnglePolar));
        cavity_att = 1.0 - (1.0 - smoothstep(3.0, 7.5, r)) * (0.85 * minor_axis_prox);
    }
    
    v_color = star_color * in_brightness * u_visibility * subpixel_factor * lum_boost * cavity_att;
}
