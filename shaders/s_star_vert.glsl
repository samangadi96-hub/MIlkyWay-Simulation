#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_brightness;
in float in_highlight;

out vec3 v_color;
out float v_relDepth;
out float v_projSize;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_bhViewDepth;
uniform float u_visibility;
uniform float u_fovHeightPx;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_projection * view_pos;

    v_relDepth = (-view_pos.z) - u_bhViewDepth;

    // Distance-scaled point size (physical orbital positions are never modified)
    float dist = max(0.05, -view_pos.z);
    float proj_px = in_size * (u_fovHeightPx * 0.025) / pow(dist, 0.62);
    float pt_size = clamp(proj_px, 2.6, 13.5) + 2.2 * in_highlight;
    gl_PointSize = pt_size;
    v_projSize = pt_size;

    float hi_boost = 1.0 + 0.35 * in_highlight;
    v_color = in_color * in_brightness * hi_boost * u_visibility;
}
