#version 330

const float TWO_PI = 6.28318530718;

in float in_starIdx;
in vec2 in_currUV;
in vec2 in_prevUV;
in vec2 in_nextUV;
in float in_orbM;
in float in_side;

out vec4 v_color;
out vec3 v_worldPos;
out float v_relDepth;
out float v_edgeDist;
out float v_coreFrac;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec2 u_viewport;
uniform float u_bhViewDepth;
uniform float u_visibility;

uniform vec3 u_P[36];
uniform vec3 u_Q[36];
uniform vec4 u_orbitColor[36];   // rgb = linear tint, a = active alpha multiplier (0 if hidden)
uniform vec4 u_orbitParams[36];  // x = star_M, y = half_width_px, z = alpha_min, w = alpha_max

vec2 project_tangent(vec3 vp, vec3 dv, float fx, float fy) {
    float Z = max(-vp.z, 0.05);
    return vec2(
        fx * (Z * dv.x + vp.x * dv.z),
        fy * (Z * dv.y + vp.y * dv.z)
    );
}

void main() {
    int sidx = clamp(int(in_starIdx + 0.5), 0, 35);
    vec4 col_info = u_orbitColor[sidx];
    vec4 par = u_orbitParams[sidx];

    vec3 P = u_P[sidx];
    vec3 Q = u_Q[sidx];

    vec3 world_pos  = P * in_currUV.x + Q * in_currUV.y;
    vec3 world_prev = P * in_prevUV.x + Q * in_prevUV.y;
    vec3 world_next = P * in_nextUV.x + Q * in_nextUV.y;

    vec4 view_pos  = u_view * vec4(world_pos, 1.0);
    vec4 view_prev = u_view * vec4(world_prev, 1.0);
    vec4 view_next = u_view * vec4(world_next, 1.0);

    vec3 d1 = view_pos.xyz - view_prev.xyz;
    vec3 d2 = view_next.xyz - view_pos.xyz;
    float l1 = length(d1);
    float l2 = length(d2);
    vec3 u1 = (l1 > 1e-6) ? (d1 / l1) : vec3(1.0, 0.0, 0.0);
    vec3 u2 = (l2 > 1e-6) ? (d2 / l2) : u1;
    vec3 tan_3d = u1 + u2;
    float lt3 = length(tan_3d);
    tan_3d = (lt3 > 1e-5) ? (tan_3d / lt3) : u1;

    float fx = u_projection[0][0] * 0.5 * u_viewport.x;
    float fy = u_projection[1][1] * 0.5 * u_viewport.y;
    vec2 tan_px = project_tangent(view_pos.xyz, tan_3d, fx, fy);
    float len_px = length(tan_px);
    if (len_px < 1e-3) {
        tan_px = project_tangent(view_pos.xyz, u2, fx, fy);
        len_px = length(tan_px);
    }
    vec2 dir_px = (len_px > 1e-3) ? (tan_px / len_px) : vec2(1.0, 0.0);
    vec2 normal_px = vec2(-dir_px.y, dir_px.x);

    float half_w = par.y;
    vec4 clip_pos = u_projection * view_pos;
    float w_safe = max(clip_pos.w, 0.05);
    vec2 perp_ndc = (normal_px * 2.0) / max(u_viewport, vec2(1.0));
    clip_pos.xy += perp_ndc * (in_side * half_w * w_safe);

    gl_Position = clip_pos;
    v_worldPos = world_pos;
    v_relDepth = (-view_pos.z) - u_bhViewDepth;
    v_edgeDist = in_side;
    v_coreFrac = clamp((half_w - 0.90) / max(half_w, 0.1), 0.12, 0.72);

    // Trailing orbital phase wake behind the moving star
    float dM = mod(par.x - in_orbM + TWO_PI, TWO_PI);
    float wake = exp(-dM * 0.85);
    float base_a = mix(par.z, par.w, wake);
    float alpha = base_a * col_info.a * u_visibility;
    vec3 tint = col_info.rgb * (1.0 + 0.25 * wake);
    v_color = vec4(tint, alpha);
}
