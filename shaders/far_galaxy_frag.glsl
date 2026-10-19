#version 330

in vec2 v_uv;
in vec3 v_worldPos;

uniform sampler2D u_stellarMap;
uniform sampler2D u_dustNebulaMap;
uniform float u_visibility;
uniform float u_layerWeight;

out vec4 fragColor;

void main() {
    vec4 stars = texture(u_stellarMap, v_uv);
    vec4 dustNeb = texture(u_dustNebulaMap, v_uv);

    // Channel convention:
    // dustNeb.r   = Dust optical depth
    // dustNeb.gba = Localized Nebula RGB emission (G=Red, B=Green, A=Blue)
    float dustDepth = dustNeb.r;
    vec3 nebEmission = dustNeb.gba;

    // Dark dust extinction: lanes absorb background and diffuse stellar light
    float dustExtinction = exp(-dustDepth * 2.5);

    // Combine diffuse stellar light attenuated by dust lanes, plus localized nebula emission
    vec3 color = (stars.rgb * dustExtinction) + (nebEmission * 1.10);
    float alpha = stars.a * u_visibility * u_layerWeight;

    if (alpha <= 0.0005) {
        discard;
    }

    // Additive pre-weighted HDR color output (paired with GL_ONE, GL_ONE blending)
    fragColor = vec4(color * u_visibility * u_layerWeight, alpha);
}
