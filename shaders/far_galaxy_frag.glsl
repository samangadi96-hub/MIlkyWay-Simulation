#version 330

in vec3 v_objPos;
in vec3 v_worldPos;

uniform sampler3D u_stellarMap;
uniform sampler3D u_dustNebulaMap;
uniform float u_visibility;
uniform float u_layerWeight;
uniform vec3 u_camPos;
uniform mat4 u_invModel;
uniform vec3 u_boxMin;
uniform vec3 u_boxMax;

out vec4 fragColor;

// Robust ray-box intersection (slab method)
bool intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax, out float tNear, out float tFar) {
    vec3 invR = 1.0 / rd;
    vec3 tbot = invR * (boxMin - ro);
    vec3 ttop = invR * (boxMax - ro);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    tNear = max(max(tmin.x, tmin.y), tmin.z);
    tFar = min(min(tmax.x, tmax.y), tmax.z);
    return tNear <= tFar && tFar > 0.0;
}

void main() {
    // Transform ray to object space where the bounding box is axis-aligned
    vec3 ro = (u_invModel * vec4(u_camPos, 1.0)).xyz;
    vec3 rd = normalize((u_invModel * vec4(normalize(v_worldPos - u_camPos), 0.0)).xyz);

    float tNear, tFar;
    if (!intersectBox(ro, rd, u_boxMin, u_boxMax, tNear, tFar)) {
        discard;
    }

    tNear = max(tNear, 0.0);
    float rayLength = tFar - tNear;
    if (rayLength <= 0.001) {
        discard;
    }

    const int NUM_STEPS = 28;
    float stepSize = rayLength / float(NUM_STEPS);
    vec3 stepVec = rd * stepSize;
    vec3 p = ro + rd * (tNear + stepSize * 0.5);

    vec3 boxExt = u_boxMax - u_boxMin;
    vec3 accumColor = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    float accumAlpha = 0.0;

    float normStep = stepSize * 0.95;

    for (int i = 0; i < NUM_STEPS; ++i) {
        vec3 uvw = vec3(
            (p.x - u_boxMin.x) / boxExt.x,
            (p.z - u_boxMin.z) / boxExt.z,
            (p.y - u_boxMin.y) / boxExt.y
        );

        if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
            vec4 stars = texture(u_stellarMap, uvw);
            vec4 dustNeb = texture(u_dustNebulaMap, uvw);

            float tau_local = dustNeb.r * normStep;
            // Cardelli, Clayton & Mathis (CCM 1989) wavelength-dependent interstellar extinction
            vec3 ccm_step = vec3(
                exp(-tau_local * 0.75),
                exp(-tau_local * 1.00),
                exp(-tau_local * 1.35)
            );

            vec3 emission = (stars.rgb + dustNeb.gba * 1.10) * normStep;

            accumColor += emission * transmittance;
            transmittance *= ccm_step;
            accumAlpha += stars.a * normStep;

            if (dot(transmittance, vec3(0.333)) < 0.01) {
                break; // Early ray termination
            }
        }

        p += stepVec;
    }

    float weight = u_visibility * (u_layerWeight > 0.0 ? u_layerWeight : 1.0);
    if (accumAlpha * weight <= 0.0002) {
        discard;
    }

    fragColor = vec4(accumColor * weight, accumAlpha * weight);
}
