#ifndef LIGHTH
#define LIGHTH

#include "vec3.h"
#include "ray.h"

enum class LightType {
    LIGHT_POINT,
    LIGHT_DIRECTIONAL,
    LIGHT_SPOT,
};

class Light {

public:
    Light() {}
    void makePoint(vec3 c, vec3 f_att, vec3 p) {
        type = LightType::LIGHT_POINT;
        color = c;
        attFunc = f_att;
        pos = p;
    }

    void makeDir(vec3 c, vec3 f_att, vec3 d) {
        type = LightType::LIGHT_DIRECTIONAL;
        color = c;
        attFunc = f_att;
        dir = d;
    }

    void makeSpot(vec3 c, vec3 f_att, vec3 position, vec3 p, vec3 a) {
        type = LightType::LIGHT_SPOT;
        color = c;
        attFunc = f_att;
        pos = position;
        penumbra = p;
        angle = a;
    }
    
    LightType type;

    vec3 color;
    vec3 attFunc;  // Attenuation function

    vec3 pos;       // Not applicable to directional lights
    vec3 dir;       // Not applicable to point lights

    float penumbra;      // Only applicable to spot lights, in RADIANS
    float angle;         // Only applicable to spot lights, in RADIANS
};


#endif