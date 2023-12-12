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

    vec3 computePhong(vec3 position, vec3 dirToCam, Triangle t);
    
    LightType type;

    vec3 color;
    vec3 attFunc;  // Attenuation function

    vec3 pos;       // Not applicable to directional lights
    vec3 dir;       // Not applicable to point lights

    float penumbra;      // Only applicable to spot lights, in RADIANS
    float angle;         // Only applicable to spot lights, in RADIANS
};

float falloff(float x, float inner, float outer) {
    float xdiff = x - inner;
    float adiff = outer-inner;
    float c1 = -2 * pow((xdiff)/adiff, 3);
    float c2 = 3 * pow((xdiff)/adiff,2);
    return c1 + c2;
}

vec3 Light::computePhong(vec3 position, vec3 dirToCam, Triangle t, const StlObject& obj) {
    vec3 illumination(0.f,0.f,0.f);

    vec3 normal = t.n;
    dirToCam.make_unit_vector();

    vec3 surfaceToLight;
    float fAtt = 1;

    switch(this->type) {
        case (LightType::LIGHT_DIRECTIONAL): {
            surfaceToLight = -vec3(this->dir);
            break;
        }
        case (LightType::LIGHT_POINT):
        case (LightType::LIGHT_SPOT): {
            surfaceToLight = vec3(this->pos) - position;
            float d = surfaceToLight.length();
            fAtt = fmin(1, 1.0/(this->attFunc[0] + this->attFunc[1] + (d * d * this->attFunc[2])));
            break;
        }
    }

    // SHADOWS:
    vec3 surfaceOffset = position + 0.01f * surfaceToLight;
    ray rayToLight = ray{surfaceOffset, surfaceToLight};
    ray_hit rec;
    if (obj.hitTreeGPU(rayToLight, rec)) {
        if (this->type == LightType::LIGHT_DIRECTIONAL) {
            return vec3(0,0,0);
        }
        if ((rec.p - position).length() < surfaceToLight.length()) {
            return vec3(0,0,0);
        }
    }

    float intensity = 1;
    if (light.type == LightType::LIGHT_SPOT) {
        float dotProd = dot(-surfaceToLight, this->dir);
        float normProd = surfaceToLight.length() * this->dir.length();
        float lightToIntAngle = acos(dotProd/normProd);

        float outer = this->angle;
        float inner = outer - (this->penumbra);

        if (lightToIntAngle > outer) {
            return vec3(0,0,0);
        }

        if ((lightToIntAngle <= outer) && (lightToIntAngle > inner)) {
            intensity *= 1 - falloff(lightToIntAngle, inner, outer);
        }
    }

    vec3 dirToSource = unit_vector(surfaceToLight);
    float dotProd = dot(normal, dirToSource);
    vec3 diffuse = dotProd > 0 ? obj.color * dotProd : vec3(0,0,0);

    vec3 incomDir = unit_vector(-surfaceToLight);
    vec3 refDir = incomDir - 2.f * dot(incomDir, normal) * normal;
    float specProd = dot(refDir, dirToCam);
    vec3 specular = specProd > 0 ? obj.specular * float(pow(specProd, obj.shininess)) : vec3(0,0,0);

    return light.color * fAtt * (diffuse + sepcular) * intensity;
}


#endif