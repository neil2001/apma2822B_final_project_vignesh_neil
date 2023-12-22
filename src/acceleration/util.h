#ifndef UTILH
#define UTILH

#include "../tracing/vec3.h"

#define LEAF_SIZE 64
#define BUF_SIZE 2048

enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
};

enum EdgeType {
    MIN = 0,
    MAX = 1,
};

struct bbox {
    vec3 min; // these aren't rly vectors tbh
    vec3 max;
};

#endif