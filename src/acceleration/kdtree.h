#pragma once
#include <vector>
#include "tracing/vec3.h"
#include "tracing/stlobject.h"
#include "tracing/triangle.h"
#include "tracing/ray.h"

#define MAX_LEVEL 20
#define MIN_OBJECTS 5

enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

enum EdgeType {
    MIN = 0,
    MAX = 1,
}

struct bbox {
    vec3 min;
    vec3 max;
    float surfaceArea;
    vec3 dims;
}

struct boxPrim {
    bbox bounds;
    Triangle t;
}

class TreeNode {

public:
    __host__ __device__ TreeNode() {}
    __host__ TreeNode(int l, Axis a, float s, bool leaf, 
               std::vector<Triangle> ts, kdtree *ltree, 
               kdtree *rtree, bbox bbox) {
        level = l;
        axis = a;
        split = s;
        isLeaf = leaf;
        triangles = ts;
        left = ltree;
        right = rtree;
        bbox = bbox;
    }

    __device__ bool hit(const ray& r);
    __device__ bool hit_bbox(const ray& r);
    __device__ float ray_inter_plane(const ray& r, float a, float b, float c, float d);

    bool isLeaf;
    float split;
    Axis axis;
    int level;

    std::vector<Triangle> triangles;

    TreeNode *left;
    TreeNode *right;
    bbox box;
}

class KdTree {

public: 
    __host__ KdTree() {}
    __host__ TreeNode* init(StlObject obj);
    __device__ bool hit(const ray& r, ray_hit& hitRec);

    TreeNode *root;

private:
    const int MAXDEPTH = 10;
    const int MINOBJS = 2;

    __host__ TreeNode* initHelper(std::vector<Triangle> ts, Axis a);
    
    __host__ bbox bound(Triangle *t);
    __host__ bbox boundFromList(std::vector<Triangle> *items);
}

