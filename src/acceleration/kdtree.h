#pragma once
#include <vector>
#include "../tracing/vec3.h"
#include "../tracing/triangle.h"
#include "../tracing/ray.h"

#define MAX_LEVEL 20
#define MIN_OBJECTS 5

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
    float surfaceArea;
    vec3 dims;
};

struct boxPrim {
    bbox bounds;
    Triangle t;
};

class TreeNode {

public:
    TreeNode() {}
    TreeNode(int l, Axis a, float s, bool leaf, 
               std::vector<Triangle> ts, TreeNode *ltree, 
               TreeNode *rtree, bbox bbox, int id) {
        level = l;
        axis = a;
        split = s;
        isLeaf = leaf;
        triangles = ts;
        left = ltree;
        right = rtree;
        bbox = bbox;
        id = id;
    }

    bool hit(const ray& r);

    bool isLeaf;
    float split;
    Axis axis;
    int level;
    int id;

    std::vector<Triangle> triangles;

    TreeNode *left;
    TreeNode *right;
    bbox box;
};

class KdTree {

public: 
    KdTree() {}
    void init(Triangle *triangles, int n);
    bool hit(const ray& r, ray_hit finalHitRec);

    void printTree();
    TreeNode *root;

private:
    const int MAXDEPTH = 10;
    const int MINOBJS = 2;

    TreeNode* initHelper(std::vector<Triangle> ts, Axis a, int l, int prevId);
    
    bbox bound(Triangle *t);
    bbox boundFromList(std::vector<Triangle> *items);
    float quickSelectHelper(std::vector<float> &data, int k);
    float quickSelect(std::vector<Triangle> ts, Axis a);
    void printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft);
};

