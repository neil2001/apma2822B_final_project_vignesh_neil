#pragma once
#include <vector>
#include "../tracing/vec3.h"
#include "../tracing/triangle.h"
#include "../tracing/ray.h"

#define MAX_LEVEL 20
#define MIN_OBJECTS 5
#define LEAF_SIZE 10 // TODO: make sure to change in cpp file too


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

class TreeNode {

public:
    TreeNode() {}
    TreeNode(int l, Axis a, float s, bool leaf, 
               std::vector<Triangle> ts, TreeNode *ltree, 
               TreeNode *rtree, bbox newBox, int nid) {
        level = l;
        axis = a;
        split = s;
        isLeaf = leaf;
        triangles = ts;
        left = ltree;
        right = rtree;
        box = newBox;
        id = nid;
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

class TreeNodeGPU {

public:
    TreeNodeGPU() {}
    TreeNodeGPU(bool leafBool, int trisArraySize, int* trisArray, 
                bbox newBox, int curIdx, int leftIdx, int rightIdx) {
        isLeaf = leafBool;
        box = newBox;
        numTris = trisArraySize;
        for (int i = 0; i < numTris; i++) {
            t_idxs[i] = trisArray[i];
        }
        idx = curIdx;
        leftNodeIdx = leftIdx;
        rightNodeIdx = rightIdx;
    }
    bool isLeaf;
    bbox box;
    int numTris;
    int idx;
    int leftNodeIdx;
    int rightNodeIdx;
    int t_idxs[LEAF_SIZE];
};

class KdTree {

public: 
    KdTree() {}
    void init(Triangle *triangles, int n);
    bool hit(const ray& r, ray_hit& finalHitRec);
    int numNodes;


    void printTree();
    TreeNode *root;

    TreeNode *nodeArray;

private:
    const int MAXDEPTH = 10;
    const int MINOBJS = 2;

    TreeNode* initHelper(std::vector<Triangle> ts, Axis a, int l, int prevId);
    void KdTree::createNodeArray();
    bbox bound(Triangle *t);
    bbox boundFromList(std::vector<Triangle> *items);
    float quickSelectHelper(std::vector<float> &data, int k);
    float quickSelect(std::vector<Triangle> ts, Axis a);
    void printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft);
};

