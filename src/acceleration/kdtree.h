#ifndef KDTREEH
#define KDTREEH
#include <vector>
#include "../tracing/vec3.h"
#include "../tracing/triangle.h"
#include "../tracing/ray.h"

#define MAX_LEVEL 20
#define MIN_OBJECTS 5
#define LEAF_SIZE 8 // TODO: make sure to change in cpp file too


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
               std::vector<int> ts, TreeNode *ltree, 
               TreeNode *rtree, bbox newBox, int nid) {
        level = l;
        axis = a;
        split = s;
        isLeaf = leaf;
        tri_idxs = ts;
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

    std::vector<int> tri_idxs;

    TreeNode *left;
    TreeNode *right;
    bbox box;
};

class TreeNodeGPU {

public:
    __host__ __device__ TreeNodeGPU() {}
    __host__ __device__ TreeNodeGPU(bool leafBool, int trisArraySize, int* trisArray, 
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
    
    __host__ __device__ bool hit(const ray& r);

    bool isLeaf;
    bbox box;
    int numTris;
    int idx;
    int leftNodeIdx;
    int rightNodeIdx;
    int t_idxs[LEAF_SIZE];
    
};

class KdTreeGPU {

public: 
    __host__ __device__ KdTreeGPU() {}
    __host__ __device__ KdTreeGPU(Triangle *ts, int nts, TreeNodeGPU *ns, int nns) {
        tri_count = nts;
        node_count = nns;

        allTriangles = ts;
        nodes = ns;
    }

    __host__ __device__ bool hit(const ray& r, ray_hit& finalHitRec);

    int tri_count;
    int node_count; 
    Triangle *allTriangles;
    TreeNodeGPU *nodes;
};

class KdTree {

public: 
    __host__ __device__ KdTree() {}
    __host__ void init(Triangle *triangles, int n);
    __host__ __device__ void hitGPU(const ray& r, ray_hit& finalHitRec);
    __host__ bool hit(const ray& r, ray_hit& finalHitRec);
    __host__ void printTree();

    int numNodes;
    TreeNode *root;
    // TreeNodeGPU *nodeArray;
    std::vector<TreeNodeGPU> nodeArray;
    Triangle *allTriangles;

    // TreeNodeGPU *rootGPU;

private:
    // const int MAXDEPTH = 10;
    // const int MINOBJS = 2;

    __host__ TreeNode* initHelper(std::vector<int> ts, Axis a, int l, int prevId);
    __host__ void renumber();
    __host__ void createNodeArray();
    __host__ bbox boundFromList(std::vector<int> *items);
    __host__ float quickSelectHelper(std::vector<float> &data, int k);
    __host__ float quickSelect(std::vector<int> ts, Axis a);
    __host__ void printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft);
    __host__ void printGPUTreeHelper(const std::string& prefix, const TreeNodeGPU* node, bool isLeft);
};

#endif

