#ifndef KDTREEH
#define KDTREEH
#include <vector>
#include <thrust/device_vector.h>
#include <random>
#include <algorithm>
#include <deque>

#include "../tracing/vec3.h"
#include "../tracing/triangle.h"
#include "../tracing/ray.h"

#define MAX_LEVEL 20
#define MIN_OBJECTS 5
#define LEAF_SIZE 8 // TODO: make sure to change in cpp file too
#define BUF_SIZE 2048

using namespace std;

std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator

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

    __device__ bool hit(const ray& r, ray_hit& finalHitRec);

    int tri_count;
    int node_count; 
    Triangle *allTriangles;
    TreeNodeGPU *nodes;
};

class KdTree {

public: 
    __host__ KdTree() {}
    __host__ void init(Triangle *triangles, int n);
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

__host__ void KdTree::init(Triangle *triangles, int n) {
    // median of first dimension, entire list
    // two leaves, 
    this->numNodes = 0;

    this->allTriangles = triangles;

    std::vector<int> ts(n);
    std::iota(ts.begin(), ts.end(), 0);

    // std::vector<Triangle> ts(triangles, triangles + n);
    // std::cerr << "starting tree init" << std::endl;
    this->root = initHelper(ts, X, 0, 1);
    // std::cerr << "finished tree init" << std::endl;
    this->renumber();
    // std::cerr << "renumbered tree" << std::endl;
    this->createNodeArray();
    // this->printTree();
    return;
}

__host__ TreeNode* KdTree::initHelper(std::vector<int> ts, Axis a, int l, int nextId) {
    // if (l == 0) {
    //     std::cout << "num triangles: " << ts.size() << std::endl;
    // }
    this->numNodes++;

    bbox newBbox = boundFromList(&ts);

    // if (l == 0) {
    //     std::cerr << "bounding box on init min: " << newBbox.min << std::endl;
    //     std::cerr << "bounding box on init max: " << newBbox.max << std::endl;
    // }

    if (ts.size() <= LEAF_SIZE) {
        TreeNode* leaf = new TreeNode(l, a, INFINITY, true, ts, NULL, 
                                    NULL, newBbox, nextId);
        return leaf;
    }
    float s;
    s = quickSelect(ts, a);
    std::vector<int> leftVector;
    std::vector<int> rightVector;
    Triangle t;
    for (int ti : ts){
        t = this->allTriangles[ti];
        float center = (t.v[0][a] + t.v[1][a] + t.v[2][a]) / 3.0;
        if (center <= s) {
            leftVector.push_back(ti);
        } else {
            rightVector.push_back(ti);
        }
    }

    TreeNode* node = new TreeNode(l, a, s, false, std::vector<int>(),
                                NULL, NULL, newBbox, nextId);
                                
    
    int axisNumRep = static_cast<int>(a);
    axisNumRep++;
    axisNumRep%=3;
    a = static_cast<Axis>(axisNumRep);
    l++;
    TreeNode *leftLeaf = initHelper(leftVector, a, l, nextId*2);
    TreeNode *rightLeaf = initHelper(rightVector, a, l, nextId*2 + 1);
    node->left = leftLeaf;
    node->right = rightLeaf;

    return node;
}

__host__ void KdTree::renumber() {
    std::deque<TreeNode*> toVisit = {this->root};
    int counter = 0;
    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();
        toVisit.pop_front();

        curr->id = counter;
        counter++;

        if (curr->isLeaf) {
            continue;
        }

        if (curr->left != NULL) {
            toVisit.push_back(curr->left);
        }

        if (curr->right != NULL) {
            toVisit.push_back(curr->right);
        }
    }
}

__host__ void KdTree::createNodeArray() {
    // this->nodeArray = new TreeNodeGPU[this->numNodes];
    std::deque<TreeNode*> toVisit = {this->root};
    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();
        toVisit.pop_front();

        // TreeNodeGPU *newNode = new TreeNodeGPU(false, 0, NULL, curr->box, curr->id, -1, -1);

        if (curr->isLeaf) {
            int t_count = curr->tri_idxs.size();
            // for (int i = 0; i < t_count; i++) {
            //     newNode->t_idxs[i] = curr->tri_idxs[i];
            // }

            // newNode->isLeaf = true;
            // newNode->numTris = t_count;

            // this->nodeArray[curr->id] = *newNode;
            this->nodeArray.push_back(TreeNodeGPU(true, t_count, curr->tri_idxs.data(), curr->box, curr->id, -1, -1));
            continue;
        }

        int leftNum = -1;
        int rightNum = -1;

        if (curr->left != NULL) {
            // newNode->leftNodeIdx = curr->left->id; 
            leftNum = curr->left->id;
            toVisit.push_back(curr->left);
        }

        if (curr->right != NULL) {
            // newNode->rightNodeIdx = curr->right->id; 
            rightNum = curr->right->id;
            toVisit.push_back(curr->right);
        }

        // this->nodeArray[curr->id] = *newNode;
        // this->nodeArray.push_back(*newNode);
        this->nodeArray.push_back(TreeNodeGPU(false, 0, NULL, curr->box, curr->id, leftNum, rightNum));
    }

    std::cerr << "creating tree size: " << this->nodeArray.size() << std::endl;

    // this->nodeArray = allNodes;
}

__host__ bool KdTree::hit(const ray& r, ray_hit& finalHitRec) {
    // check if ray hits bounding box of curNode
    // check if ray hits bounding box of left or right child
    // traverse again with either the left or right child
    /*
    bounding box max: 18.7272 -1.29747 1.01005e-08
    ounding box min: 16.1273 -1.26498 1.01005e-08
    */

    std::deque<TreeNode*> toVisit = {this->root};

    bool has_hit = false;
    float t_max = INFINITY;
    ray_hit rec;

    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();

        // if (curr->level == 0) {
        //     std::cerr << "bounding box min: " << curr->box.min << std::endl;
        //     std::cerr << "bounding box max: " << curr->box.max << std::endl;
        // }


        toVisit.pop_front();
        if (!curr->hit(r) && curr->level > 0) {
            // std::cerr << "how did we get here, level:" << curr->level << std::endl;
        }
        if (curr->isLeaf) {
            // LEAF NODE
            int hitCount = 0;
            Triangle t;
            for (int ti : curr->tri_idxs) {
                t = this->allTriangles[ti];
                if (t.hit(r, t_max, rec)) {
                    hitCount++;
                    has_hit = true;
                    t_max = rec.t;
                    finalHitRec = rec;
                }
                // std::cerr << "hit " << hitCount << " triangle(s) in leaf node, level:" << curr->level << std::endl;
            }

            continue;
        }

        bool hitLeft = curr->left->hit(r);
        if (hitLeft) {
            // std::cerr << "hit left tree bounding box, level:" << curr->left->level << std::endl;
            toVisit.push_back(curr->left);
        }

        bool hitRight = curr->right->hit(r);

        if (hitRight) {
            // std::cerr << "hit right tree bounding box, level:" << curr->right->level << std::endl;
            toVisit.push_back(curr->right);
        }

        if (!hitLeft && !hitRight && curr->level > 0) {
            // std::cerr << "how did we hit neither box, but we hit the box above" << std::endl;
        }
    }

    return has_hit;
}

__host__ bbox KdTree::boundFromList(std::vector<int> *items) {
    float min_x = INFINITY;
    float min_y = INFINITY;
    float min_z = INFINITY;
    
    float max_x = -INFINITY;
    float max_y = -INFINITY;
    float max_z = -INFINITY;

    int count = items->size();

    Triangle t;

    for (int i=0; i < count; i++) {
        t = this->allTriangles[(*items)[i]];
        for (int j = 0; j < 3; j++) {
            max_x = max(max_x, t.v[j][0]);
            max_y = max(max_y, t.v[j][1]);
            max_z = max(max_z, t.v[j][2]);

            min_x = min(min_x, t.v[j][0]);
            min_y = min(min_y, t.v[j][1]);
            min_z = min(min_z, t.v[j][2]);
        }
    }

    vec3 maxVec(max_x, max_y, max_z);
    vec3 minVec(min_x, min_y, min_z);
    return bbox{maxVec, minVec};
}

__host__ void KdTree::printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft)
{
    if( node != nullptr )
    {
        std::cerr << prefix;

        std::cerr << (isLeft ? "├──" : "└──" );

        // print the value of the node
        std::cerr << node->id << std::endl;
        // std::cerr << node->level << std::endl;

        // enter the next tree level - left and right branch
        printTreeHelper( prefix + (isLeft ? "│   " : "    "), node->left, true);
        printTreeHelper( prefix + (isLeft ? "│   " : "    "), node->right, false);
    }
}

__host__ void KdTree::printGPUTreeHelper(const std::string& prefix, const TreeNodeGPU* node, bool isLeft)
{   
    if (node->idx > 100) {
        return;
    }

    std::cerr << prefix;

    std::cerr << (isLeft ? "├──" : "└──" );

    // print the value of the node
    std::cerr << node->idx << std::endl;
    // std::cerr << node->level << std::endl;

    // enter the next tree level - left and right branch
    if (node->leftNodeIdx != -1) {
        printGPUTreeHelper( prefix + (isLeft ? "│   " : "    "), &(this->nodeArray[node->leftNodeIdx]), true);
    }

    if (node->rightNodeIdx != -1) {
        printGPUTreeHelper( prefix + (isLeft ? "│   " : "    "), &(this->nodeArray[node->rightNodeIdx]), false);
    }
}

__host__ void KdTree::printTree()
{
    // this->printTreeHelper("", this->root, false);
    // std::cerr << "num nodes: " << this->numNodes << std::endl;
    // std::cerr << "node array size: " << this->nodeArray.size() << std::endl;

    for (int i=0; i<this->numNodes; i++) {
        std::cerr << this->nodeArray[i].idx << std::endl;
    }
    this->printGPUTreeHelper("", &(this->nodeArray[0]), false);
}

__device__ bool KdTreeGPU::hit(const ray& r, ray_hit& finalHitRec) {
    // thrust::device_vector<int> toVisit;
    // std::vector<int> toVisit;
    // toVisit.push_back(0);

    int toVisit[BUF_SIZE];

    int visitIdx = 0;
    int pushIdx = 1;
    // std::deque<TreeNode*> toVisit = {this->root};

    bool has_hit = false;
    float t_max = INFINITY;
    ray_hit rec;

    while (visitIdx < pushIdx) {
        TreeNodeGPU *curr = &(this->nodes[toVisit[visitIdx]]);
        visitIdx++;
        visitIdx %= BUF_SIZE;
        if (curr->isLeaf) {
            // LEAF NODE
            int hitCount = 0;
            Triangle t;
            for (int ti : curr->t_idxs) {
                t = this->allTriangles[ti];
                if (t.hit(r, t_max, rec)) {
                    hitCount++;
                    has_hit = true;
                    t_max = rec.t;
                    // finalHitRec = rec;
                    finalHitRec.t = t_max;
                    finalHitRec.p = rec.p;
                    finalHitRec.normal = rec.normal;
                }
                // std::cerr << "hit " << hitCount << " triangle(s) in leaf node, level:" << curr->level << std::endl;
            }

            continue;
        }

        bool hitLeft = this->nodes[curr->leftNodeIdx].hit(r);
        if (hitLeft) {
            toVisit[pushIdx] = curr->leftNodeIdx;
            pushIdx++;
            pushIdx %= BUF_SIZE;
        }

        bool hitRight = this->nodes[curr->rightNodeIdx].hit(r);

        if (hitRight) {
            toVisit[pushIdx] = curr->rightNodeIdx;
            pushIdx++;
            pushIdx %= BUF_SIZE;
        }
    }

    return has_hit;
}

// From stack overflow: https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
__host__ bool TreeNode::hit(const ray& r) {

    // r.dir is unit direction vector of ray
    // float dirfrac_x = 1.0f / r.direction()[0];
    // float dirfrac_y = 1.0f / r.direction()[1];
    // float dirfrac_z = 1.0f / r.direction()[2];
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (this->box.min[0] - r.origin()[0]) / r.direction()[0];
    float t2 = (this->box.max[0] - r.origin()[0]) / r.direction()[0];
    float t3 = (this->box.min[1] - r.origin()[1]) / r.direction()[1];
    float t4 = (this->box.max[1] - r.origin()[1]) / r.direction()[1];
    float t5 = (this->box.min[2] - r.origin()[2]) / r.direction()[2];
    float t6 = (this->box.max[2] - r.origin()[2]) / r.direction()[2];

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        // t = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        // t = tmax;
        return false;
    }

    // t = tmin;
    return true;
}

__host__ __device__ bool TreeNodeGPU::hit(const ray& r) {

    // r.dir is unit direction vector of ray
    // float dirfrac_x = 1.0f / r.direction()[0];
    // float dirfrac_y = 1.0f / r.direction()[1];
    // float dirfrac_z = 1.0f / r.direction()[2];
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (this->box.min[0] - r.origin()[0]) / r.direction()[0];
    float t2 = (this->box.max[0] - r.origin()[0]) / r.direction()[0];
    float t3 = (this->box.min[1] - r.origin()[1]) / r.direction()[1];
    float t4 = (this->box.max[1] - r.origin()[1]) / r.direction()[1];
    float t5 = (this->box.min[2] - r.origin()[2]) / r.direction()[2];
    float t6 = (this->box.max[2] - r.origin()[2]) / r.direction()[2];

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        // t = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        // t = tmax;
        return false;
    }

    // t = tmin;
    return true;
}

__host__ float KdTree::quickSelectHelper(std::vector<float> &data, int k) {
    if (data.size() == 1) {
        return data[0];
    }
    // choosing a random pivot
    std::uniform_int_distribution<> dist (0, data.size()-1);
    int idx = dist(gen);
    float pivot = data[idx];

    std::vector<float> less;
    std::vector<float> equal;
    std::vector<float> greater;
    // partitioning the list
    for (float val : data) {
        if (val < pivot) {
            less.push_back(val);
        } else if (val > pivot) {
            greater.push_back(val);
        } else {
            equal.push_back(val);
        }
    }
    // recursive calls to quickselect
    if (k <= int(less.size())) {
        return quickSelectHelper(less, k);
    } else if (k <= int(less.size() + equal.size())) {
        return pivot;
    } else {
        return quickSelectHelper(greater, k - less.size() - equal.size());
    }
}

__host__ float KdTree::quickSelect(std::vector<int> ts, Axis a) {
    std::vector<float> data;
    int axis = static_cast<int>(a);
    int count = ts.size();

    Triangle t;
    for (int i=0; i<count; i++) {
        t = this->allTriangles[ts[i]];
        data.push_back(t.v[0][axis]);
        data.push_back(t.v[1][axis]);
        data.push_back(t.v[2][axis]);
    }
    std::sort(data.begin(), data.end());
    int dSize = data.size();
    if (dSize % 2 == 0) {
        return (data[dSize / 2 - 1] + data[dSize / 2]) / 2.0;
    } else {
        return data[dSize / 2];
    }
    // fix later
    // return quickSelectHelper(data, count/2);
}

#endif

