#include <vector>
#include <deque>

#include "kdtree.h"
#define LEAF_SIZE 10 //TODO: make sure to change in header file too


using namespace std;

void KdTree::init(Triangle *triangles, int n) {
    // median of first dimension, entire list
    // two leaves, 

    std::vector<Triangle> ts(triangles, triangles + n);
    std::cerr << "starting tree init" << std::endl;
    this->root = initHelper(ts, X, 0, 1);
    std::cerr << "finished tree init" << std::endl;
    this->printTree();
    this->numNodes = 0;
    return;
}

TreeNode* KdTree::initHelper(std::vector<Triangle> ts, Axis a, int l, int nextId) {
    // if (l == 0) {
    //     std::cout << "num triangles: " << ts.size() << std::endl;
    // }

    bbox newBbox = boundFromList(&ts);

    if (l == 0) {
        std::cerr << "bounding box on init min: " << newBbox.min << std::endl;
        std::cerr << "bounding box on init max: " << newBbox.max << std::endl;
    }

    if (ts.size() <= LEAF_SIZE) {
        this->numNodes++;
        TreeNode* leaf = new TreeNode(l, a, INFINITY, true, ts, NULL, 
                                    NULL, newBbox, nextId);
        return leaf;
    }
    float s;
    s = quickSelect(ts, a);
    std::vector<Triangle> leftVector;
    std::vector<Triangle> rightVector;
    for (Triangle t : ts){

        // TODO: use the center of the triangle instead
        float center = (t.v[0][a] + t.v[1][a] + t.v[2][a]) / 3.0;
        if (center <= s) {
            leftVector.push_back(t);
        } else {
            rightVector.push_back(t);
        }
        // bool inLeft = false;
        // bool inRight = false;

        // if (t.v[0][a] <= s) {
        //     inLeft = true;
        // } else {
        //     inRight = true;
        // }

        // if (t.v[1][a] <= s) {
        //     inLeft = true;
        // } else {
        //     inRight = true;
        // }

        // if (t.v[2][a] <= s) {
        //     inLeft = true;
        // } else {
        //     inRight = true;
        // }

        // if (inLeft) {
        //     leftVector.push_back(t);
        // }
        // if (inRight) {
        //     rightVector.push_back(t);
        // }
    }

    // std::cerr << "nextId:" << nextId << std::endl;
    this->numNodes++;
    TreeNode* node = new TreeNode(l, a, s, false, std::vector<Triangle>(),
                                NULL, NULL, newBbox, nextId);
                                
    // std::cerr << "level expected:" << l << std::endl;
    // std::cerr << "level node:" << node->level << std::endl;

    // std::cerr << "nodeId:" << node->id << std::endl; //TODO: thisis buggy???
    // std::cerr << "nextId after creation:" << nextId << std::endl;

    // std::cerr << "s:" << s << std::endl;
    // std::cerr << "initial list size:" << ts.size() << ", left size:" << leftVector.size() << ", right size:" << rightVector.size() << std::endl;
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

void KdTree::createNodeArray() {
    TreeNode allNodes[this->numNodes];
    std::deque<TreeNode*> toVisit = {this->root};
    int counter = 0;
    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();
        curr->
    }
}

bool KdTree::hit(const ray& r, ray_hit& finalHitRec) {
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
            for (Triangle t : curr->triangles) {
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

bbox KdTree::boundFromList(std::vector<Triangle> *items) {
    float min_x = INFINITY;
    float min_y = INFINITY;
    float min_z = INFINITY;
    
    float max_x = -INFINITY;
    float max_y = -INFINITY;
    float max_z = -INFINITY;

    int count = items->size();

    Triangle t;

    for (int i=0; i < count; i++) {
        t = (*items)[i];
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

void KdTree::printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft)
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

void KdTree::printTree()
{
    this->printTreeHelper("", this->root, false);    
}