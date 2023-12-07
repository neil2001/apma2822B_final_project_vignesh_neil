#include <vector>
#include "kdtree.h"
#include "../tracing/vec3.h"
#include <cmath>
#include <random>
#include <algorithm>


std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator

 float KdTree::quickSelectHelper(std::vector<float> &data, int k) {
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

 float KdTree::quickSelect(std::vector<int> ts, Axis a) {
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


