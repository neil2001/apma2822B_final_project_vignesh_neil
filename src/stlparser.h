#ifndef STLPARSERH
#define STLPARSERH

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <array>

#include "vec3.h"
#include "triangle.h"
#include "stlobject.h"

#define HEADER_LENGTH 81

template <typename T>

struct Facet
{
    float normal[3];
    float v1[3];
    float v2[3];
    float v3[3];
};

class StlParser{

public:
    __host__ StlParser() {}
    __host__ std::vector<Triangle> parseFile(const char* filename);

private: 
    __host__ static std::ifstream open_binary_stl(const char* filename) {
        return std::ifstream{filename, std::ifstream::in | std::ifstream::binary};
    }

    __host__ static void read_binary_value(std::ifstream& in, T* dst) {
        in.read(as_char_ptr(dst), sizeof(T));
    }

    __host__ static char* as_char_ptr(T* pointer) {
        return reinterpret_cast<char*>(pointer);
    }

    __host static void read_binary_array(std::ifstream& in, T* dst, size_t array_length) {
        size_t n_bytes = array_length * sizeof(T);
        in.read(as_char_ptr(dst), n_bytes);
    }
}

__host__ std::vector<Triangle> StlParser::parseFile(const char* filename) {
    std::ifstream in = open_binary_stl(filename);

    char header[HEADER_LENGTH];
    unsigned int triangle_count;

    read_binary_array<char>(in, header, 80);
    header[80] = '\0';

    read_binary_value<unsigned int>(in, &triangle_count);

    std::vector<Triangle> triangles;
    triangles.reserve(triangle_count);

    for (int i=0; i<triangle_count; i++) {
        Facet f{};

        read_binary_array<float>(in, f.normal, 3);
        read_binary_array<float>(in, f.v1, 3);
        read_binary_array<float>(in, f.v2, 3);
        read_binary_array<float>(in, f.v3, 3);
        
        triangles.emplace_back(
            Triangle(vec3(f.v1), vec3(f.v2), vec3(f.v3), vec3(f.normal))
        );
    }

    return triangles;
}

#endif