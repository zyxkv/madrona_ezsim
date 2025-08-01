#pragma once

#include <madrona/mesh_bvh.hpp>
#include <madrona/cuda_utils.hpp>

namespace madrona::render {

struct MeshBVHData {
    QBVHNode *nodes;
    uint64_t numNodes;

    MeshBVH::LeafMaterial *leafMaterial;
    uint64_t numLeaves;

    MeshBVH::BVHVertex *vertices;
    uint64_t numVerts;

    MeshBVH *meshBVHs;
    uint64_t numBVHs;
};

struct MaterialData {
    // GPU buffer containing array of texture objects
    cudaTextureObject_t *textures;
    uint32_t numTextureBuffers;
    cudaArray_t *textureBuffers;
    uint32_t numMipmapTextureBuffers;
    cudaMipmappedArray_t *mipmapTextureBuffers;
    Material *materials;
};


}
