add_library(madrona_mw_cpu STATIC
    ${MADRONA_INC_DIR}/mw_cpu.hpp ${MADRONA_INC_DIR}/mw_cpu.inl cpu_exec.cpp
)

target_link_libraries(madrona_mw_cpu
    PUBLIC
        madrona_mw_core
)

if (NOT CUDAToolkit_FOUND OR CUDAToolkit_VERSION_MAJOR LESS 12)
    return()
endif()

# libcu++ headers currently fail to compile under MSVC
if (FRONTEND_MSVC)
    return()
endif()

add_library(madrona_mw_gpu STATIC
    cpp_compile.hpp cpp_compile.cpp
    cuda_compile_helpers.hpp cuda_compile_helpers.inl cuda_compile_helpers.cpp
    ${MADRONA_INC_DIR}/mw_gpu.hpp ${MADRONA_INC_DIR}/mw_ext_gpu_mem.hpp 
    cuda_exec.cpp ext_gpu_mem.cpp
)

target_link_libraries(madrona_mw_gpu
    PUBLIC
        madrona_hdrs madrona_cuda
    PRIVATE
        madrona_mw_core madrona_json
        CUDA::nvJitLink  # FIXME: CUDA::nvJitLink_static
)

set(MADRONA_MW_GPU_COMPILE_FLAGS
    -std=c++20
    -default-device
    -rdc=true
    -use_fast_math
    -DMADRONA_GPU_MODE=1
    -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    -DCCCL_DISABLE_BF16_SUPPORT=1
    -DCUB_DISABLE_BF16_SUPPORT=1
)

if (SKBUILD)
    set(MADRONA_SRC_DIR "./src")
else()
    set(MADRONA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

set(DEVICE_SRC_DIR "${MADRONA_SRC_DIR}/mw/device")

set(JOB_SYS_DEVICE_SOURCES
    ${DEVICE_SRC_DIR}/job.cpp
    ${DEVICE_SRC_DIR}/state.cpp
    ${DEVICE_SRC_DIR}/crash.cpp
    ${DEVICE_SRC_DIR}/const.cpp
)

set(BVH_DEVICE_SOURCES
    ${DEVICE_SRC_DIR}/bvh.cpp
    ${DEVICE_SRC_DIR}/memory.cpp
    ${DEVICE_SRC_DIR}/host_print.cpp
    ${DEVICE_SRC_DIR}/bvh_raycast.cpp
)

set(TASK_GRAPH_DEVICE_SOURCES
    ${DEVICE_SRC_DIR}/memory.cpp
    ${DEVICE_SRC_DIR}/state.cpp
    ${DEVICE_SRC_DIR}/crash.cpp
    ${DEVICE_SRC_DIR}/consts.cpp
    ${DEVICE_SRC_DIR}/taskgraph.cpp
    ${DEVICE_SRC_DIR}/taskgraph_utils.cpp
    ${DEVICE_SRC_DIR}/sort_archetype.cpp
    ${DEVICE_SRC_DIR}/host_print.cpp
    ${DEVICE_SRC_DIR}/ecs_render_bridge.cpp
    # FIXME
    ${MADRONA_SRC_DIR}/common/hashmap.cpp
    ${MADRONA_SRC_DIR}/common/navmesh.cpp
    ${MADRONA_SRC_DIR}/core/base.cpp
    ${MADRONA_SRC_DIR}/physics/physics.cpp
    ${MADRONA_SRC_DIR}/physics/geo.cpp
    ${MADRONA_SRC_DIR}/physics/xpbd.cpp
    ${MADRONA_SRC_DIR}/physics/tgs.cpp
    ${MADRONA_SRC_DIR}/physics/narrowphase.cpp
    ${MADRONA_SRC_DIR}/physics/broadphase.cpp
    ${MADRONA_SRC_DIR}/render/ecs_system.cpp
)

set(JOB_SYS_INTERNAL_SRC_LIST)
foreach(f ${JOB_SYS_DEVICE_SOURCES})
    set(JOB_SYS_INTERNAL_SRC_LIST "${JOB_SYS_INTERNAL_SRC_LIST}\"${f}\", ")
endforeach()
set(JOB_SYS_INTERNAL_SRC_LIST "${JOB_SYS_INTERNAL_SRC_LIST}")

set(TASK_GRAPH_INTERNAL_SRC_LIST)
foreach(f ${TASK_GRAPH_DEVICE_SOURCES})
    set(TASK_GRAPH_INTERNAL_SRC_LIST "${TASK_GRAPH_INTERNAL_SRC_LIST}\"${f}\", ")
endforeach()
set(TASK_GRAPH_INTERNAL_SRC_LIST "${TASK_GRAPH_INTERNAL_SRC_LIST}")

set(BVH_INTERNAL_SRC_LIST)
foreach(f ${BVH_DEVICE_SOURCES})
    set(BVH_INTERNAL_SRC_LIST "${BVH_INTERNAL_SRC_LIST}\"${f}\", ")
endforeach()
set(BVH_INTERNAL_SRC_LIST "${BVH_INTERNAL_SRC_LIST}")

get_target_property(MADRONA_HDRS_INCLUDES madrona_hdrs INTERFACE_INCLUDE_DIRECTORIES)

set(CCCL_OVERRIDE_INCLUDE_DIRS "")
if (SKBUILD)
    if (CUDAToolkit_VERSION_MINOR LESS 5)
        list(APPEND CCCL_OVERRIDE_INCLUDE_DIRS
            "./cccl/libcudacxx/include/"
            "./cccl/cub"
            "./cccl/thrust"
        )
    endif()

    set(NVRTC_INCLUDE_DIRS
        ${DEVICE_SRC_DIR}/include
        "./include"
        ${CCCL_OVERRIDE_INCLUDE_DIRS}
        "./cuda/include"
    )
else()
    if (CUDAToolkit_VERSION_MINOR LESS 5)
        set(CCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/cccl")
        list(APPEND CCCL_OVERRIDE_INCLUDE_DIRS
            "${CCCL_DIR}/libcudacxx/include/"
            "${CCCL_DIR}/cub"
            "${CCCL_DIR}/thrust"
        )
    endif()

    set(NVRTC_INCLUDE_DIRS
        ${DEVICE_SRC_DIR}/include
        ${MADRONA_HDRS_INCLUDES}
        ${CCCL_OVERRIDE_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()
set(NVRTC_INCLUDE_DIRS_LIST)
foreach(f ${NVRTC_INCLUDE_DIRS})
    set(NVRTC_INCLUDE_DIRS_LIST "${NVRTC_INCLUDE_DIRS_LIST}\"${f}\", ")
endforeach()
set(NVRTC_INCLUDE_DIRS_LIST "${NVRTC_INCLUDE_DIRS_LIST}")

set(NVRTC_OPTIONS)
foreach(f ${MADRONA_MW_GPU_COMPILE_FLAGS})
    set(NVRTC_OPTIONS "${NVRTC_OPTIONS}\"${f}\", ")
endforeach()
set(NVRTC_OPTIONS "${NVRTC_OPTIONS}")

target_compile_definitions(madrona_mw_gpu
    PUBLIC
        MADRONA_MWGPU_SUPPORT=1
    PRIVATE
        MADRONA_NVRTC_INCLUDE_DIRS=${NVRTC_INCLUDE_DIRS_LIST}
        MADRONA_NVRTC_OPTIONS=${NVRTC_OPTIONS}
        MADRONA_MW_GPU_JOB_SYS_INTERNAL_CPP=${JOB_SYS_INTERNAL_SRC_LIST}
        MADRONA_MW_GPU_TASK_GRAPH_INTERNAL_CPP=${TASK_GRAPH_INTERNAL_SRC_LIST}
        MADRONA_MW_GPU_BVH_INTERNAL_CPP=${BVH_INTERNAL_SRC_LIST}
        MADRONA_MW_GPU_DEVICE_SRC_DIR=\"${DEVICE_SRC_DIR}\"
)

# Helper function for library users to get list of sources / compileflags
function(madrona_build_compile_defns)
    set(one_val_args OUT_TARGET SOURCES_DEFN FLAGS_DEFN)
    set(multi_val_args SRCS EXISTING_TARGETS CUSTOM_FLAGS)

    cmake_parse_arguments(M "" "${one_val_args}" "${multi_val_args}"
                          ${ARGN})

    set(ALL_TARGET_SOURCES ${M_SRCS})
    list(FILTER ALL_TARGET_SOURCES INCLUDE REGEX ".+\.cpp")
    list(TRANSFORM ALL_TARGET_SOURCES REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "")
    if (SKBUILD)
        file(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR_REL
            "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
        list(TRANSFORM ALL_TARGET_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR_REL}/")
    else()
        list(TRANSFORM ALL_TARGET_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    endif()

    set(ALL_TARGET_INCLUDES "")
    foreach(t ${M_SRC_TARGETS})
        get_target_property(TARGET_DIR ${t} SOURCE_DIR) 

        get_target_property(TARGET_SOURCES ${t} SOURCES) 
        list(FILTER TARGET_SOURCES INCLUDE REGEX ".+\.cpp")
        list(TRANSFORM TARGET_SOURCES REPLACE "${TARGET_DIR}" "")
        list(TRANSFORM TARGET_SOURCES PREPEND "${TARGET_DIR}/")
        list(APPEND ALL_TARGET_SOURCES ${TARGET_SOURCES})

        get_target_property(TARGET_INCLUDES ${t} INCLUDE_DIRECTORIES) 
        list(APPEND ALL_TARGET_INCLUDES ${TARGET_INCLUDES})

        get_target_property(TARGET_INTERFACE_INCLUDES ${t}
            INTERFACE_INCLUDE_DIRECTORIES) 
        list(APPEND ALL_TARGET_INCLUDES ${TARGET_INTERFACE_INCLUDES})
    endforeach()

    set(SRC_LIST)
    foreach(src ${ALL_TARGET_SOURCES})
        set(SRC_LIST "${SRC_LIST}\"${src}\", ")
    endforeach()
    set(SRC_LIST "${SRC_LIST}")

    set(ALL_FLAGS)
    foreach(i ${ALL_TARGET_INCLUDES})
        set(ALL_FLAGS "${ALL_FLAGS}\"-I${i}\", ")
    endforeach()

    foreach(i ${M_CUSTOM_FLAGS})
        set(ALL_FLAGS "${ALL_FLAGS}\"${i}\", ")
    endforeach()

    set(ALL_FLAGS "${ALL_FLAGS}")

    add_library(${M_OUT_TARGET} INTERFACE)
    
    target_compile_definitions(${M_OUT_TARGET} INTERFACE
        ${M_SOURCES_DEFN}=${SRC_LIST}
        ${M_FLAGS_DEFN}=${ALL_FLAGS}
    )
endfunction()
