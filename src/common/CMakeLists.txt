set(MADRONA_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona"
    CACHE INTERNAL "")

add_library(madrona_hdrs INTERFACE)
target_include_directories(madrona_hdrs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(madrona_hdrs INTERFACE
    madrona_sys_defns)

if (FRONTEND_GCC)
    target_compile_options(madrona_hdrs INTERFACE
        -fdiagnostics-color=always  
    )
elseif (FRONTEND_CLANG)
    target_compile_options(madrona_hdrs INTERFACE
        -fcolor-diagnostics -Wshadow
    )
endif ()

if (FRONTEND_GCC OR FRONTEND_CLANG)
    target_compile_options(madrona_hdrs INTERFACE
        -pedantic -Wall -Wextra
    )

    if (MADRONA_X64 AND MADRONA_LINUX)
        target_compile_options(madrona_hdrs INTERFACE
            -march=x86-64-v3
        )
    elseif (MADRONA_ARM AND MADRONA_MACOS)
        target_compile_options(madrona_hdrs INTERFACE
            -mcpu=apple-m1
        )
    endif()
elseif (FRONTEND_MSVC)
    # FIXME: some of these options (/permissive-, /Zc:__cplusplus,
    # /Zc:preprocessor) should just be applied globally to the toolcahin
    target_compile_options(madrona_hdrs INTERFACE
        /Zc:__cplusplus
        /permissive-
        /W4
        /wd4324 # Struct padded for alignas ... yeah that's the point
        /wd4701 # Potentially uninitialized variable. MSVC analysis really sucks on this
        /wd4244 /wd4267 # Should reenable these
    )

    if (MSVC_VERSION LESS 1936)
        target_compile_options(madrona_hdrs INTERFACE
            /wd5223 /wd5030 # msvc::lifetimebound not supported in older msvc, causes warnings
        )
    endif()

    if (NOT FRONTEND_CLANG_CL)
        target_compile_options(madrona_hdrs INTERFACE
            /Zc:preprocessor
        )
    endif()
endif()

add_library(madrona_err STATIC
    ${MADRONA_INC_DIR}/crash.hpp crash.cpp
)

target_link_libraries(madrona_err
    PUBLIC
        madrona_hdrs
    PRIVATE
        madrona_noexceptrtti
)

if (MADRONA_WINDOWS)
    set(MADRONA_STD_MEM_LIB_TYPE STATIC)
else()
    set(MADRONA_STD_MEM_LIB_TYPE SHARED)
endif()

add_library(madrona_std_mem ${MADRONA_STD_MEM_LIB_TYPE}
    op_new_delete.cpp
)

target_link_libraries(madrona_std_mem PRIVATE
    madrona_hdrs madrona_noexceptrtti madrona_libcxx
)

add_library(madrona_mem STATIC
    ${MADRONA_INC_DIR}/memory.hpp ${MADRONA_INC_DIR}/memory.inl memory.cpp
    ${MADRONA_INC_DIR}/stack_alloc.hpp ${MADRONA_INC_DIR}/stack_alloc.inl
        stack_alloc.cpp
)

target_link_libraries(madrona_mem PRIVATE
  madrona_hdrs madrona_noexceptrtti)

if (NOT WIN32)
    target_link_libraries(madrona_mem PRIVATE
        madrona_libcxx_hdrs 
    )
endif()

add_library(madrona_common STATIC
    ${MADRONA_INC_DIR}/heap_array.hpp
    ${MADRONA_INC_DIR}/span.hpp
    ${MADRONA_INC_DIR}/math.hpp ${MADRONA_INC_DIR}/math.inl
    ${MADRONA_INC_DIR}/rand.hpp ${MADRONA_INC_DIR}/rand.inl
    ${MADRONA_INC_DIR}/utils.hpp ${MADRONA_INC_DIR}/utils.inl
    ${MADRONA_INC_DIR}/ecs.hpp ${MADRONA_INC_DIR}/ecs.inl
    ${MADRONA_INC_DIR}/type_tracker.hpp ${MADRONA_INC_DIR}/type_tracker.inl
        type_tracker.cpp
    ${MADRONA_INC_DIR}/hashmap.hpp ${MADRONA_INC_DIR}/hashmap.inl hashmap.cpp
    ${MADRONA_INC_DIR}/table.hpp ${MADRONA_INC_DIR}/table.inl table.cpp
    ${MADRONA_INC_DIR}/virtual.hpp virtual.cpp
    ${MADRONA_INC_DIR}/tracing.hpp tracing.cpp
    ${MADRONA_INC_DIR}/io.hpp io.cpp
    #${MADRONA_INC_DIR}/hash.hpp
    #${INC_DIR}/platform_utils.hpp ${INC_DIR}/platform_utils.inl
    #    platform_utils.cpp
)

option(MADRONA_ENABLE_TRACING "Enable tracing" OFF)
if (MADRONA_ENABLE_TRACING)
    target_compile_definitions(madrona_common PUBLIC
        MADRONA_TRACING=1
    )
endif()

target_link_libraries(madrona_common
    PUBLIC 
        madrona_hdrs
        madrona_libcxx
        madrona_err
        madrona_noexceptrtti
        madrona_std_mem
        madrona_mem
)

set_property(TARGET madrona_common PROPERTY
    POSITION_INDEPENDENT_CODE TRUE)
set_property(TARGET madrona_common PROPERTY
    INTERFACE_POSITION_INDEPENDENT_CODE TRUE)

add_library(madrona_bvh_builder STATIC
    ${MADRONA_INC_DIR}/mesh_bvh_builder.hpp mesh_bvh_builder.cpp
)

target_link_libraries(madrona_bvh_builder
    PUBLIC
        madrona_common
    PRIVATE
        madrona_embree
)

add_library(madrona_navmesh STATIC
    ${MADRONA_INC_DIR}/navmesh.hpp ${MADRONA_INC_DIR}/navmesh.inl navmesh.cpp
)

target_link_libraries(madrona_navmesh PUBLIC madrona_common)

if (CUDAToolkit_FOUND)
    add_library(madrona_cuda STATIC
        ${MADRONA_INC_DIR}/cuda_utils.hpp ${MADRONA_INC_DIR}/cuda_utils.inl
            cuda_utils.cpp
    )

    target_compile_definitions(madrona_cuda
        PUBLIC
            MADRONA_CUDA_SUPPORT=1

            # CCCL is bugged and will define __sanitizer_annotate_contiguous_container
            # with visibility hidden if ASAN is detected. This is a problem because
            # address sanitizer actually provides this symbol with default visibility.
            # This causes linker failures if compiling a shared object, since the
            # linker needs to find a definition for undefined hidden symbols, but
            # ASAN is expecting to provide that definition at runtime (for example
            # in a preloaded shared-libasan, or linked into the executable
            _LIBCUDACXX_HAS_NO_ASAN=1
    )

    target_link_libraries(madrona_cuda
        PUBLIC
            CUDA::cudart_static
        PRIVATE
            madrona_err
            madrona_noexceptrtti
    )

    if (NOT WIN32) # FIXME
        target_link_libraries(madrona_cuda PRIVATE 
            madrona_libcxx_hdrs
        )
    endif ()

    if (WIN32)
        get_property(MULTI_CFG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if (MULTI_CFG)
            set(LIB_OUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>")
        else()
            set(LIB_OUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        endif()

        file(GLOB CUDA_LIB_SRC "${CUDAToolkit_BIN_DIR}/cudart*.dll")
        cmake_path(GET CUDA_LIB_SRC FILENAME CUDA_LIB_NAME)

        set(CUDA_LIB_COPY "${LIB_OUT_DIR}/${CUDA_LIB_NAME}")
        add_custom_command(
            OUTPUT "${CUDA_LIB_COPY}"
            DEPENDS "${CUDA_LIB_SRC}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDA_LIB_SRC}" "${CUDA_LIB_COPY}"
        )
        add_custom_target(cuda_lib_copy
            DEPENDS "${CUDA_LIB_COPY}"
        )

        add_dependencies(madrona_cuda cuda_lib_copy)
    endif()
endif ()
