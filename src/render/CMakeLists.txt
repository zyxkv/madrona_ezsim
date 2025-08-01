add_subdirectory(shader)

if (MADRONA_LINUX OR MADRONA_WINDOWS OR MADRONA_ANDROID OR MADRONA_MACOS)
    add_subdirectory(vk)
endif()

if (MADRONA_MACOS OR MADRONA_IOS)
    add_subdirectory(metal)
endif ()

#add_subdirectory(core)

add_library(madrona_rendering_system STATIC
    ${MADRONA_INC_DIR}/render/ecs.hpp ecs_interop.hpp ecs_system.cpp
    ${MADRONA_INC_DIR}/mesh_bvh.hpp 
    ${MADRONA_INC_DIR}/mesh_bvh.inl 
)

target_link_libraries(madrona_rendering_system PRIVATE
    madrona_mw_core
)

add_library(madrona_render_core STATIC
    ${MADRONA_INC_DIR}/render/render_mgr.hpp
        render_mgr.cpp
    ${MADRONA_INC_DIR}/render/common.hpp
    render_ctx.hpp render_ctx.cpp
    batch_renderer.hpp batch_renderer.cpp
    render_common.hpp
    image_util.cpp
)

target_compile_definitions(madrona_render_core PUBLIC
    "-DMADRONA_RENDER_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_link_libraries(madrona_render_core
    PUBLIC
        madrona_common
        madrona_render_vk
        madrona_render_shader_compiler
        madrona_importer
        madrona_rendering_system
    PRIVATE
        stb
)


add_library(madrona_render_asset_processor STATIC
    ${MADRONA_INC_DIR}/render/cuda_batch_render_assets.hpp
    ${MADRONA_INC_DIR}/render/asset_processor.hpp asset_processor.cpp
)

target_link_libraries(madrona_render_asset_processor PRIVATE
    madrona_bvh_builder
    madrona_common
    stb
)

if (TARGET madrona_cuda)
    target_link_libraries(madrona_render_asset_processor 
        PRIVATE
            madrona_cuda
    )
endif()


target_link_libraries(madrona_render_core 
    PUBLIC
        madrona_render_asset_processor
)


if (TARGET madrona_render_vk_cuda)
    target_link_libraries(madrona_render_core PUBLIC
        madrona_render_vk_cuda
    )
endif()

if (TARGET madrona_cuda)
    target_link_libraries(madrona_render_core PUBLIC
        madrona_cuda
    )
endif()

add_library(madrona_render STATIC
    ${MADRONA_INC_DIR}/render/api.hpp ../render/api.cpp
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(madrona_render PRIVATE
        MADRONA_VK_VALIDATE
    )
    message(STATUS "Building with Vulkan validation layers")
endif()

target_link_libraries(madrona_render
    PUBLIC
        madrona_common
    PRIVATE
        madrona_render_core
)