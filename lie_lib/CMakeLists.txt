
# ----- Lie Algebra Library
set(LIE_LIBRARY lie)
if (${${PROJECT_NAME}_BUILD_SHARE})
    add_library(${LIE_LIBRARY} SHARED)
else(${${PROJECT_NAME}_BUILD_SHARE})
    add_library(${LIE_LIBRARY} STATIC)
endif(${${PROJECT_NAME}_BUILD_SHARE})

target_compile_features(${LIE_LIBRARY} PRIVATE ${${PROJECT_NAME}_COMPILE_FEATURE})
target_compile_options(${LIE_LIBRARY} PRIVATE ${${PROJECT_NAME}_WARNING_FLAGS})

option(${LIE_LIBRARY}_CHECK_Sim3_CACHE "Check cache consistency on internal SE3/Sim3 computation" OFF)
if (${LIE_LIBRARY}_CHECK_Sim3_CACHE)
    target_compile_definitions(${LIE_LIBRARY} PRIVATE CHECK_Sim3_CACHE=ON)
endif()
unset(${LIE_LIBRARY}_CHECK_Sim3_CACHE)

target_include_directories(${LIE_LIBRARY}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/lie_lib/inc)

target_link_libraries(${LIE_LIBRARY}
        PRIVATE
        Eigen3::Eigen
        Ceres::ceres)

target_sources(${LIE_LIBRARY} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lie_group.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lie_3d.cc)

