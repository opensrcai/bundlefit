cmake_minimum_required(VERSION 3.20)
project(bundlefit)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# -- library type
option(${PROJECT_NAME}_BUILD_SHARE "build bundlefit as a shared library" OFF)

# -- sanitizer
option(${PROJECT_NAME}_USE_SANITIZER "Use ASAN address sanitizer" OFF)
if (${PROJECT_NAME}_USE_SANITIZER)
    message("Enable: sanitizer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
else ()
    message("Disable: sanitizer")
endif (${PROJECT_NAME}_USE_SANITIZER)
unset(${PROJECT_NAME}_USE_SANITIZER)

# -- Use C++20 compiler
set(${PROJECT_NAME}_COMPILE_FEATURE "cxx_std_20")

# -- compile warning
set(${PROJECT_NAME}_CLANG_WARNING_FLAGS -Wall -Wextra -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-gnu-zero-variadic-macro-arguments -Wno-vla-extension -Wno-vla)
set(${PROJECT_NAME}_GNU_WARNING_FLAGS -Wall -Wextra -Wformat-nonliteral -Wcast-align -Wpointer-arith
        -Wmissing-declarations -Winline -Wundef
        -Wcast-qual -Wshadow -Wwrite-strings
        -Wfloat-equal)
# set(${PROJECT_NAME}_MSVC_WARNING_FLAGS /W4)
set(${PROJECT_NAME}_MSVC_WARNING_FLAGS)
set(${PROJECT_NAME}_WARNING_FLAGS $<$<CXX_COMPILER_ID:Clang>:${${PROJECT_NAME}_CLANG_WARNING_FLAGS}>$<$<CXX_COMPILER_ID:GNU>:${${PROJECT_NAME}_GNU_WARNING_FLAGS}>$<$<CXX_COMPILER_ID:MSVC>:${${PROJECT_NAME}_MSVC_WARNING_FLAGS}>)

add_subdirectory(lie_lib)

# ------ bundlefit Library
if (${${PROJECT_NAME}_BUILD_SHARE})
    add_library(${PROJECT_NAME} SHARED)
else(${${PROJECT_NAME}_BUILD_SHARE})
    add_library(${PROJECT_NAME} STATIC)
endif(${${PROJECT_NAME}_BUILD_SHARE})

if (ENABLE_TEMPORARY_CODE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TEMPORARY_CODE=ON)
endif (ENABLE_TEMPORARY_CODE)

target_compile_features(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_COMPILE_FEATURE})

target_compile_options(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_WARNING_FLAGS})

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/inc)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        Eigen3::Eigen
        lie
        Ceres::ceres)

add_subdirectory(src/bundlefit)

option(${PROJECT_NAME}_BUILD_DEMO "Build demo" OFF)
if (${PROJECT_NAME}_BUILD_DEMO)
  add_subdirectory(demo)
endif (${PROJECT_NAME}_BUILD_DEMO)

option(${PROJECT_NAME}_ENABLE_PYTHON "Enable python" OFF)
if (${PROJECT_NAME}_ENABLE_PYTHON)
    message("Enable: Python")
    add_subdirectory(python_wrap)
else ()
    message("Disable: Python")
endif (${PROJECT_NAME}_ENABLE_PYTHON)
unset(${PROJECT_NAME}_ENABLE_PYTHON)
