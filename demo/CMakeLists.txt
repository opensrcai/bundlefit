find_package(OpenCV 4.4 REQUIRED)

message("Building a demo")

list(APPEND DEMO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/demo_corridor_ba.cc )
list(APPEND DEMO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/demo_sim3_point_alignment.cc )

foreach(DEMO_SOURCE ${DEMO_SOURCES})
    get_filename_component(DEMO_MODULE_NAME  ${DEMO_SOURCE} NAME_WE)

    add_executable(${DEMO_MODULE_NAME} ${DEMO_SOURCE})

    target_compile_features(${DEMO_MODULE_NAME}  PRIVATE "cxx_std_17")
    target_compile_definitions(${DEMO_MODULE_NAME}  PRIVATE IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images/")

    target_link_libraries(${DEMO_MODULE_NAME}
            PRIVATE
            ${PROJECT_NAME}
            opencv_core
            opencv_highgui
            opencv_imgproc)
endforeach ()

