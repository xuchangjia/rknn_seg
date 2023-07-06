cmake_minimum_required(VERSION 3.4.1)

project(PP_Seg_demo)

find_package(OpenCV REQUIRED)

# rknn api
set(RKNN_PATH ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api)

include_directories(
        ./src
        ${RKNN_PATH}/include
        ${OpenCV_INCLUDE_DIRS}
)

link_libraries(
    ${RKNN_PATH}/lib/librknnrt.so
    ${RKNN_PATH}/lib/librknn_api.so
    ${OpenCV_LIBRARIES}
)

add_executable(PP_Seg_demo
        test/main.cpp
        src/PPSeg.cpp
        )
