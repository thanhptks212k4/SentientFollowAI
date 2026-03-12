cmake_minimum_required(VERSION 3.10)
project(astra_driver)

set(CMAKE_CXX_STANDARD 17)

# 1. SETUP ORBBEC SDK (Giữ nguyên như cũ)
set(ORBBEC_SDK_DIR "/home/hoang/Documents/cam_depth/OrbbecSDK")
set(ORBBEC_LIB_DIR "${ORBBEC_SDK_DIR}/lib/linux_x64")

include_directories(${ORBBEC_SDK_DIR}/include)
link_directories(${ORBBEC_LIB_DIR})

# --- PHẦN MỚI: TÌM OPENCV ---
find_package(OpenCV REQUIRED)

add_executable(fast_exporter src/fast_exporter.cpp)

# --- PHẦN MỚI: LINK OPENCV ---
target_link_libraries(fast_exporter 
    OrbbecSDK 
    pthread
    ${OpenCV_LIBS}
)

# Copy thư viện Orbbec (Giữ nguyên logic Smart Copy cũ)
file(GLOB ALL_ORBBEC_LIBS "${ORBBEC_LIB_DIR}/*.so*")
foreach(LIB_FILE ${ALL_ORBBEC_LIBS})
    get_filename_component(FILE_NAME ${LIB_FILE} NAME)
    add_custom_command(TARGET fast_exporter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIB_FILE}
        ${CMAKE_BINARY_DIR}/${FILE_NAME}
    )
endforeach()
