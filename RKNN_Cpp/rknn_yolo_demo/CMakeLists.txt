cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolo_demo_linux)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


set(TOOLCHAIN_DIR /opt/atk-dlrv1126-toolchain)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/usr/bin/arm-linux-gnueabihf-gcc)
set(SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/arm-buildroot-linux-gnueabihf/sysroot)


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "64bit")
	set(LIB_ARCH lib64)
else()
	message(STATUS "32bit")
	set(LIB_ARCH lib)
endif()

# opencv

#set(OpenCV_DIR /home/alientek/atk/sdk/buildroot/output/alientek_rv1126/build/opencv3-3.4.12/buildroot-build)

#find_package(OpenCV REQUIRED)

set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_features2d opencv_flann opencv_highgui opencv_videoio)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

#stb
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/)
include_directories(${CMAKE_SOURCE_DIR}/../)

include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/drm/include/libdrm)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/rga/include)

# include_directories(${CMAKE_SOURCE_DIR}/../easymedia)
# include_directories(${CMAKE_SOURCE_DIR}/../rkaiq/uAPI)
# include_directories(${CMAKE_SOURCE_DIR}/../rkaiq/xcore)
# include_directories(${CMAKE_SOURCE_DIR}/rkaiq/algos)
# include_directories(${CMAKE_SOURCE_DIR}/rkaiq/common)
# include_directories(${CMAKE_SOURCE_DIR}/rkaiq/iq_parser)
# include_directories(${CMAKE_SOURCE_DIR}/rknn)


set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_yolo_demo
        src/main.cc src/drm_func.c src/rga_func.c
)

target_link_libraries(rknn_yolo_demo
	${RKNN_API_LIB}
	${OPENCV_LIBS}
	dl
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolo_demo)
install(TARGETS rknn_yolo_demo DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
