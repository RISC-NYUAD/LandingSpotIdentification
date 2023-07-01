cmake_minimum_required (VERSION 3.4)
project (ocams)
#https://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do
SET(DPYTHON_EXECUTABLE /usr/bin/python3.8)
SET(PYTHON_INCLUDE_DIR /usr/include/python3.8)
SET(PYTHON_LIBRARY /usr/lib/libpython3.8.so)
set(ENV{PYTHONPATH}  /usr/bin/python3.8)

set(ENV{DPYTHON_EXECUTABLE}  /usr/bin/python3.8)
set(ENV{PYTHON_INCLUDE_DIR}  /usr/include/python3.8)
set(ENV{PYTHON_LIBRARY}  /usr/lib/libpython3.8.so)

##export PYTHON_INCLUDE_DIRS=/usr/include/python3.8 
#export PYTHON_LIBRARIES=/usr/lib/python3.8/config/libpython3.8.so
##export PYTHON_LIBRARIES=/usr/lib/libpython3.8.so
##export DPYTHON_EXECUTABLE=/usr/bin/python3.8

#https://stackoverflow.com/questions/24174394/cmake-is-not-able-to-find-python-libraries
#cmake .. \
#-DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
#-DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")


INCLUDE_DIRECTORIES("/usr/include/python3.8")
find_package(Python COMPONENTS Interpreter Development)



#find_package(Python REQUIRED PATHS "/usr/bin/python3.8")

set(OpenCV_STATIC ON)

#set(OpenCV_DIR "~/libs/OpenCV-4.5.4/")
SET(OpenCV_DIR /home/risc/OPENCV/installation/OpenCV-4.5.0/)
#find_package(OpenCV 4.5.0 REQUIRED PATHS "~/libs/OpenCV-4.5.4/lib/cmake/opencv4/")
find_package(OpenCV 4.5.0 REQUIRED PATHS "/home/risc/OPENCV/installation/OpenCV-4.5.0/lib/cmake/opencv4/")


include_directories(${OpenCV_INCLUDE_DIRS})

#link_directories("~/libs/OpenCV-4.5.4/lib" )
#include_directories( "~/libs/OpenCV-4.5.4/lib"   )

add_library(darknet STATIC IMPORTED)
set_property(TARGET darknet PROPERTY IMPORTED_LOCATION "../libdarknetcpu.so")



add_definitions(-std=c++14)
#set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${CMAKE_CXX_FLAGS}")

#YOLO 4
set(OpenCV_LIBS1
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_dnn.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_features2d.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_ml.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_stitching.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_highgui.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_imgproc.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_core.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_video.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_imgcodecs.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_dnn.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_shape.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_calib3d.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_photo.so
#/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_viz.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_videostab.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_flann.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_videoio.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_objdetect.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_superres.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_rgbd.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_tracking.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_ccalib.so
#/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_sfm.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_datasets.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_xphoto.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_aruco.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_structured_light.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_face.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_plot.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_line_descriptor.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_hfs.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_saliency.so
#/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_hdf.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_fuzzy.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_optflow.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_xobjdetect.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_bioinspired.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_xfeatures2d.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_phase_unwrapping.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_dpm.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_ximgproc.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_reg.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_freetype.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_dnn_objdetect.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_bgsegm.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_surface_matching.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_text.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_stereo.so
/home/risc/softwares/Opencv/opencv/opencv4.5.4/lib/libopencv_img_hash.so
)



find_package(catkin REQUIRED COMPONENTS
  roscpp
  image_transport
  camera_info_manager
  sensor_msgs
)

find_package(Boost REQUIRED COMPONENTS system)

#find_package(aruco REQUIRED)


find_package (Eigen3 3.2 REQUIRED)

#https://blog.csdn.net/keeplearning365/article/details/121549806
#find_library(DNN_LIBS libopencv_dnn.so /home/risc/OPENCV/installation/OpenCV-4.5.0/lib/)


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS1}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    cv_bridge
    image_transport
    camera_info_manager
    sensor_msgs
)

include_directories( ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ./usr/include ./lib/include ./include /usr/include/pcl-1.7/ /usr/include/eigen3 ~/openCVmine/testROS/ocams/src/glm/gtc/ )

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS1}")
message(STATUS "PCL_INCLUDE_DIRS = ${PCL_INCLUDE_DIRS}")

set (CMAKE_CXX_FLAGS "-std=c++11")

###LANDING
include_directories(src/include/peac)
include_directories(src/include/MRF2.2)
link_directories(src/include/MRF2.2)
file(GLOB sources "src/plane_detection.cpp")

add_executable(track src/videostabKalman.cpp)
#add_executable(trackBALOONS src/videostabKalmanBALOONS.cpp)
#add_executable(clipbot src/clipbot.cpp)
#add_executable(trackLANDING src/videostabKalmanLANDING.cpp)
add_executable(trackLANDINGPlanes src/videostabKalmanLANDINGPlanes.cpp ${sources})
#target_link_libraries(webStream Qt5::Widgets)
#target_link_libraries(webStream Qt5::Core)

#target_include_directories(track PRIVATE "/home/risc/OPENCV/installation/OpenCV-4.5.0/lib/")
target_link_libraries(track ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} -lv4l2 -ludev -lSOIL -lGL -lGLEW -lsfml-window -lsfml-system  darknet -lm -luvc -lopencv_dnn ) 

#target_link_libraries(clipbot ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} -lv4l2 -ludev -lSOIL -lGL -lGLEW -lsfml-window -lsfml-system  darknet -lm -luvc) # Eigen3::Eigen)
#target_link_libraries(track ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${DNN_LIBS} -lv4l2 -ludev -lSOIL -lGL -lGLEW -lsfml-window -lsfml-system  darknet -lm -luvc -lopencv_dnn ) 
# Eigen3::Eigen)
#target_link_libraries(trackBALOONS ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} -lv4l2 darknet -lm -lrealsense2)
#target_link_libraries(trackLANDING ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} -lv4l2 darknet -lm -lrealsense2)
target_link_libraries(trackLANDINGPlanes ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} -lv4l2 -lm -lrealsense2 libMRF.a) 
