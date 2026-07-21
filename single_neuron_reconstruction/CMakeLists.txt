cmake_minimum_required(VERSION 3.22)
project(single_neuron_reconstruction LANGUAGES CXX)

aux_source_directory(./src/cpp SRC_CPP)
set(VAA3DPATH D:/ruanjian/Vaa3D)
set(TensorRT_PATH ~/TensorRT)
find_package(Qt4 REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 14)

set(SRC_CPP ${SRC_CPP} ${VAA3DPATH}/v3d_external/v3d_main/basic_c_fun/v3d_message.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/basic_c_fun/basic_surf_objs.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/basic_c_fun/basic_4dimage_create.cpp
        #${VAA3DPATH}/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2_connector.cpp
        #${VAA3DPATH}/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/vn_imgpreprocess.cpp
        #${VAA3DPATH}/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2/my_surf_objs.cpp
        ${VAA3DPATH}/v3d_external/v3d_main//worm_straighten_c/mst_prim_c.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/worm_straighten_c/bfs_1root.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/graph/dijk.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/neuron_editing/neuron_sim_scores.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/neuron_editing/v_neuronswc.cpp
        ${VAA3DPATH}/v3d_external/v3d_main/neuron_editing/global_feature_compute.cpp
        )
file(GLOB UI_FILES ./src/cpp/*.ui)
file(GLOB QRC_FILES ./images/*.qrc)
file(GLOB QPropertyCPP ${QT_INCLUDE_DIR}/../tools/shared/qtpropertybrowser/*.cpp)

include_directories(
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/src/cpp/include
        ${QT_INCLUDE_DIR}
	${VAA3DPATH}/v3d_external/v3d_main/common_lib/include
        ${VAA3DPATH}/v3d_external/v3d_main/basic_c_fun
        ${VAA3DPATH}/v3d_external/v3d_main/neuron_editing
        ${VAA3DPATH}/v3d_external/v3d_main/worm_straighten_c
        #${VAA3DPATH}/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2
        #${VAA3DPATH}/vaa3d_tools/released_plugins/v3d_plugins/neurontracing_vn2/app2
        ${TensorRT_PATH}/include
        ${CUDA_INCLUDE_DIRS}
        ${QT_INCLUDE_DIR}/../tools/shared/qtpropertybrowser
)

if (WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VAA3DPATH}/v3d_external/bin/plugins/D_LSNARS/${PROJECT_NAME}/)
endif (WIN32)
if (UNIX)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VAA3DPATH}/v3d_external/bin/plugins/D_LSNARS/${PROJECT_NAME}/)
endif (UNIX)

add_library(${PROJECT_NAME} SHARED ${SRC_CPP} ${UI_FILES} ${QRC_FILES} ${QPropertyCPP})
target_link_libraries(${PROJECT_NAME} Qt4::QtCore Qt4::QtGui ${CUDA_LIBRARIES})