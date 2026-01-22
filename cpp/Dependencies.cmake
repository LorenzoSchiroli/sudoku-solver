# Dependencies.cmake

set(DEPS_DIR ${CMAKE_SOURCE_DIR}/../deps)

# ================= OpenCV =================

if(ANDROID)
    set(OPENCV_ROOT ${DEPS_DIR}/opencv/android/OpenCV-android-sdk/sdk/native)

    add_library(opencv SHARED IMPORTED)
    set_target_properties(opencv PROPERTIES
        IMPORTED_LOCATION
            ${OPENCV_ROOT}/libs/${ANDROID_ABI}/libopencv_java4.so
        INTERFACE_INCLUDE_DIRECTORIES
            ${OPENCV_ROOT}/jni/include
    )

    set(OpenCV_LIBS opencv)

elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_library(opencv INTERFACE)
    target_include_directories(opencv INTERFACE
        ${DEPS_DIR}/opencv/ios/opencv2.framework/Headers
    )
    target_link_libraries(opencv INTERFACE
        ${DEPS_DIR}/opencv/ios/opencv2.framework
    )

    set(OpenCV_LIBS opencv)

elseif(APPLE AND NOT IOS)
    find_package(OpenCV CONFIG REQUIRED)
    set(OpenCV_LIBS ${OpenCV_LIBRARIES})
endif()

# ================= ONNX Runtime =================

if(ANDROID)
    add_library(onnx_lib SHARED IMPORTED)
    set_target_properties(onnx_lib PROPERTIES
        IMPORTED_LOCATION "${DEPS_DIR}/onnxruntime/android/jni/${ANDROID_ABI}/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${DEPS_DIR}/onnxruntime/android/headers"
    )
    add_library(onnxruntime::onnxruntime ALIAS onnx_lib)

elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_library(onnxruntime INTERFACE)
    target_include_directories(onnxruntime INTERFACE
        ${DEPS_DIR}/onnxruntime/ios/onnxruntime.framework/Headers
    )
    target_link_libraries(onnxruntime INTERFACE
        ${DEPS_DIR}/onnxruntime/ios/onnxruntime.framework
    )

elseif(APPLE AND NOT IOS)
    # find_package(onnxruntime CONFIG REQUIRED)
    # add_library(onnxruntime ALIAS onnxruntime::onnxruntime)

    # find_package(onnxruntime CONFIG REQUIRED)
    # target_include_directories(onnxruntime::onnxruntime
    #     INTERFACE
    #     /opt/homebrew/Cellar/onnxruntime/1.23.2_2/include
    # )

    find_package(onnxruntime CONFIG REQUIRED)
    file(GLOB ONNX_INCLUDE_DIR "/opt/homebrew/Cellar/onnxruntime/*/include")
    target_include_directories(onnxruntime::onnxruntime INTERFACE ${ONNX_INCLUDE_DIR})
endif()

