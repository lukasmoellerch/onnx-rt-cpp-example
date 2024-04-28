cmake_minimum_required(VERSION 3.10)

project(Tutorial)

find_package(onnxruntime REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main onnxruntime::onnxruntime)