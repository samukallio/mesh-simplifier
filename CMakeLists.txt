cmake_minimum_required(VERSION 3.8)

project(simplify)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

add_executable(simplify simplify.cpp)

set_property(TARGET simplify PROPERTY CXX_STANDARD 20)

add_subdirectory(glm)

target_link_libraries(simplify glm-header-only)

# target_include_directories(simplify PUBLIC ${GLM_INCLUDE_DIRS})
# target_link_libraries(simplify ${GLM_LIBRARY})
