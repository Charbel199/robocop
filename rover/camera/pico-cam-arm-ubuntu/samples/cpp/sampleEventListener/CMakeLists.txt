cmake_minimum_required (VERSION 3.16)

set (CMAKE_PREFIX_PATH "../../../share")

find_package (royale REQUIRED)

link_directories (${royale_LIB_DIR})

project (sampleEventListener)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../../inc)

add_executable (sampleEventListener sampleEventListener.cpp)

copy_royale_libs (sampleEventListener)

target_link_libraries (sampleEventListener "${royale_LIBS}")