cmake_minimum_required (VERSION 3.16)

# On systems that need it, find QT based on the QTDIR environment variable
if (DEFINED ENV{QTDIR})
    set (CMAKE_PREFIX_PATH "$ENV{QTDIR}")
else()
    message (
        "Make sure the QTDIR environment variable is set."
    )
endif ()

# Widgets finds its own dependencies.
find_package (Qt5 COMPONENTS Widgets QUIET)

if (NOT Qt5Widgets_FOUND)
    message (
        "Qt example will not be build as no Qt was found! Please set the CMAKE_PREFIX_PATH to the lib/cmake folder inside your Qt installation"
    )
    return ()
endif ()

# As moc files are generated in the binary dir, tell CMake to always look for includes there:
set (CMAKE_INCLUDE_CURRENT_DIR ON)

set (CMAKE_PREFIX_PATH "../../../share")

set (SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
             "${CMAKE_CURRENT_SOURCE_DIR}/src/sampleQtViewer.cpp")

set (HEADERS)

set (MOC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/inc/sampleQtViewer.hpp")

set (UI_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow.ui")

find_package (royale REQUIRED)

link_directories (${royale_LIB_DIR})

qt5_wrap_ui (UI_HEADERS ${UI_FILES})
qt5_wrap_cpp (MOCED_HEADERS ${MOC_HEADERS})

project (sampleQtViewer)

add_executable (sampleQtViewer ${SOURCES} ${HEADERS} ${MOC_HEADERS} ${UI_HEADERS} ${MOCED_HEADERS}
                               ${ASSEMBLY_RC})

copy_royale_libs (sampleQtViewer)

target_include_directories (sampleQtViewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../inc
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_link_libraries (sampleQtViewer "${royale_LIBS}")
target_link_libraries (sampleQtViewer Qt5::Widgets)

set_target_properties (sampleQtViewer PROPERTIES WIN32_EXECUTABLE ON)
