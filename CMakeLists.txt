cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0112 NEW)
project(DSO LANGUAGES CXX CUDA)

add_compile_definitions(_WIN32_WINNT=0x0603 WINVER=0x0603)

# -------------------------------------------------
# 1. Compiler Configuration
# -------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimisation flags
add_compile_options(/O2 /fp:fast /arch:AVX2)

# -------------------------------------------------
# 2. LibTorch Setup
# -------------------------------------------------
set(TORCH_INSTALL_DIR "C:/libtorch_cuda/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_DIR}")

find_package(Torch CONFIG REQUIRED)
find_package(TBB REQUIRED)

# -------------------------------------------------
# 3. Core Static Library
# -------------------------------------------------

# Collect all framework .cpp files
file(GLOB_RECURSE DSO_SOURCES
    CONFIGURE_DEPENDS
    src/*.cpp
)

add_library(dso_core STATIC
    ${DSO_SOURCES}
)

target_include_directories(dso_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dso_core
    PUBLIC
        ${TORCH_LIBRARIES}
        TBB::tbb
)

target_compile_features(dso_core PUBLIC cxx_std_23)

# Propagate Torch compile definitions
target_compile_definitions(dso_core PUBLIC ${TORCH_CXX_FLAGS})

# -------------------------------------------------
# 4. Experiment Executable
# -------------------------------------------------

# Get all subdirectories in 'experiments'
file(GLOB SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/experiments experiments/*)
if(WIN32)
    file(GLOB TORCH_DLLS
        "${TORCH_INSTALL_DIR}/lib/*.dll"
    )
endif()
foreach(DIR ${SUBDIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/experiments/${DIR})
        
        # Define the target name based on the folder name
        set(EXP_NAME "exp_${DIR}")

        # Collect all .cpp and .hpp files in THAT specific folder
        file(GLOB_RECURSE EXP_SOURCES 
            "experiments/${DIR}/*.cpp"
            "experiments/${DIR}/*.hpp"
        )

        add_executable(${EXP_NAME} ${EXP_SOURCES})
        
        target_link_libraries(${EXP_NAME} PRIVATE dso_core)
        target_include_directories(${EXP_NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            "experiments/${DIR}" # So #include "heston_utils.hpp" works
        )
        
        target_compile_features(${EXP_NAME} PRIVATE cxx_std_23)

        if(WIN32)
            add_custom_command(TARGET ${EXP_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:${EXP_NAME}>
                COMMAND_EXPAND_LISTS
                VERBATIM
            )
        endif()

    endif()
endforeach()