cmake_minimum_required(VERSION 3.21)
project(DSO LANGUAGES CXX)

add_compile_definitions(_WIN32_WINNT=0x0603 WINVER=0x0603)

# -------------------------------------------------
# 1. Compiler Configuration
# -------------------------------------------------
if(MSVC)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Optimisation flags
  add_compile_options(/O2 /fp:fast /arch:AVX2)
endif()

# -------------------------------------------------
# 2. LibTorch Setup
# -------------------------------------------------
set(TORCH_INSTALL_DIR "C:/libtorch")
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

file(GLOB EXPERIMENT_SOURCES
    CONFIGURE_DEPENDS
    experiments/*.cpp
)

foreach(EXP_SRC ${EXPERIMENT_SOURCES})

    # Extract filename without extension
    get_filename_component(EXP_NAME ${EXP_SRC} NAME_WE)

    add_executable(${EXP_NAME}
        ${EXP_SRC}
    )

    target_link_libraries(${EXP_NAME}
        PRIVATE
            dso_core
    )

    target_include_directories(${EXP_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_compile_features(${EXP_NAME} PRIVATE cxx_std_23)

    if(WIN32)
        add_custom_command(TARGET ${EXP_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${EXP_NAME}>
                $<TARGET_FILE_DIR:${EXP_NAME}>
            COMMAND_EXPAND_LISTS
        )
    endif()

endforeach()