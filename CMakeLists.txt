cmake_minimum_required(VERSION 3.21)
project(DSO LANGUAGES CXX)

add_compile_definitions(_WIN32_WINNT=0x0603 WINVER=0x0603)

# Optional: keep CMake away from conda
if(DEFINED CONDA_PATH)
  list(APPEND CMAKE_IGNORE_PREFIX_PATH
    "${CONDA_PATH}"
    "${CONDA_PATH}/Library"
  )
endif()

# MSVC language mode
if(MSVC)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  add_compile_options(/O2 /fp:fast /arch:AVX2)
  add_compile_options(/Qvec-report:1)
endif()

# -------------------------------
# vcpkg integration
# -------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  message(FATAL_ERROR "Configure with -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake")
endif()

# -------------------------------
# Libtorch via vcpkg
# -------------------------------
find_package(Torch CONFIG REQUIRED)
find_package(TBB REQUIRED)

add_executable(DSO
  src/main.cpp
)
target_include_directories(DSO PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(DSO PRIVATE torch TBB::tbb)

# Torch headers are usually pulled transitively, but this doesn't hurt:
target_compile_features(DSO PRIVATE cxx_std_23)

# On Windows, torch often ships DLLs; copying them next to the exe avoids runtime "missing DLL" issues.
if(WIN32)
  add_custom_command(TARGET DSO POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:DSO>
      $<TARGET_FILE_DIR:DSO>
    COMMAND_EXPAND_LISTS
  )
endif()
