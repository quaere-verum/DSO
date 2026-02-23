# cmake_minimum_required(VERSION 3.21)
# project(DSO LANGUAGES CXX)

# add_compile_definitions(_WIN32_WINNT=0x0603 WINVER=0x0603)

# # Optional: keep CMake away from conda
# if(DEFINED CONDA_PATH)
#   list(APPEND CMAKE_IGNORE_PREFIX_PATH
#     "${CONDA_PATH}"
#     "${CONDA_PATH}/Library"
#   )
# endif()

# # MSVC language mode
# if(MSVC)
#   set(CMAKE_CXX_STANDARD 23)
#   set(CMAKE_CXX_STANDARD_REQUIRED ON)
#   set(CMAKE_CXX_EXTENSIONS OFF)
#   add_compile_options(/O2 /fp:fast /arch:AVX2)
#   add_compile_options(/Qvec-report:1)
# endif()

# # -------------------------------
# # vcpkg integration
# # -------------------------------
# if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#   message(FATAL_ERROR "Configure with -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake")
# endif()

# # -------------------------------
# # Libtorch via vcpkg
# # -------------------------------
# find_package(Torch CONFIG REQUIRED)
# find_package(TBB REQUIRED)

# add_executable(DSO
#   src/main.cpp
# )
# target_include_directories(DSO PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/src
# )
# target_link_libraries(DSO PRIVATE torch TBB::tbb)

# # Torch headers are usually pulled transitively, but this doesn't hurt:
# target_compile_features(DSO PRIVATE cxx_std_23)

# # On Windows, torch often ships DLLs; copying them next to the exe avoids runtime "missing DLL" issues.
# if(WIN32)
#   add_custom_command(TARGET DSO POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       $<TARGET_RUNTIME_DLLS:DSO>
#       $<TARGET_FILE_DIR:DSO>
#     COMMAND_EXPAND_LISTS
#   )
# endif()
cmake_minimum_required(VERSION 3.21)
project(DSO LANGUAGES CXX)

add_compile_definitions(_WIN32_WINNT=0x0603 WINVER=0x0603)

# MSVC language mode
if(MSVC)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  add_compile_options(/O2 /fp:fast /arch:AVX2)
endif()

# -------------------------------
# 1. Manual LibTorch Setup
# -------------------------------
# We set this BEFORE find_package so it overrides vcpkg
set(TORCH_INSTALL_DIR "C:/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_DIR}")

# -------------------------------
# 2. Package Discovery
# -------------------------------
# This will now find the version in C:/users/bramb/libtorch
find_package(Torch CONFIG REQUIRED)

# This will still use vcpkg to find TBB
find_package(TBB REQUIRED)

# -------------------------------
# 3. Executable & Linking
# -------------------------------
add_executable(DSO src/main.cpp)

target_include_directories(DSO PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Use ${TORCH_LIBRARIES} for the pre-built version as it handles 
# all the sub-libraries (c10, torch_cpu, etc.) correctly.
target_link_libraries(DSO PRIVATE ${TORCH_LIBRARIES} TBB::tbb)

target_compile_features(DSO PRIVATE cxx_std_23)

# -------------------------------
# 4. DLL Handling
# -------------------------------
if(WIN32)
  # NOTE: $<TARGET_RUNTIME_DLLS> works great for vcpkg, but 
  # sometimes struggles with manual LibTorch installs.
  # If the EXE fails to start, you may need to manually copy 
  # the DLLs from C:/users/bramb/libtorch/lib to your build folder.
  add_custom_command(TARGET DSO POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:DSO>
      $<TARGET_FILE_DIR:DSO>
    COMMAND_EXPAND_LISTS
  )
endif()