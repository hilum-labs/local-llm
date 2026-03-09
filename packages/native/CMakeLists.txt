cmake_minimum_required(VERSION 3.14)
project(hilum_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Static linking — single .node file with no shared lib deps
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/hilum-local-llm-engine)

# Strip vendor components we don't ship — smaller binary, faster build
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "Skip llama.cpp tests"     FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Skip llama.cpp examples"  FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "Skip llama.cpp server"    FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "Skip llama.cpp CLI tools" FORCE)
set(LLAMA_TOOLS_INSTALL  OFF CACHE BOOL "Skip llama.cpp tool install" FORCE)

# Use native CPU instructions for maximum performance.
# CI overrides this to OFF for portable binaries via cmake_flags matrix.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "Optimize for host CPU" FORCE)
endif()

# Build hilum-local-llm-engine (llama + ggml)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Not needed — libhilum replaces common" FORCE)

# Enable libhilum
set(HILUM_BUILD_LIB ON CACHE BOOL "Build libhilum shared C API" FORCE)

add_subdirectory(
  ${VENDOR_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/hilum-local-llm-engine
)

# Some vendor snapshots may not honor HILUM_BUILD_LIB from parent cache.
# Ensure the shared C API target exists for linking when the hilum dir exists.
if(NOT TARGET hilum AND EXISTS "${VENDOR_DIR}/hilum/CMakeLists.txt")
  add_subdirectory(
    ${VENDOR_DIR}/hilum
    ${CMAKE_CURRENT_BINARY_DIR}/hilum-local-llm-engine/hilum
  )
endif()

if(NOT TARGET hilum)
  message(FATAL_ERROR
    "libhilum target not found. Ensure vendor/hilum-local-llm-engine includes the 'hilum/' C API directory, or update the submodule to a compatible revision.")
endif()

# MSVC emits a large volume of narrowing/sign-conversion warnings from vendor code.
# Suppress the noisy warning IDs only for targets that come from the vendor tree.
if(MSVC)
  get_property(_all_targets GLOBAL PROPERTY TARGETS)
  foreach(_t IN LISTS _all_targets)
    get_target_property(_type ${_t} TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY" OR _type STREQUAL "UTILITY")
      continue()
    endif()

    get_target_property(_src_dir ${_t} SOURCE_DIR)
    if(NOT _src_dir)
      continue()
    endif()

    string(FIND "${_src_dir}" "${VENDOR_DIR}" _is_vendor_target)
    if(_is_vendor_target EQUAL 0)
      target_compile_options(${_t} PRIVATE
        /wd4018 # signed/unsigned mismatch
        /wd4101 # unreferenced local variable
        /wd4244 # narrowing conversion
        /wd4267 # size_t -> smaller type
        /wd4305 # truncation to smaller floating type
      )
    endif()
  endforeach()
endif()

# Vendor uses CMAKE_SOURCE_DIR for its include path, which resolves to our
# project root instead of the vendor root. Fix by adding the correct path.
target_include_directories(llama PUBLIC $<BUILD_INTERFACE:${VENDOR_DIR}/include>)

# Find node-addon-api headers
execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# N-API addon
add_library(${PROJECT_NAME} SHARED
  src/addon.cpp
  src/llama_build_info_stub.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  SUFFIX ".node"
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${VENDOR_DIR}
  ${VENDOR_DIR}/hilum
  ${VENDOR_DIR}/vendor
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  NAPI_VERSION=8
  NAPI_DISABLE_CPP_EXCEPTIONS
)

# Windows static-link build: avoid dllimport (__imp_*) decoration for hilum symbols.
if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE HILUM_BUILD)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${CMAKE_JS_LIB}
  hilum
)

# On Apple, ensure Metal frameworks are linked
if(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
  )
endif()
