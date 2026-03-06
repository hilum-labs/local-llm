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

# Build hilum-local-llm-engine (llama + ggml + common for grammar support)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common library for grammar support" FORCE)
add_subdirectory(
  ${VENDOR_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/hilum-local-llm-engine
)

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
add_library(${PROJECT_NAME} SHARED src/addon.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  SUFFIX ".node"
)

# Build mtmd (multimodal support)
# LLAMA_INSTALL_VERSION is set inside the vendor's add_subdirectory scope,
# but mtmd needs it for its target properties. Provide a fallback.
if(NOT DEFINED LLAMA_INSTALL_VERSION)
  set(LLAMA_INSTALL_VERSION "0.0.0")
endif()
add_subdirectory(
  ${VENDOR_DIR}/tools/mtmd
  ${CMAKE_CURRENT_BINARY_DIR}/mtmd
  EXCLUDE_FROM_ALL
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/hilum-local-llm-engine/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/hilum-local-llm-engine/ggml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/hilum-local-llm-engine/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/hilum-local-llm-engine/vendor
  ${VENDOR_DIR}/tools/mtmd
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  NAPI_VERSION=8
  NAPI_DISABLE_CPP_EXCEPTIONS
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${CMAKE_JS_LIB}
  llama
  common
  mtmd
)

# On Apple, ensure Metal frameworks are linked
if(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
  )
endif()
