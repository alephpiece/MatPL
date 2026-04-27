execute_process(
    COMMAND python3 -c "
try:
    import torch
    if torch._C._GLIBCXX_USE_CXX11_ABI:
        print('1')
    else:
        print('0')
except:
    print('0')  # 检测失败设置为0
"
    OUTPUT_VARIABLE TORCH_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_ABI STREQUAL "0")
    message(STATUS "set _GLIBCXX_USE_CXX11_ABI=0")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
else()
    message(STATUS "set _GLIBCXX_USE_CXX11_ABI=1(default)")
endif()

# 1. Basic info
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(CalcOps)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(WITH_OP_BACKEND "auto" CACHE STRING "Operator GPU backend: auto, cuda, hip, or cpu")
set_property(CACHE WITH_OP_BACKEND PROPERTY STRINGS auto cuda hip cpu)

# 2. Include Torch and Python paths
execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH)
string(REPLACE "\n" "" TORCH_CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
string(REPLACE "\n" "" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0])" OUTPUT_VARIABLE TORCH_PATH)
string(REPLACE "\n" "" TORCH_PATH ${TORCH_PATH})

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${TORCH_PATH}/include/)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include/)

# 3. Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# 4. Detect PyTorch GPU backend support (compile-time support)
execute_process(
    COMMAND python3 -c "
import torch
print('TRUE' if getattr(torch.version, 'cuda', None) is not None else 'FALSE')
"
    OUTPUT_VARIABLE PYTORCH_BUILT_WITH_CUDA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTHON_EXIT_CODE
)
execute_process(
    COMMAND python3 -c "
import torch
print('TRUE' if getattr(torch.version, 'hip', None) is not None else 'FALSE')
"
    OUTPUT_VARIABLE PYTORCH_BUILT_WITH_HIP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTHON_HIP_EXIT_CODE
)

# Handle Python execution errors
if(NOT PYTHON_EXIT_CODE EQUAL 0)
    message(WARNING "Failed to check PyTorch CUDA support. Assuming no CUDA.")
    set(PYTORCH_BUILT_WITH_CUDA "FALSE")
endif()
if(NOT PYTHON_HIP_EXIT_CODE EQUAL 0)
    message(WARNING "Failed to check PyTorch HIP support. Assuming no HIP.")
    set(PYTORCH_BUILT_WITH_HIP "FALSE")
endif()

message(STATUS "PyTorch built with CUDA support: ${PYTORCH_BUILT_WITH_CUDA}")
message(STATUS "PyTorch built with HIP support: ${PYTORCH_BUILT_WITH_HIP}")

# 5. Detect GPU toolchains available on the system (for compilation)
find_program(NVCC nvcc)
if(NVCC)
    set(SYSTEM_HAS_CUDA_TOOLKIT TRUE)
    message(STATUS "CUDA toolkit found: ${NVCC}")
else()
    set(SYSTEM_HAS_CUDA_TOOLKIT FALSE)
    message(STATUS "CUDA toolkit not found in PATH")
    
    # Check CUDA_HOME or CUDA_PATH as fallback
    if(DEFINED ENV{CUDA_HOME} AND EXISTS $ENV{CUDA_HOME}/bin/nvcc)
        set(SYSTEM_HAS_CUDA_TOOLKIT TRUE)
        set(NVCC $ENV{CUDA_HOME}/bin/nvcc)
        message(STATUS "Found CUDA toolkit via CUDA_HOME: $ENV{CUDA_HOME}")
    elseif(DEFINED ENV{CUDA_PATH} AND EXISTS $ENV{CUDA_PATH}/bin/nvcc)
        set(SYSTEM_HAS_CUDA_TOOLKIT TRUE)
        set(NVCC $ENV{CUDA_PATH}/bin/nvcc)
        message(STATUS "Found CUDA toolkit via CUDA_PATH: $ENV{CUDA_PATH}")
    endif()
endif()
find_program(HIPCC hipcc)
if(HIPCC)
    set(SYSTEM_HAS_HIP_TOOLKIT TRUE)
    message(STATUS "HIP compiler found: ${HIPCC}")
else()
    set(SYSTEM_HAS_HIP_TOOLKIT FALSE)
    message(STATUS "HIP compiler not found in PATH")
endif()

# 6. Determine if GPU compilation should proceed
# We need BOTH: PyTorch built with CUDA/HIP AND CUDA/HIP toolkit available for compilation
set(COMPILE_WITH_GPU FALSE)
set(WITH_GPU_BACKEND "")
string(TOLOWER "${WITH_OP_BACKEND}" WITH_OP_BACKEND_LOWER)

if(WITH_OP_BACKEND_LOWER STREQUAL "cpu")
    message(STATUS "WITH_OP_BACKEND=cpu. Building only CPU operators.")
elseif(WITH_OP_BACKEND_LOWER STREQUAL "hip")
    if(PYTORCH_BUILT_WITH_HIP AND SYSTEM_HAS_HIP_TOOLKIT)
        set(COMPILE_WITH_GPU TRUE)
        set(WITH_GPU_BACKEND "HIP")
        message(STATUS "Enabling HIP operator compilation.")
    else()
        message(FATAL_ERROR "WITH_OP_BACKEND=hip requested, but PyTorch HIP support or hipcc was not found.")
    endif()
elseif(WITH_OP_BACKEND_LOWER STREQUAL "cuda")
    if(PYTORCH_BUILT_WITH_CUDA AND SYSTEM_HAS_CUDA_TOOLKIT)
        set(COMPILE_WITH_GPU TRUE)
        set(WITH_GPU_BACKEND "CUDA")
        message(STATUS "Enabling CUDA operator compilation.")
    else()
        message(FATAL_ERROR "WITH_OP_BACKEND=cuda requested, but PyTorch CUDA support or nvcc was not found.")
    endif()
elseif(WITH_OP_BACKEND_LOWER STREQUAL "auto")
    if(PYTORCH_BUILT_WITH_HIP AND SYSTEM_HAS_HIP_TOOLKIT)
        set(COMPILE_WITH_GPU TRUE)
        set(WITH_GPU_BACKEND "HIP")
        message(STATUS "Auto-selected HIP operator backend.")
    elseif(PYTORCH_BUILT_WITH_CUDA AND SYSTEM_HAS_CUDA_TOOLKIT)
        set(COMPILE_WITH_GPU TRUE)
        set(WITH_GPU_BACKEND "CUDA")
        message(STATUS "Auto-selected CUDA operator backend.")
    else()
        message(STATUS "No compatible GPU operator backend detected. Building only CPU operators.")
    endif()
else()
    message(FATAL_ERROR "Unsupported WITH_OP_BACKEND='${WITH_OP_BACKEND}'. Use auto, cuda, hip, or cpu.")
endif()

# 7. Always add CPU subdirectory
add_subdirectory(cmake/cpu)

# 8. Conditionally add GPU subdirectory
if(COMPILE_WITH_GPU)
    message(STATUS "Adding GPU subdirectory for ${WITH_GPU_BACKEND} compilation.")
    
    # Pass CUDA toolkit info to GPU subdirectory if needed
    if(WITH_GPU_BACKEND STREQUAL "CUDA" AND DEFINED ENV{CUDA_HOME})
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME} CACHE PATH "CUDA Toolkit root directory")
    endif()
    if(WITH_GPU_BACKEND STREQUAL "CUDA" AND NVCC)
        set(CUDA_NVCC_EXECUTABLE ${NVCC} CACHE FILEPATH "CUDA NVCC executable")
    endif()
    
    add_subdirectory(cmake/gpu)
else()
    message(STATUS "Skipping GPU subdirectory.")
endif()
