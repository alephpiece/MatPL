add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 1. Basic info
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(CalcOps)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

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

# 4. Detect if PyTorch was built with CUDA support (compile-time support)
execute_process(
    COMMAND python3 -c "
import torch
print('TRUE' if getattr(torch.version, 'cuda', None) is not None else 'FALSE')
"
    OUTPUT_VARIABLE PYTORCH_BUILT_WITH_CUDA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTHON_EXIT_CODE
)

# Handle Python execution errors
if(NOT PYTHON_EXIT_CODE EQUAL 0)
    message(WARNING "Failed to check PyTorch CUDA support. Assuming no CUDA.")
    set(PYTORCH_BUILT_WITH_CUDA "FALSE")
endif()

message(STATUS "PyTorch built with CUDA support: ${PYTORCH_BUILT_WITH_CUDA}")

# 5. Detect if CUDA toolkit is available on the system (for compilation)
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

# 6. Determine if GPU compilation should proceed
# We need BOTH: PyTorch built with CUDA AND CUDA toolkit available for compilation
if(PYTORCH_BUILT_WITH_CUDA AND SYSTEM_HAS_CUDA_TOOLKIT)
    set(COMPILE_WITH_GPU TRUE)
    message(STATUS "Enabling GPU compilation: PyTorch has CUDA support AND CUDA toolkit is available")
elseif(PYTORCH_BUILT_WITH_CUDA AND NOT SYSTEM_HAS_CUDA_TOOLKIT)
    set(COMPILE_WITH_GPU FALSE)
    message(WARNING "PyTorch was built with CUDA support but CUDA toolkit is not available for compilation.")
    message(WARNING "Please install CUDA toolkit or set CUDA_HOME environment variable.")
    message(WARNING "Building only CPU operators.")
elseif(NOT PYTORCH_BUILT_WITH_CUDA AND SYSTEM_HAS_CUDA_TOOLKIT)
    set(COMPILE_WITH_GPU FALSE)
    message(WARNING "CUDA toolkit is available but PyTorch was not built with CUDA support.")
    message(WARNING "Please install PyTorch with CUDA support.")
    message(WARNING "Building only CPU operators.")
else()
    set(COMPILE_WITH_GPU FALSE)
    message(STATUS "No CUDA support detected. Building only CPU operators.")
endif()

set(HAS_CUDA_SUPPORT ${COMPILE_WITH_GPU})

# 7. Always add CPU subdirectory
add_subdirectory(cmake/cpu)

# 8. Conditionally add GPU subdirectory
if(COMPILE_WITH_GPU)
    message(STATUS "Adding GPU subdirectory for CUDA compilation.")
    
    # Pass CUDA toolkit info to GPU subdirectory if needed
    if(DEFINED ENV{CUDA_HOME})
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME} CACHE PATH "CUDA Toolkit root directory")
    endif()
    if(NVCC)
        set(CUDA_NVCC_EXECUTABLE ${NVCC} CACHE FILEPATH "CUDA NVCC executable")
    endif()
    
    add_subdirectory(cmake/gpu)
else()
    message(STATUS "Skipping GPU subdirectory.")
endif()
