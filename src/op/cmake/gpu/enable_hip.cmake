enable_language(HIP)

if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
else()
    execute_process(
        COMMAND hipconfig -R
        RESULT_VARIABLE HIPCONFIG_RESULT
        OUTPUT_VARIABLE ROCM_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT HIPCONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to run hipconfig -R. Make sure DTK/ROCm is installed and hipconfig is available.")
    endif()
endif()

find_program(HIPIFY_TOOL hipify-perl
    HINTS
        "${ROCM_PATH}/bin"
        "${ROCM_PATH}/bin/hipify"
        "$ENV{ROCM_PATH}/bin"
        "$ENV{ROCM_PATH}/bin/hipify"
)
if(NOT HIPIFY_TOOL)
    message(FATAL_ERROR "hipify-perl not found! Cannot translate CUDA to HIP.")
endif()

set(MATPL_HIPIFY_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipified")

function(hipify_sources OUTPUT_VAR_NAME)
    set(GENERATED_HIP_FILES "")

    foreach(SRC_FILE ${ARGN})
        get_filename_component(FILE_ABS ${SRC_FILE} ABSOLUTE)
        get_filename_component(FILE_EXT ${SRC_FILE} EXT)
        file(RELATIVE_PATH FILE_REL "${CMAKE_SOURCE_DIR}" "${FILE_ABS}")

        if(FILE_EXT STREQUAL ".cu")
            string(REGEX REPLACE "\\.cu$" ".hip" OUT_REL "${FILE_REL}")
        else()
            set(OUT_REL "${FILE_REL}")
        endif()

        set(OUT_FILE "${MATPL_HIPIFY_OUTPUT_DIR}/${OUT_REL}")
        get_filename_component(OUT_DIR "${OUT_FILE}" DIRECTORY)

        add_custom_command(
            OUTPUT "${OUT_FILE}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
            COMMAND ${HIPIFY_TOOL} -print-stats -o "${OUT_FILE}" "${FILE_ABS}"
            COMMAND python3 -c "from pathlib import Path; p = Path('${OUT_FILE}'); s = p.read_text(); s = s.replace('c10/cuda/CUDAStream.h', 'c10/hip/HIPStream.h').replace('c10::cuda::getCurrentCUDAStream()', 'c10::hip::getCurrentHIPStream()').replace('// #include <hip/hip_runtime.h>', '#include <hip/hip_runtime.h>'); s = s if ('#include <hip/hip_runtime.h>' in s or '#include \"hip/hip_runtime.h\"' in s) else '#include <hip/hip_runtime.h>\\n' + s; p.write_text(s)"
            DEPENDS "${FILE_ABS}"
            COMMENT "Auto-hipifying ${FILE_REL}"
            VERBATIM
        )

        if(FILE_EXT STREQUAL ".cu")
            set_source_files_properties("${OUT_FILE}" PROPERTIES
                COMPILE_OPTIONS "-Wno-unused-result;-Wno-return-type"
                LANGUAGE HIP
            )
        else()
            set_source_files_properties("${OUT_FILE}" PROPERTIES GENERATED TRUE)
        endif()

        list(APPEND GENERATED_HIP_FILES "${OUT_FILE}")
    endforeach()

    set(${OUTPUT_VAR_NAME} ${GENERATED_HIP_FILES} PARENT_SCOPE)
endfunction()