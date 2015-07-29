/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define DEBUG 1

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

#define CHECK_CL_ERROR(err) if (err != CL_SUCCESS) { fprintf(stderr, "Error: OpenCL returned error %i\n", err); goto error; }
#define CHECK_ALLOCATION(ptr) if (!ptr) { fprintf(stderr, "Error: Memory allocation failure\n"); goto error; }

int get_desired_platform(const char *substr,
                         cl_platform_id *platform_id_out,
                         cl_int *err)
{
    cl_int _err = CL_SUCCESS;
    cl_uint i, num_platforms, selected_platform_idx;
    cl_platform_id *platform_ids = NULL;
    char *platform_name = NULL;

    assert(platform_id_out != NULL);

    if (!err)
        err = &_err;

    *err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_CL_ERROR(*err);

    platform_ids = malloc(sizeof(*platform_ids) * num_platforms);
    CHECK_ALLOCATION(platform_ids);

    *err = clGetPlatformIDs(num_platforms, platform_ids, NULL);
    CHECK_CL_ERROR(*err);

    for (i = 0; i < num_platforms; i++) {
        size_t platform_name_size;

        *err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL,
                                 &platform_name_size);
        CHECK_CL_ERROR(*err);

        platform_name = realloc(platform_name,
                                sizeof(*platform_name) * platform_name_size);
        CHECK_ALLOCATION(platform_name);

        *err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME,
                                 platform_name_size, platform_name, NULL);
        CHECK_CL_ERROR(*err);

        if (DEBUG)
            printf("Platform %u: \"%s\"\n", i, platform_name);

        if (strstr(platform_name, substr)) {
            selected_platform_idx = i;
            break;
        }
    }

    *platform_id_out = platform_ids[selected_platform_idx];

    free(platform_ids);
    free(platform_name);
    return 0;
error:
    free(platform_ids);
    free(platform_name);
    return -1;
}

int get_gpu_device_id(cl_platform_id platform_id,
                      cl_device_id *device_out,
                      cl_bool fallback,
                      cl_int *err)
{
    cl_int _err = CL_SUCCESS;
    cl_uint num_devices = 1;

    assert(device_out != NULL);

    if (!err)
        err = &_err;

    /* TODO: multi-gpu / multi-device? */
    *err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, device_out,
                          NULL);
    if (*err == CL_DEVICE_NOT_FOUND && !fallback) {
        fprintf(stderr, "Error: No GPU devices found\n");
        goto error;
    }
    else if (*err != CL_DEVICE_NOT_FOUND) {
        CHECK_CL_ERROR(*err);
        return 0;
    }

    *err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                          device_out, NULL);
    if (*err == CL_DEVICE_NOT_FOUND) {
        fprintf(stderr, "Error: No devices found\n");
        goto error;
    }
    CHECK_CL_ERROR(*err);

    return 0;
error:
    return -1;
}

int create_context(cl_platform_id platform,
                   cl_device_id device,
                   cl_context *context_out,
                   cl_int *err)
{
    cl_int _err = CL_SUCCESS;
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    assert(context_out != NULL);

    if (!err)
        err = &_err;

    *context_out = clCreateContext(context_properties, 1, &device, NULL, NULL,
                                   err);
    CHECK_CL_ERROR(*err);

    return 0;
error:
    return -1;
}

int build_program_from_file(const char *filename,
                            const char *options,
                            cl_context context,
                            cl_device_id device,
                            cl_int *err)
{
    cl_int _err;
    FILE *file;
    char *program_source = NULL;
    size_t program_source_size;
    cl_program program = NULL;
    char *build_log = NULL;

    assert(filename != NULL);

    if (!err)
        err = &_err;

    file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Couldn't open file \"%s\"\n", filename);
        goto error;
    }

    if (fseek(file, 0L, SEEK_END)) {
        fprintf(stderr, "Error: cannot determine file size of \"%s\"\n", filename);
        goto error;
    }
    program_source_size = ftell(file);
    if (fseek(file, 0L, SEEK_SET)) {
        fprintf(stderr, "Error: cannot determine file size of \"%s\"\n", filename);
        goto error;
    }

    program_source = malloc(sizeof(*program_source) * (program_source_size + 1));
    CHECK_ALLOCATION(program_source);

    if (fread(program_source, 1, program_source_size, file) != program_source_size) {
        fprintf(stderr, "Error: failed to read file \"%s\"\n", filename);
        goto error;
    }
    program_source[program_source_size] = '\0';

    fclose(file);

    program = clCreateProgramWithSource(context, 1, (const char **)&program_source, NULL, err);
    CHECK_CL_ERROR(*err);

    *err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (*err == CL_BUILD_PROGRAM_FAILURE) {
        size_t build_log_size;

        *err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        CHECK_CL_ERROR(*err);

        build_log = malloc(sizeof(*build_log) * build_log_size);
        CHECK_ALLOCATION(build_log);

        *err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
        CHECK_CL_ERROR(*err);

        fprintf(stderr, "Error: Failed to build program in file %s\n", filename);
        fprintf(stderr, "       with options %s\n\n", options ? options : "[NULL]");
        fprintf(stderr, "================================== BUILD LOG ===================================\n\n");
        fprintf(stderr, "%s", build_log);
        goto error;
    }
    CHECK_CL_ERROR(*err);

    return 0;
error:
    free(build_log);
    if (program)
        clReleaseProgram(program);
    free(program_source);
    return -1;
}

int main(void)
{
    cl_int err = CL_SUCCESS;
    cl_platform_id selected_platform;
    cl_device_id selected_device;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    if (get_desired_platform("NVIDIA", &selected_platform, &err))
        goto error;

    if (get_gpu_device_id(selected_platform, &selected_device, CL_TRUE, &err))
        goto error;

    if (create_context(selected_platform, selected_device, &context, &err))
        goto error;

    if (build_program_from_file("Kernels.cl", NULL, context, selected_device, &err))
        goto error;

    queue = clCreateCommandQueue(context, selected_device, 0, &err);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return EXIT_SUCCESS;
error:
    if (queue)
        clReleaseCommandQueue(queue);
    if (context)
        clReleaseContext(context);
    return EXIT_FAILURE;
}
