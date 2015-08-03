/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define DEBUG 1

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

static void debug_printf(const char *fmt,
                         const char *filename,
                         int line,
                         ...)
{
    va_list args;

    fprintf(stderr, "%s:%d: ", filename, line);
    va_start (args, line);
    vfprintf (stderr, fmt, args);
    va_end (args);

    fprintf(stderr, "\n");
}

#define CHECK_CL_ERROR(err)                                             \
    do {                                                                \
        if (DEBUG && err != CL_SUCCESS) {                               \
            debug_printf("Error: OpenCL returned error %d",             \
                         __FILE__, __LINE__, err);                      \
            goto error;                                                 \
        }                                                               \
    } while(1)

#define CHECK_CL_ERROR_MSG(err, fmt, ...)                               \
    do {                                                                \
        if (DEBUG && err != CL_SUCCESS) {                               \
            debug_printf("Error: " fmt " (OpenCL returned error %d)",   \
                         __FILE__, __LINE__, __VA_ARGS__, err);         \
            goto error;                                                 \
        }                                                               \
    } while(1)

#define CHECK_ALLOCATION(ptr)                                           \
    do {                                                                \
        if (!ptr) {                                                     \
            debug_printf("Error: Memory allocation failure",            \
                         __FILE__, __LINE__);                           \
            goto error;                                                 \
        }                                                               \
    } while(1)

typedef struct _opencl_plugin
{
    cl_platform_id   selected_platform;
    cl_device_id     selected_device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        voxelize_kernel;

    cl_mem           voxel_grid_buffer;
    size_t           voxel_grid_buffer_size;

    cl_mem           vertex_buffer;
    cl_mem           triangles_buffer;
    size_t           mesh_buffers_max_triangles;
} *opencl_plugin;

typedef struct _mesh_data
{
    float  *verticies;
    cl_int *triangles;
    cl_int num_triangles;
    size_t buffer_offset;
} mesh_data;

static int get_desired_platform(const char *substr,
                                cl_platform_id *platform_id_out,
                                cl_int *err)
{
    cl_int _err = CL_SUCCESS;
    cl_uint i, num_platforms;
    cl_platform_id *platform_ids = NULL;
    char *platform_name = NULL;

    assert(platform_id_out != NULL);

    if (!err) err = &_err;

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

        if (strstr(platform_name, substr))
            break;
    }

    if (i < num_platforms)
        *platform_id_out = platform_ids[i];
    else
        goto error; /* No platforms found */

    free(platform_ids);
    free(platform_name);
    return 0;
error:
    free(platform_ids);
    free(platform_name);
    return -1;
}

static int get_gpu_device_id(cl_platform_id platform_id,
                             cl_device_id *device_out,
                             cl_bool fallback,
                             cl_int *err)
{
    cl_int _err = CL_SUCCESS;

    assert(device_out != NULL);

    if (!err) err = &_err;

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

static int create_context(cl_platform_id platform,
                          cl_device_id device,
                          cl_context *context_out,
                          cl_int *err)
{
    cl_int _err = CL_SUCCESS;
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    assert(context_out != NULL);

    if (!err) err = &_err;

    *context_out = clCreateContext(context_properties, 1, &device, NULL, NULL,
                                   err);
    CHECK_CL_ERROR(*err);

    return 0;
error:
    return -1;
}

static int build_program_from_file(const char *filename,
                                   const char *options,
                                   cl_context context,
                                   cl_device_id device,
                                   cl_program *program_out,
                                   cl_int *err)
{
    cl_int _err;
    FILE *file;
    char *program_source = NULL;
    size_t program_source_size;
    cl_program program = NULL;
    char *build_log = NULL;

    assert(filename != NULL);
    assert(program_out != NULL);

    if (!err) err = &_err;

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
        if (options)
            fprintf(stderr, "       with options \"%s\"\n\n", options);
        fprintf(stderr, "================================== BUILD LOG ===================================\n\n");
        fprintf(stderr, "%s", build_log);
        goto error;
    }
    CHECK_CL_ERROR(*err);

    *program_out = program;
    return 0;
error:
    free(build_log);
    if (program)
        clReleaseProgram(program);
    *program_out = NULL;
    free(program_source);
    return -1;
}

static int enqueue_zero_buffer(cl_command_queue queue,
                               cl_mem buffer,
                               size_t size,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event,
                               cl_int *err)
{
    cl_int _err;
    cl_uchar c = 0;

    if (!err) err = &_err;

    *err = clEnqueueFillBuffer(queue, (cl_mem)buffer, &c, sizeof(c), 0,
                               size, num_events_in_wait_list, event_wait_list,
                               event);
    CHECK_CL_ERROR(*err);

    return 0;
error:
    return -1;
}

cl_int opencl_plugin_create(opencl_plugin *plugin_out)
{
    cl_int err = CL_SUCCESS;
    opencl_plugin plugin;

    assert(plugin_out != NULL);

    plugin = calloc(1, sizeof(*plugin));
    CHECK_ALLOCATION(plugin);

    if (get_desired_platform("NVIDIA", &plugin->selected_platform, &err))
        goto error;

    if (get_gpu_device_id(plugin->selected_platform, &plugin->selected_device,
                          CL_TRUE, &err))
        goto error;

    if (create_context(plugin->selected_platform, plugin->selected_device,
                       &plugin->context, &err))
        goto error;

    if (build_program_from_file("program.cl", NULL, plugin->context,
                                plugin->selected_device, &plugin->program, &err))
        goto error;

    plugin->queue = clCreateCommandQueue(plugin->context, plugin->selected_device, 0, &err);
    CHECK_CL_ERROR(err);

    plugin->voxelize_kernel = clCreateKernel(plugin->program, "voxelize", &err);
    CHECK_CL_ERROR(err);

    *plugin_out = plugin;
    return 0;
error:
    if (plugin->voxelize_kernel)
        clReleaseKernel(plugin->voxelize_kernel);
    if (plugin->queue)
        clReleaseCommandQueue(plugin->queue);
    if (plugin->context)
        clReleaseContext(plugin->context);
    free(plugin);
    return -1;
}

cl_int opencl_plugin_set_num_voxels(opencl_plugin plugin,
                                    cl_int num_voxels)
{
    cl_int err;
    cl_mem buffer = NULL;

    /* TODO: Maybe do other way around? */
    buffer = clCreateBuffer(plugin->context, CL_MEM_WRITE_ONLY, (size_t)num_voxels, NULL, &err);
    CHECK_CL_ERROR(err);

    /* Make sure all commands are finished before freeing the old mem object */
    clFinish(plugin->queue);
    if (plugin->voxel_grid_buffer)
        clReleaseMemObject(plugin->voxel_grid_buffer);

    plugin->voxel_grid_buffer = buffer;
    plugin->voxel_grid_buffer_size = (size_t)num_voxels;
    return 0;
error:
    if (buffer)
        clReleaseMemObject(buffer);
    return -1;
}

static cl_int opencl_plugin_init_buffers(opencl_plugin plugin,
                                         cl_int mesh_data_count,
                                         mesh_data *mesh_data_list)
{
    cl_int err;
    cl_int i;
    cl_mem new_vertex_buffer = NULL, new_triangles_buffer = NULL;
    size_t total_num_triangles = 0;

    for (i = 0; i < mesh_data_count; i++)
        total_num_triangles += mesh_data_list[i].num_triangles;

    if (total_num_triangles <= plugin->mesh_buffers_max_triangles)
        return 0;

    /* TODO: Maybe do other way around? */
    /* TODO: Maybe better dynamic resizing (factor = 1.5)? */
    new_vertex_buffer =
        clCreateBuffer(plugin->context, CL_MEM_READ_ONLY,
                       sizeof(cl_int) * total_num_triangles, NULL, &err);
    CHECK_CL_ERROR(err);

    new_triangles_buffer =
        clCreateBuffer(plugin->context, CL_MEM_READ_ONLY,
                       sizeof(float) * 3 * total_num_triangles, NULL, &err);
    CHECK_CL_ERROR(err);

    if (plugin->vertex_buffer)
        clReleaseMemObject(plugin->vertex_buffer);
    if (plugin->triangles_buffer)
        clReleaseMemObject(plugin->triangles_buffer);

    return 0;
error:
    if (new_vertex_buffer)
        clReleaseMemObject(new_vertex_buffer);
    if (new_triangles_buffer)
        clReleaseMemObject(new_triangles_buffer);
    return -1;
}

cl_int opencl_plugin_voxelize_meshes(opencl_plugin plugin,
                                     float inv_element_size,
                                     float corner_x,
                                     float corner_y,
                                     float corner_z,
                                     cl_int x_cell_length,
                                     cl_int y_cell_length,
                                     cl_int z_cell_length,
                                     cl_int mesh_data_count,
                                     mesh_data *mesh_data_list)
{
    /* TODO: check buffer */
    cl_int err = CL_SUCCESS;
    cl_int i;
    cl_int next_row_offset, next_slice_offset;
    size_t local_work_size;

    assert(inv_element_size >= 0);
    assert(x_cell_length >= 0);
    assert(y_cell_length >= 0);
    assert(z_cell_length >= 0);
    assert(mesh_data_count >= 0);
    assert(mesh_data_list != NULL);

    if (opencl_plugin_init_buffers(plugin, mesh_data_count, mesh_data_list))
        goto error;

    err = clGetKernelWorkGroupInfo(
        plugin->voxelize_kernel, plugin->selected_device,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local_work_size, NULL);
    CHECK_CL_ERROR(err);

    if (enqueue_zero_buffer(plugin->queue, plugin->voxel_grid_buffer, plugin->voxel_grid_buffer_size,
                            0, NULL, NULL, &err))
        goto error;

    next_row_offset = x_cell_length;
    next_slice_offset = x_cell_length * y_cell_length;

    err |= clSetKernelArg(plugin->voxelize_kernel, 0, sizeof(cl_mem), &plugin->voxel_grid_buffer_size);
    err |= clSetKernelArg(plugin->voxelize_kernel, 1, sizeof(float),  &inv_element_size);
    err |= clSetKernelArg(plugin->voxelize_kernel, 2, sizeof(float),  &corner_x);
    err |= clSetKernelArg(plugin->voxelize_kernel, 3, sizeof(float),  &corner_y);
    err |= clSetKernelArg(plugin->voxelize_kernel, 4, sizeof(float),  &corner_z);
    err |= clSetKernelArg(plugin->voxelize_kernel, 5, sizeof(cl_int), &next_row_offset);
    err |= clSetKernelArg(plugin->voxelize_kernel, 6, sizeof(cl_int), &next_slice_offset);
    err |= clSetKernelArg(plugin->voxelize_kernel, 7, sizeof(cl_int), &x_cell_length);
    err |= clSetKernelArg(plugin->voxelize_kernel, 8, sizeof(cl_int), &y_cell_length);
    err |= clSetKernelArg(plugin->voxelize_kernel, 9, sizeof(cl_int), &z_cell_length);
    CHECK_CL_ERROR(err);

    for (i = 0; i < mesh_data_count; i++) {
        size_t global_work_size;
        err |= clSetKernelArg(plugin->voxelize_kernel, 10, sizeof(cl_mem), &plugin->vertex_buffer);
        err |= clSetKernelArg(plugin->voxelize_kernel, 11, sizeof(cl_mem), &plugin->triangles_buffer);
        err |= clSetKernelArg(plugin->voxelize_kernel, 12, sizeof(cl_int), &mesh_data_list[i].num_triangles);
        err |= clSetKernelArg(plugin->voxelize_kernel, 13, sizeof(size_t), &mesh_data_list[i].buffer_offset);
        CHECK_CL_ERROR(err);

        /* As per the OpenCL spec, global_work_size must divide evenly by
         * local_work_size */
        global_work_size = mesh_data_list[i].num_triangles / local_work_size;
        global_work_size *= local_work_size;
        if (global_work_size < (size_t)mesh_data_list[i].num_triangles)
            global_work_size += local_work_size;

        err = clEnqueueNDRangeKernel(plugin->queue, plugin->voxelize_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        CHECK_CL_ERROR(err);
    }

    return 0;
error:
    return -1;
}

void opencl_plugin_destroy(opencl_plugin plugin)
{
    if (!plugin) return;

    if (plugin->voxelize_kernel)
        clReleaseKernel(plugin->voxelize_kernel);
    if (plugin->queue)
        clReleaseCommandQueue(plugin->queue);
    if (plugin->context)
        clReleaseContext(plugin->context);
    free(plugin);
}
