/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define DEBUG 1

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>

#include <opencl_experiments_export.h>

enum logging_msg_type
{
    LOGGING_MSG_TRACE,
    LOGGING_MSG_WARNING,
    LOGGING_MSG_ERROR
};

typedef void (*debug_print_handler)(const char *, cl_int, cl_int, const char *);

debug_print_handler g_debug_print_handler = NULL;

OPENCL_EXPERIMENTS_EXPORT
void init_debug_print_handler(debug_print_handler func)
{
    g_debug_print_handler = func;
}

#define CL_ERROR_CASE(err) case err: return #err

static const char *get_cl_error_string(cl_int err)
{
    switch (err) {
        CL_ERROR_CASE(CL_SUCCESS);
        CL_ERROR_CASE(CL_DEVICE_NOT_FOUND);
        CL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE);
        CL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE);
        CL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CL_ERROR_CASE(CL_OUT_OF_RESOURCES);
        CL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY);
        CL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        CL_ERROR_CASE(CL_MEM_COPY_OVERLAP);
        CL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH);
        CL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE);
        CL_ERROR_CASE(CL_MAP_FAILURE);
        CL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        CL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#ifdef CL_VERSION_1_2
        CL_ERROR_CASE(CL_COMPILE_PROGRAM_FAILURE);
        CL_ERROR_CASE(CL_LINKER_NOT_AVAILABLE);
        CL_ERROR_CASE(CL_LINK_PROGRAM_FAILURE);
        CL_ERROR_CASE(CL_DEVICE_PARTITION_FAILED);
        CL_ERROR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif

        CL_ERROR_CASE(CL_INVALID_VALUE);
        CL_ERROR_CASE(CL_INVALID_DEVICE_TYPE);
        CL_ERROR_CASE(CL_INVALID_PLATFORM);
        CL_ERROR_CASE(CL_INVALID_DEVICE);
        CL_ERROR_CASE(CL_INVALID_CONTEXT);
        CL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES);
        CL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE);
        CL_ERROR_CASE(CL_INVALID_HOST_PTR);
        CL_ERROR_CASE(CL_INVALID_MEM_OBJECT);
        CL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CL_ERROR_CASE(CL_INVALID_IMAGE_SIZE);
        CL_ERROR_CASE(CL_INVALID_SAMPLER);
        CL_ERROR_CASE(CL_INVALID_BINARY);
        CL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS);
        CL_ERROR_CASE(CL_INVALID_PROGRAM);
        CL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        CL_ERROR_CASE(CL_INVALID_KERNEL_NAME);
        CL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION);
        CL_ERROR_CASE(CL_INVALID_KERNEL);
        CL_ERROR_CASE(CL_INVALID_ARG_INDEX);
        CL_ERROR_CASE(CL_INVALID_ARG_VALUE);
        CL_ERROR_CASE(CL_INVALID_ARG_SIZE);
        CL_ERROR_CASE(CL_INVALID_KERNEL_ARGS);
        CL_ERROR_CASE(CL_INVALID_WORK_DIMENSION);
        CL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE);
        CL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE);
        CL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET);
        CL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST);
        CL_ERROR_CASE(CL_INVALID_EVENT);
        CL_ERROR_CASE(CL_INVALID_OPERATION);
        CL_ERROR_CASE(CL_INVALID_GL_OBJECT);
        CL_ERROR_CASE(CL_INVALID_BUFFER_SIZE);
        CL_ERROR_CASE(CL_INVALID_MIP_LEVEL);
        CL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_2
        CL_ERROR_CASE(CL_INVALID_PROPERTY);
        CL_ERROR_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
        CL_ERROR_CASE(CL_INVALID_COMPILER_OPTIONS);
        CL_ERROR_CASE(CL_INVALID_LINKER_OPTIONS);
        CL_ERROR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
        CL_ERROR_CASE(CL_INVALID_PIPE_SIZE);
        CL_ERROR_CASE(CL_INVALID_DEVICE_QUEUE);
#endif
    /* OpenCL extension errors not included */
    default: return "Unknown OpenCL error";
    }
}

static void debug_printf(const char *fmt,
                         const char *filename,
                         int line,
                         int msg_type,
                         ...)
{
    char buf[4096];
    char *heap_buf;
    size_t len;
    va_list ap;

    va_start (ap, msg_type);
    len = vsnprintf(buf, 4096, fmt, ap);
    va_end(ap);
    if (len > 4095) {
        /* Large output, allocate heap buffer */
        heap_buf = malloc(sizeof(*heap_buf) * (len + 1));
        if (!heap_buf)
            g_debug_print_handler(filename, line, msg_type, buf); /* Bail */
        else {
            va_start (ap, msg_type);
            len = vsnprintf(heap_buf, len + 1, fmt, ap);
            va_end(ap);

            g_debug_print_handler(filename, line, msg_type, heap_buf);
        }
        free(heap_buf);
    }
    else
        g_debug_print_handler(filename, line, msg_type, buf);
}

#define CHECK_CL_ERROR(err)                                             \
    do {                                                                \
        if (DEBUG && err != CL_SUCCESS) {                               \
            debug_printf("OpenCL returned %s",                          \
                         __FILE__, __LINE__, LOGGING_MSG_ERROR,         \
                         get_cl_error_string(err));                     \
            goto error;                                                 \
        }                                                               \
    } while(0)

#define CHECK_CL_ERROR_MSG(err, fmt, ...)                               \
    do {                                                                \
        if (DEBUG && err != CL_SUCCESS) {                               \
            debug_printf(fmt " (OpenCL returned %s)",                   \
                         __FILE__, __LINE__, LOGGING_MSG_ERROR,         \
                         __VA_ARGS__,   get_cl_error_string(err));      \
            goto error;                                                 \
        }                                                               \
    } while(0)

#define CHECK_ALLOCATION(ptr)                                           \
    do {                                                                \
        if (!ptr) {                                                     \
            debug_printf("Memory allocation failure",                   \
                         __FILE__, __LINE__, LOGGING_MSG_ERROR);        \
            goto error;                                                 \
        }                                                               \
    } while(0)

#define TRACE(fmt, ...)                                                 \
    do {                                                                \
        debug_printf(fmt, __FILE__, __LINE__, LOGGING_MSG_TRACE,        \
                     __VA_ARGS__);                                      \
    } while(0)

#define WARNING(fmt, ...)                                               \
    do {                                                                \
        debug_printf(fmt, __FILE__, __LINE__, LOGGING_MSG_WARNING,      \
                     __VA_ARGS__);                                      \
    } while(0)

#define ERROR(fmt, ...)                                                 \
    do {                                                                \
        debug_printf(fmt, __FILE__, __LINE__, LOGGING_MSG_ERROR,        \
                     __VA_ARGS__);                                      \
    } while(0)

typedef struct _opencl_plugin
{
    cl_platform_id   selected_platform;
    cl_device_id     selected_device;
    cl_context       context;
    cl_command_queue queue;
    int              num_queues;
    cl_command_queue *queues;
    cl_program       program;
    cl_kernel        voxelize_kernel;

    cl_mem           voxel_grid_buffer;
    size_t           voxel_grid_buffer_capacity;

    cl_mem           vertex_buffer;
    cl_mem           triangle_buffer;
    size_t           vertex_buffer_capacity;
    size_t           triangle_buffer_capacity;
} *opencl_plugin;

typedef struct _mesh_data
{
    float  *vertices;
    cl_int num_vertices;
    cl_int *triangles;
    cl_int num_triangles;
    size_t triangle_buffer_base_idx;
    size_t vertex_buffer_base_idx;
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
        ERROR("No GPU devices found", 0);
        goto error;
    }
    else if (*err != CL_DEVICE_NOT_FOUND) {
        CHECK_CL_ERROR(*err);
        return 0;
    }

    *err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                          device_out, NULL);
    if (*err == CL_DEVICE_NOT_FOUND) {
        ERROR("No devices found", 0);
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
        ERROR("Couldn't open file \"%s\"", filename);
        goto error;
    }

    if (fseek(file, 0L, SEEK_END)) {
        ERROR("Cannot determine file size of \"%s\"", filename);
        goto error;
    }
    program_source_size = ftell(file);
    if (fseek(file, 0L, SEEK_SET)) {
        ERROR("Cannot determine file size of \"%s\"", filename);
        goto error;
    }

    program_source = malloc(sizeof(*program_source) * (program_source_size + 1));
    CHECK_ALLOCATION(program_source);

    if (fread(program_source, 1, program_source_size, file) != program_source_size) {
        ERROR("Failed to read file \"%s\"", filename);
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

        if (options)
            ERROR("Failed to build program in file \"%s\" with options \"%s\"", filename, options);
        else
            ERROR("Failed to build program in file \"%s\"", filename);

        debug_printf("================================== BUILD LOG ===================================\n"
                     "%s", NULL, 0, LOGGING_MSG_ERROR, build_log);
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

OPENCL_EXPERIMENTS_EXPORT
cl_int opencl_plugin_create(opencl_plugin *plugin_out)
{
    cl_int err = CL_SUCCESS;
    opencl_plugin plugin;
    int i;
    int num_queues = 50;

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

    plugin->num_queues = num_queues;
    plugin->queues = calloc(num_queues, sizeof(cl_command_queue));
    CHECK_ALLOCATION(plugin->queues);

    for (i = 0; i < num_queues; i++) {
        plugin->queues[i] = clCreateCommandQueue(plugin->context, plugin->selected_device, 0, &err);
        CHECK_CL_ERROR(err);
    }

    plugin->voxelize_kernel = clCreateKernel(plugin->program, "voxelize", &err);
    CHECK_CL_ERROR(err);

    *plugin_out = plugin;
    return 0;
error:
    if (plugin) {
        if (plugin->voxelize_kernel)
            clReleaseKernel(plugin->voxelize_kernel);
        if (plugin->queue)
            clReleaseCommandQueue(plugin->queue);
        if (plugin->queues) {
            for (i = 0; i < num_queues; i++) {
                if (plugin->queues[i])
                    clReleaseCommandQueue(plugin->queues[i]);
            }
            free(plugin->queues);
        }
        if (plugin->context)
            clReleaseContext(plugin->context);
        free(plugin);
    }
    return -1;
}

static cl_int opencl_plugin_init_voxel_buffer(opencl_plugin plugin,
                                              cl_int num_voxels)
{
    cl_int err;
    cl_mem new_voxel_buffer = NULL;

    assert(plugin != NULL);
    assert(num_voxels >= 0);

    if ((size_t)num_voxels > plugin->voxel_grid_buffer_capacity) {
        /* Current buffer not big enough, free old buffer first */
        if (plugin->voxel_grid_buffer) {
            clReleaseMemObject(plugin->voxel_grid_buffer);
            plugin->voxel_grid_buffer = NULL;
        }

        plugin->voxel_grid_buffer_capacity = 0;

        /* TODO: Maybe better dynamic resizing (factor = 1.5)? */
        new_voxel_buffer =
            clCreateBuffer(plugin->context, CL_MEM_WRITE_ONLY,
                           (size_t)num_voxels, NULL, &err);
        CHECK_CL_ERROR(err);

        plugin->voxel_grid_buffer_capacity = (size_t)num_voxels;
        plugin->voxel_grid_buffer = new_voxel_buffer;
        new_voxel_buffer = NULL;
    }

    return 0;
error:
    if (new_voxel_buffer)
        clReleaseMemObject(new_voxel_buffer);
    return -1;
}

static cl_int opencl_plugin_init_mesh_buffers(opencl_plugin plugin,
                                              cl_int mesh_data_count,
                                              mesh_data *mesh_data_list)
{
    cl_int err;
    cl_int i;
    cl_mem new_vertex_buffer = NULL, new_triangle_buffer = NULL;
    size_t total_num_vertices = 0, total_num_triangles = 0;

    assert(plugin != NULL);
    assert(mesh_data_count >= 0);
    assert(mesh_data_list != NULL);

    for (i = 0; i < mesh_data_count; i++) {
        total_num_vertices += mesh_data_list[i].num_vertices;
        total_num_triangles += mesh_data_list[i].num_triangles;
    }

    if (total_num_vertices > plugin->vertex_buffer_capacity) {
        /* Current buffer not big enough, free old buffer first */
        if (plugin->vertex_buffer) {
            clReleaseMemObject(plugin->vertex_buffer);
            plugin->vertex_buffer = NULL;
        }

        plugin->vertex_buffer_capacity = 0;

        /* TODO: Maybe better dynamic resizing (factor = 1.5)? */
        new_vertex_buffer =
            clCreateBuffer(plugin->context, CL_MEM_READ_ONLY,
                           sizeof(float) * 3 * total_num_vertices, NULL, &err);
        CHECK_CL_ERROR(err);

        plugin->vertex_buffer_capacity = total_num_vertices;
        plugin->vertex_buffer = new_vertex_buffer;
        new_vertex_buffer = NULL;
    }

    if (total_num_triangles > plugin->triangle_buffer_capacity) {
        /* Current buffer not big enough, free old buffer first */
        if (plugin->triangle_buffer) {
            clReleaseMemObject(plugin->triangle_buffer);
            plugin->triangle_buffer = NULL;
        }

        plugin->triangle_buffer_capacity = 0;

        /* TODO: Maybe better dynamic resizing (factor = 1.5)? */
        new_triangle_buffer =
            clCreateBuffer(plugin->context, CL_MEM_READ_ONLY,
                           sizeof(cl_int) * 3 * total_num_triangles, NULL, &err);
        CHECK_CL_ERROR(err);

        plugin->triangle_buffer_capacity = total_num_triangles;
        plugin->triangle_buffer = new_triangle_buffer;
        new_triangle_buffer = NULL;
    }

    total_num_vertices = 0;
    total_num_triangles = 0;
    for (i = 0; i < mesh_data_count; i++) {
        mesh_data *mesh_data = &mesh_data_list[i];

        err = clEnqueueWriteBuffer(
            plugin->queue, plugin->vertex_buffer, CL_FALSE,
            sizeof(float) * 3 * total_num_vertices,
            sizeof(float) * 3 * mesh_data->num_vertices, mesh_data->vertices,
            0, NULL, NULL);
        CHECK_CL_ERROR(err);

        err = clEnqueueWriteBuffer(
            plugin->queue, plugin->triangle_buffer, CL_FALSE,
            sizeof(cl_int) * 3 * total_num_triangles,
            sizeof(cl_int) * 3 * mesh_data->num_triangles, mesh_data->triangles,
            0, NULL, NULL);
        CHECK_CL_ERROR(err);

        total_num_vertices += mesh_data_list[i].num_vertices;
        total_num_triangles += mesh_data_list[i].num_triangles;
    }

    /* Wait for all buffer writes to finish, TODO: investigate this further */
    err = clFinish(plugin->queue);
    CHECK_CL_ERROR(err);

    return 0;
error:
    if (new_vertex_buffer)
        clReleaseMemObject(new_vertex_buffer);
    if (new_triangle_buffer)
        clReleaseMemObject(new_triangle_buffer);
    return -1;
}

OPENCL_EXPERIMENTS_EXPORT
cl_int opencl_plugin_voxelize_meshes(opencl_plugin plugin,
                                     float inv_element_size,
                                     float corner_x,
                                     float corner_y,
                                     float corner_z,
                                     cl_int x_cell_length,
                                     cl_int y_cell_length,
                                     cl_int z_cell_length,
                                     cl_int mesh_data_count,
                                     mesh_data *mesh_data_list,
                                     cl_uchar *voxel_grid_out)
{
    cl_int err = CL_SUCCESS;
    cl_int i;
    cl_int next_row_offset, next_slice_offset;
    size_t local_work_size;
    cl_int num_voxels;

    clock_t t1;
    clock_t t2;
    clock_t t3;

    assert(plugin != NULL);
    assert(inv_element_size >= 0);
    assert(x_cell_length >= 0);
    assert(y_cell_length >= 0);
    assert(z_cell_length >= 0);
    assert(mesh_data_count >= 0);
    assert(mesh_data_list != NULL);

    t1 = clock();

    /* (Re-)allocate buffer for voxel grid */
    num_voxels = x_cell_length * y_cell_length * z_cell_length;
    if (opencl_plugin_init_voxel_buffer(plugin, num_voxels))
        goto error;

    /* (Re-)allocate buffers for mesh data */
    if (opencl_plugin_init_mesh_buffers(plugin, mesh_data_count, mesh_data_list))
        goto error;

    err = clGetKernelWorkGroupInfo(
        plugin->voxelize_kernel, plugin->selected_device,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local_work_size, NULL);
    CHECK_CL_ERROR(err);

    if (enqueue_zero_buffer(plugin->queue, plugin->voxel_grid_buffer, plugin->voxel_grid_buffer_capacity,
                            0, NULL, NULL, &err))
        goto error;

    err = clFinish(plugin->queue);
    CHECK_CL_ERROR(err);

    t1 = clock() - t1;
    t2 = clock();

    next_row_offset = x_cell_length;
    next_slice_offset = x_cell_length * y_cell_length;

    err |= clSetKernelArg(plugin->voxelize_kernel, 0, sizeof(cl_mem), &plugin->voxel_grid_buffer);
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
        cl_uint vertex_buffer_base_idx = mesh_data_list[i].vertex_buffer_base_idx;
        cl_uint triangle_buffer_base_idx = mesh_data_list[i].triangle_buffer_base_idx;
        err |= clSetKernelArg(plugin->voxelize_kernel, 10, sizeof(cl_mem), &plugin->vertex_buffer);
        err |= clSetKernelArg(plugin->voxelize_kernel, 11, sizeof(cl_mem), &plugin->triangle_buffer);
        err |= clSetKernelArg(plugin->voxelize_kernel, 12, sizeof(cl_int), &mesh_data_list[i].num_triangles);
        err |= clSetKernelArg(plugin->voxelize_kernel, 13, sizeof(cl_uint), &vertex_buffer_base_idx);
        err |= clSetKernelArg(plugin->voxelize_kernel, 14, sizeof(cl_uint), &triangle_buffer_base_idx);
        CHECK_CL_ERROR(err);

        /* As per the OpenCL spec, global_work_size must divide evenly by
         * local_work_size */
        global_work_size = mesh_data_list[i].num_triangles / local_work_size;
        global_work_size *= local_work_size;
        if (global_work_size < (size_t)mesh_data_list[i].num_triangles)
            global_work_size += local_work_size;

        err = clEnqueueNDRangeKernel(
            plugin->queues[i % plugin->num_queues], plugin->voxelize_kernel, 1, NULL, &global_work_size,
            &local_work_size, 0, NULL, NULL);
        CHECK_CL_ERROR_MSG(err, "clEnqueueNDRangeKernel failed on mesh %d/%d",
                           i + 1, mesh_data_count);

        err = clFinish(plugin->queue);
        CHECK_CL_ERROR_MSG(err, "clFinish failed on mesh %d/%d",
                           i + 1, mesh_data_count);
    }

    err = clFinish(plugin->queue);
    CHECK_CL_ERROR(err);

    for (i = 0; i < plugin->num_queues; i++) {
        err = clFinish(plugin->queues[i]);
        CHECK_CL_ERROR(err);
    }

    t2 = clock() - t2;
    t3 = clock();

    err = clEnqueueReadBuffer(
        plugin->queue, plugin->voxel_grid_buffer, CL_TRUE, 0,
        num_voxels, voxel_grid_out, 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    t3 = clock() - t3;

    TRACE("Clock T1: %f", ((float)t1 * 1000.0f) / CLOCKS_PER_SEC);
    TRACE("Clock T2: %f", ((float)t2 * 1000.0f) / CLOCKS_PER_SEC);
    TRACE("Clock T3: %f", ((float)t3 * 1000.0f) / CLOCKS_PER_SEC);
    return 0;
error:
    return -1;
}

OPENCL_EXPERIMENTS_EXPORT
void opencl_plugin_destroy(opencl_plugin plugin)
{
    if (!plugin) return;

    if (plugin->voxelize_kernel)
        clReleaseKernel(plugin->voxelize_kernel);
    if (plugin->queue)
        clReleaseCommandQueue(plugin->queue);
    if (plugin->context)
        clReleaseContext(plugin->context);
    if (plugin->voxel_grid_buffer)
        clReleaseMemObject(plugin->voxel_grid_buffer);
    if (plugin->vertex_buffer)
        clReleaseMemObject(plugin->vertex_buffer);
    if (plugin->triangle_buffer)
        clReleaseMemObject(plugin->triangle_buffer);

    free(plugin);
}
