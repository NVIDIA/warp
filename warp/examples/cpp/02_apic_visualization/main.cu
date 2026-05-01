/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * APIC Wave Simulation Example
 *
 * This program demonstrates how to:
 * 1. Load a pre-captured CUDA graph (.wrp file) using Warp's APIC API
 * 2. Execute the graph with dynamic input parameters
 * 3. Visualize the results using GLFW/OpenGL
 *
 * The wave simulation graph is captured by the Python script (capture_wave.py)
 * and contains multiple kernel launches for a full simulation frame.
 *
 * Key APIC concepts demonstrated:
 * - Graph contains 17 kernel launches (1 displacement + 16 wave solve)
 * - C++ issues ONE cudaGraphLaunch() per frame
 * - Mouse position is passed as a dynamic parameter
 * - No graph rebuilding needed - structure is fixed, only data changes
 *
 * See README.md for build instructions.
 */

// clang-format off
// Include order matters: GLAD must come before other GL headers,
// and aot.h (CUDA) must come after GLAD to avoid type conflicts
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "aot.h"  // Warp AOT utilities (includes CUDA)
#include "warp.h" // Warp C API
#include "apic.h" // APIC graph loading and execution

#include <cmath>
#include <cstdio>
#include <vector>
// clang-format on

// Simulation constants
constexpr int GRID_WIDTH = 128;
constexpr int GRID_HEIGHT = 128;
constexpr float GRID_SCALE = 0.1f;  // World-space size per grid cell

// Window and mouse state
static int g_window_width = 800;
static int g_window_height = 600;
static double g_mouse_x = 0;
static double g_mouse_y = 0;
static bool g_mouse_down = false;
static float g_camera_distance = 20.0f;
static float g_camera_angle_x = 45.0f;
static float g_camera_angle_y = 45.0f;
static double g_last_mouse_x = 0;
static double g_last_mouse_y = 0;
static bool g_right_mouse_down = false;

// GLFW callbacks
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse_down = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_right_mouse_down = (action == GLFW_PRESS);
        if (action == GLFW_PRESS) {
            glfwGetCursorPos(window, &g_last_mouse_x, &g_last_mouse_y);
        }
    }
}

void cursor_pos_callback(GLFWwindow* window, double x, double y)
{
    g_mouse_x = x;
    g_mouse_y = y;

    if (g_right_mouse_down) {
        double dx = x - g_last_mouse_x;
        double dy = y - g_last_mouse_y;
        g_camera_angle_y += (float)dx * 0.5f;
        g_camera_angle_x += (float)dy * 0.5f;
        g_camera_angle_x = fmaxf(-89.0f, fminf(89.0f, g_camera_angle_x));
        g_last_mouse_x = x;
        g_last_mouse_y = y;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    g_camera_distance -= (float)yoffset * 2.0f;
    g_camera_distance = fmaxf(5.0f, fminf(50.0f, g_camera_distance));
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    g_window_width = width;
    g_window_height = height;
    glViewport(0, 0, width, height);
}

// Simple matrix utilities
void mat4_identity(float* m)
{
    for (int i = 0; i < 16; i++)
        m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

void mat4_perspective(float* m, float fov, float aspect, float z_near, float z_far)
{
    float f = 1.0f / tanf(fov * 0.5f);
    mat4_identity(m);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (z_far + z_near) / (z_near - z_far);
    m[11] = -1.0f;
    m[14] = (2.0f * z_far * z_near) / (z_near - z_far);
    m[15] = 0.0f;
}

void mat4_look_at(float* m, float ex, float ey, float ez, float cx, float cy, float cz, float ux, float uy, float uz)
{
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float len = sqrtf(fx * fx + fy * fy + fz * fz);
    fx /= len;
    fy /= len;
    fz /= len;

    float sx = fy * uz - fz * uy;
    float sy = fz * ux - fx * uz;
    float sz = fx * uy - fy * ux;
    len = sqrtf(sx * sx + sy * sy + sz * sz);
    sx /= len;
    sy /= len;
    sz /= len;

    float ux2 = sy * fz - sz * fy;
    float uy2 = sz * fx - sx * fz;
    float uz2 = sx * fy - sy * fx;

    mat4_identity(m);
    m[0] = sx;
    m[4] = sy;
    m[8] = sz;
    m[1] = ux2;
    m[5] = uy2;
    m[9] = uz2;
    m[2] = -fx;
    m[6] = -fy;
    m[10] = -fz;
    m[12] = -(sx * ex + sy * ey + sz * ez);
    m[13] = -(ux2 * ex + uy2 * ey + uz2 * ez);
    m[14] = (fx * ex + fy * ey + fz * ez);
}

void mat4_multiply(float* result, const float* a, const float* b)
{
    float tmp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[j * 4 + i]
                = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
        }
    }
    for (int i = 0; i < 16; i++)
        result[i] = tmp[i];
}

// Shaders
const char* vertex_shader_src = R"(
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 mvp;
out vec3 world_pos;
void main() {
    world_pos = position;
    gl_Position = mvp * vec4(position, 1.0);
}
)";

const char* fragment_shader_src = R"(
#version 330 core
in vec3 world_pos;
uniform vec3 light_dir;
out vec4 frag_color;
void main() {
    // Compute normal from derivatives
    vec3 dx = dFdx(world_pos);
    vec3 dy = dFdy(world_pos);
    vec3 normal = normalize(cross(dx, dy));

    // Simple diffuse + ambient lighting
    float diffuse = max(dot(normal, light_dir), 0.0);
    vec3 color = vec3(0.2, 0.4, 0.8);  // Blue water color
    vec3 ambient = color * 0.3;
    vec3 lit = color * diffuse * 0.7 + ambient;
    frag_color = vec4(lit, 1.0);
}
)";

GLuint compile_shader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        fprintf(stderr, "Shader compilation failed: %s\n", log);
    }
    return shader;
}

GLuint create_shader_program()
{
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        fprintf(stderr, "Program linking failed: %s\n", log);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

// CUDA kernel to update mesh vertices from height field
__global__ void update_vertices(float* vertices, const float* heights, int width, int height, float scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= width * height)
        return;

    int x = tid % width;
    int y = tid / width;

    // Update Y coordinate (height), keep X and Z
    vertices[tid * 3 + 0] = x * scale;
    vertices[tid * 3 + 1] = heights[tid];
    vertices[tid * 3 + 2] = y * scale;
}

int main(int argc, char** argv)
{
    printf("=== APIC Wave Simulation Example ===\n\n");

    // Parse command line
    const char* graph_path = "generated/wave_sim";
    bool smoke = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
            graph_path = argv[++i];
        } else if (strcmp(argv[i], "--smoke") == 0) {
            smoke = true;
        }
    }

    // Initialize CUDA
    printf("Initializing CUDA...\n");
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Using device: %s\n", device_name);

    CUcontext context;
    CHECK_CU(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_CU(cuCtxSetCurrent(context));

    // Initialize Warp runtime (wp_init returns 0 on success)
    printf("Initializing Warp runtime...\n");
    wp_init(nullptr);

    // Load APIC graph
    printf("\nLoading APIC graph from: %s\n", graph_path);
    APICGraph graph = wp_apic_load_graph(context, graph_path, 0);  // 0 = CUDA device
    if (!graph) {
        fprintf(stderr, "Failed to load graph: %s\n", wp_get_error_string());
        return 1;
    }

    // Query and print graph parameters
    int n_params = wp_apic_get_num_params(graph);
    printf("Graph parameters (%d):\n", n_params);
    for (int i = 0; i < n_params; i++) {
        const char* name = wp_apic_get_param_name(graph, i);
        size_t size = wp_apic_get_param_size(graph, name);
        printf("  [%d] %s: %zu bytes\n", i, name, size);
    }

    if (smoke) {
        cudaGraphExec_t exec = (cudaGraphExec_t)wp_apic_get_cuda_graph_exec(graph);
        if (!exec) {
            fprintf(stderr, "Failed to get graph executable: %s\n", wp_get_error_string());
            wp_apic_destroy_graph(graph);
            return 1;
        }
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        const int kSmokeIterations = 10;
        for (int i = 0; i < kSmokeIterations; i++) {
            CHECK_CUDA(cudaGraphLaunch(exec, stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        printf("smoke OK (%d graph launches)\n", kSmokeIterations);
        wp_apic_destroy_graph(graph);
        return 0;
    }

    // Initialize GLFW
    printf("\nInitializing GLFW/OpenGL...\n");
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(g_window_width, g_window_height, "APIC Wave Simulation", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Load OpenGL functions using GLAD
    if (!gladLoadGL(glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwTerminate();
        return 1;
    }

    // Create shader program
    GLuint program = create_shader_program();
    GLint mvp_loc = glGetUniformLocation(program, "mvp");
    GLint light_loc = glGetUniformLocation(program, "light_dir");

    // Create mesh (grid of vertices)
    printf("Creating mesh (%dx%d)...\n", GRID_WIDTH, GRID_HEIGHT);
    std::vector<float> vertices(GRID_WIDTH * GRID_HEIGHT * 3);
    std::vector<unsigned int> indices;

    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int i = y * GRID_WIDTH + x;
            vertices[i * 3 + 0] = x * GRID_SCALE;
            vertices[i * 3 + 1] = 0.0f;
            vertices[i * 3 + 2] = y * GRID_SCALE;

            if (x > 0 && y > 0) {
                int i00 = (y - 1) * GRID_WIDTH + (x - 1);
                int i10 = (y - 1) * GRID_WIDTH + x;
                int i01 = y * GRID_WIDTH + (x - 1);
                int i11 = y * GRID_WIDTH + x;

                indices.push_back(i00);
                indices.push_back(i10);
                indices.push_back(i11);

                indices.push_back(i00);
                indices.push_back(i11);
                indices.push_back(i01);
            }
        }
    }

    // Create OpenGL buffers
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Allocate CUDA arrays (two height buffers for wave equation)
    size_t heights_size = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
    float* d_heights[2];  // Double buffer: [0]=current, [1]=previous
    float* d_vertices;
    float* d_mouse_pos;
    int current_buffer = 0;

    CHECK_CUDA(cudaMalloc(&d_heights[0], heights_size));
    CHECK_CUDA(cudaMalloc(&d_heights[1], heights_size));
    CHECK_CUDA(cudaMalloc(&d_vertices, vertices.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mouse_pos, 2 * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_heights[0], 0, heights_size));
    CHECK_CUDA(cudaMemset(d_heights[1], 0, heights_size));

    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Get graph executable (builds on first call)
    printf("\nBuilding CUDA graph executable...\n");
    cudaGraphExec_t exec = (cudaGraphExec_t)wp_apic_get_cuda_graph_exec(graph);
    if (!exec) {
        fprintf(stderr, "Failed to get graph executable: %s\n", wp_get_error_string());
        return 1;
    }

    printf("\nStarting simulation loop...\n");
    printf("  Left-click to create waves\n");
    printf("  Right-drag to rotate camera\n");
    printf("  Scroll to zoom\n\n");

    glEnable(GL_DEPTH_TEST);

    int frame = 0;
    double last_time = glfwGetTime();
    int fps_counter = 0;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Compute mouse position in grid coordinates
        float mouse_grid[2] = { -1000.0f, -1000.0f };  // Off-grid by default
        if (g_mouse_down) {
            // Simple mapping: screen coords to grid coords
            // (approximate - proper implementation would use ray casting)
            mouse_grid[0] = (float)(g_mouse_x / g_window_width) * GRID_WIDTH;
            mouse_grid[1] = (float)(g_mouse_y / g_window_height) * GRID_HEIGHT;
        }

        // Upload mouse position
        CHECK_CUDA(cudaMemcpyAsync(d_mouse_pos, mouse_grid, 2 * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Set graph input parameters (wave eq needs current and previous heights)
        wp_apic_set_param(graph, "heights", d_heights[current_buffer], heights_size);
        wp_apic_set_param(graph, "heights_prev", d_heights[1 - current_buffer], heights_size);
        wp_apic_set_param(graph, "mouse_pos", d_mouse_pos, 2 * sizeof(float));

        // Execute the graph (runs all 17 kernels in one launch!)
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));  // Must sync before reading results

        // Get output heights (swap buffers for next frame)
        wp_apic_get_param(graph, "heights_out", d_heights[1 - current_buffer], heights_size);
        wp_apic_get_param(graph, "heights_prev_out", d_heights[current_buffer], heights_size);
        current_buffer = 1 - current_buffer;

        // Update vertex positions from heights (current_buffer now points to heights_out)
        int block_size = 256;
        int num_blocks = (GRID_WIDTH * GRID_HEIGHT + block_size - 1) / block_size;
        update_vertices<<<num_blocks, block_size, 0, stream>>>(
            d_vertices, d_heights[current_buffer], GRID_WIDTH, GRID_HEIGHT, GRID_SCALE
        );

        // Copy vertices to host for OpenGL
        CHECK_CUDA(cudaMemcpyAsync(
            vertices.data(), d_vertices, vertices.size() * sizeof(float), cudaMemcpyDeviceToHost, stream
        ));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());

        // Compute camera position
        float cx = GRID_WIDTH * GRID_SCALE * 0.5f;
        float cy = 0.0f;
        float cz = GRID_HEIGHT * GRID_SCALE * 0.5f;

        float angle_x_rad = g_camera_angle_x * 3.14159f / 180.0f;
        float angle_y_rad = g_camera_angle_y * 3.14159f / 180.0f;

        float eye_x = cx + g_camera_distance * cosf(angle_x_rad) * sinf(angle_y_rad);
        float eye_y = cy + g_camera_distance * sinf(angle_x_rad);
        float eye_z = cz + g_camera_distance * cosf(angle_x_rad) * cosf(angle_y_rad);

        // Compute MVP matrix
        float proj[16], view[16], mvp[16];
        mat4_perspective(proj, 60.0f * 3.14159f / 180.0f, (float)g_window_width / g_window_height, 0.1f, 100.0f);
        mat4_look_at(view, eye_x, eye_y, eye_z, cx, cy, cz, 0, 1, 0);
        mat4_multiply(mvp, proj, view);

        // Render
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp);

        // Light direction (normalized)
        float lx = 0.5f, ly = 0.8f, lz = 0.3f;
        float ll = sqrtf(lx * lx + ly * ly + lz * lz);
        glUniform3f(light_loc, lx / ll, ly / ll, lz / ll);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);

        // FPS counter
        fps_counter++;
        double current_time = glfwGetTime();
        if (current_time - last_time >= 1.0) {
            char title[256];
            snprintf(title, sizeof(title), "APIC Wave Simulation - %d FPS", fps_counter);
            glfwSetWindowTitle(window, title);
            fps_counter = 0;
            last_time = current_time;
        }

        frame++;
    }

    // Cleanup
    printf("\nCleaning up...\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_heights[0]));
    CHECK_CUDA(cudaFree(d_heights[1]));
    CHECK_CUDA(cudaFree(d_vertices));
    CHECK_CUDA(cudaFree(d_mouse_pos));

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteProgram(program);

    wp_apic_destroy_graph(graph);

    glfwDestroyWindow(window);
    glfwTerminate();

    CHECK_CU(cuDevicePrimaryCtxRelease(device));

    printf("Done.\n");
    return 0;
}
