/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * APIC Wave Simulation Example (CPU)
 *
 * This program demonstrates how to:
 * 1. Load a pre-captured graph (.wrp file) using Warp's APIC API
 * 2. Execute the graph on the CPU with dynamic input parameters
 * 3. Visualize the results using GLFW/OpenGL
 *
 * Unlike the CUDA version (02_apic_visualization), this example:
 * - Does not require a GPU or CUDA toolkit
 * - Links only against warp.dll/libwarp.so (plus warp-clang for CPU JIT)
 * - Uses host memory exclusively
 * - Calls wp_apic_cpu_replay_graph() instead of cudaGraphLaunch()
 *
 * See README.md for build instructions.
 */

// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "apic.h"  // APIC graph loading and execution
#include "warp.h"  // Warp C API

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#endif
// clang-format on

// Simulation constants
constexpr int GRID_WIDTH = 128;
constexpr int GRID_HEIGHT = 128;
constexpr float GRID_SCALE = 0.1f;

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
void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        g_mouse_down = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_right_mouse_down = (action == GLFW_PRESS);
        if (action == GLFW_PRESS)
            glfwGetCursorPos(window, &g_last_mouse_x, &g_last_mouse_y);
    }
}

void cursor_pos_callback(GLFWwindow* /*window*/, double x, double y)
{
    g_mouse_x = x;
    g_mouse_y = y;
    if (g_right_mouse_down) {
        g_camera_angle_y += static_cast<float>(x - g_last_mouse_x) * 0.5f;
        g_camera_angle_x += static_cast<float>(y - g_last_mouse_y) * 0.5f;
        g_camera_angle_x = fmaxf(-89.0f, fminf(89.0f, g_camera_angle_x));
        g_last_mouse_x = x;
        g_last_mouse_y = y;
    }
}

void scroll_callback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset)
{
    g_camera_distance -= static_cast<float>(yoffset) * 2.0f;
    g_camera_distance = fmaxf(5.0f, fminf(50.0f, g_camera_distance));
}

void framebuffer_size_callback(GLFWwindow* /*window*/, int width, int height)
{
    g_window_width = width;
    g_window_height = height;
    glViewport(0, 0, width, height);
}

// Matrix utilities
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
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            tmp[j * 4 + i]
                = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
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
    vec3 dx = dFdx(world_pos);
    vec3 dy = dFdy(world_pos);
    vec3 normal = normalize(cross(dx, dy));
    float diffuse = max(dot(normal, light_dir), 0.0);
    vec3 color = vec3(0.2, 0.4, 0.8);
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

// CPU version of vertex update (no CUDA kernel needed)
void update_vertices(float* vertices, const float* heights, int width, int height, float scale)
{
    for (int tid = 0; tid < width * height; tid++) {
        int x = tid % width;
        int y = tid / width;
        vertices[tid * 3 + 0] = x * scale;
        vertices[tid * 3 + 1] = heights[tid];
        vertices[tid * 3 + 2] = y * scale;
    }
}

// ---------------------------------------------------------------------------
// CPU module loading via warp-clang
// ---------------------------------------------------------------------------

// Function pointers loaded from warp-clang shared library
typedef int (*wp_load_obj_fn)(const char* object_file, const char* module_name, bool use_legacy_linker);
typedef uint64_t (*wp_lookup_fn)(const char* dll_name, const char* function_name);

static wp_load_obj_fn g_wp_load_obj = nullptr;
static wp_lookup_fn g_wp_lookup = nullptr;

bool load_warp_clang()
{
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("warp-clang.dll");
    if (!lib) {
        fprintf(stderr, "Failed to load warp-clang.dll\n");
        return false;
    }
    g_wp_load_obj = (wp_load_obj_fn)GetProcAddress(lib, "wp_load_obj");
    g_wp_lookup = (wp_lookup_fn)GetProcAddress(lib, "wp_lookup");
#else
    const char* lib_name = "warp-clang.so";
#ifdef __APPLE__
    lib_name = "libwarp-clang.dylib";
#endif
    void* lib = dlopen(lib_name, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "Failed to load %s: %s\n", lib_name, dlerror());
        return false;
    }
    g_wp_load_obj = (wp_load_obj_fn)dlsym(lib, "wp_load_obj");
    g_wp_lookup = (wp_lookup_fn)dlsym(lib, "wp_lookup");
#endif
    return g_wp_load_obj && g_wp_lookup;
}

bool load_cpu_modules(APICGraph graph, const char* modules_dir)
{
    // Load all .o files from the modules directory and resolve kernel functions
    int num_kernels = wp_apic_get_num_kernels(graph);
    if (num_kernels == 0)
        return true;

    // Scan directory for .o files and load each one
    std::vector<std::string> handles;

#ifdef _WIN32
    std::string pattern = std::string(modules_dir) + "\\*.o";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "No .o files found in %s\n", modules_dir);
        return false;
    }
    do {
        std::string filename = fd.cFileName;
        std::string path = std::string(modules_dir) + "\\" + filename;
        std::string stem = filename.substr(0, filename.size() - 2);  // Remove .o
        std::string handle = "wp_apic_" + stem;

        if (g_wp_load_obj(path.c_str(), handle.c_str(), false) != 0) {
            fprintf(stderr, "Failed to load CPU module: %s\n", path.c_str());
            FindClose(hFind);
            return false;
        }
        handles.push_back(handle);
        printf("  Loaded: %s\n", filename.c_str());
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
#else
    DIR* dir = opendir(modules_dir);
    if (!dir) {
        fprintf(stderr, "No .o files found in %s\n", modules_dir);
        return false;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.size() < 3 || filename.substr(filename.size() - 2) != ".o")
            continue;
        std::string path = std::string(modules_dir) + "/" + filename;
        std::string stem = filename.substr(0, filename.size() - 2);
        std::string handle = "wp_apic_" + stem;

        if (g_wp_load_obj(path.c_str(), handle.c_str(), false) != 0) {
            fprintf(stderr, "Failed to load CPU module: %s\n", path.c_str());
            closedir(dir);
            return false;
        }
        handles.push_back(handle);
        printf("  Loaded: %s\n", filename.c_str());
    }
    closedir(dir);
#endif

    // Resolve kernel function pointers and register with graph
    for (int i = 0; i < num_kernels; i++) {
        const char* key = wp_apic_get_kernel_key(graph, i);
        if (!key)
            continue;
        const char* fwd_name = wp_apic_get_kernel_forward_name(graph, key);
        const char* bwd_name = wp_apic_get_kernel_backward_name(graph, key);
        if (!fwd_name)
            continue;

        void* fwd_fn = nullptr;
        void* bwd_fn = nullptr;
        for (const auto& h : handles) {
            uint64_t fn = g_wp_lookup(h.c_str(), fwd_name);
            if (fn) {
                fwd_fn = reinterpret_cast<void*>(fn);
                if (bwd_name)
                    bwd_fn = reinterpret_cast<void*>(g_wp_lookup(h.c_str(), bwd_name));
                break;
            }
        }

        if (fwd_fn)
            wp_apic_register_loaded_cpu_kernel(graph, key, fwd_fn, bwd_fn);
        else
            fprintf(stderr, "Warning: kernel '%s' not found in loaded modules\n", key);
    }

    return true;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    printf("=== APIC Wave Simulation Example (CPU) ===\n\n");

    const char* graph_path = "generated/wave_sim";
    bool smoke = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc)
            graph_path = argv[++i];
        else if (strcmp(argv[i], "--smoke") == 0)
            smoke = true;
    }

    // Initialize Warp runtime
    printf("Initializing Warp runtime...\n");
    wp_init(nullptr);

    // Load warp-clang for CPU JIT
    printf("Loading warp-clang...\n");
    if (!load_warp_clang()) {
        fprintf(stderr, "warp-clang is required for CPU graph execution\n");
        return 1;
    }

    // Load APIC graph for CPU device (device_type=1)
    printf("\nLoading APIC graph from: %s\n", graph_path);
    APICGraph graph = wp_apic_load_graph(nullptr, graph_path, 1);  // 1 = CPU device
    if (!graph) {
        fprintf(stderr, "Failed to load graph: %s\n", wp_get_error_string());
        return 1;
    }

    // Load compiled CPU modules (.o files) and register kernel function pointers
    std::string modules_dir = std::string(graph_path) + "_modules";
    printf("Loading CPU modules from: %s\n", modules_dir.c_str());
    if (!load_cpu_modules(graph, modules_dir.c_str())) {
        fprintf(stderr, "Failed to load CPU modules\n");
        wp_apic_destroy_graph(graph);
        return 1;
    }

    // Query parameters
    int n_params = wp_apic_get_num_params(graph);
    printf("Graph parameters (%d):\n", n_params);
    for (int i = 0; i < n_params; i++) {
        const char* name = wp_apic_get_param_name(graph, i);
        size_t size = wp_apic_get_param_size(graph, name);
        printf("  [%d] %s: %zu bytes\n", i, name, size);
    }

    if (smoke) {
        const int kSmokeIterations = 10;
        for (int i = 0; i < kSmokeIterations; i++) {
            if (!wp_apic_cpu_replay_graph(graph)) {
                fprintf(stderr, "CPU graph replay failed at iteration %d\n", i);
                wp_apic_destroy_graph(graph);
                return 1;
            }
        }
        printf("smoke OK (%d replay iterations)\n", kSmokeIterations);
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

    GLFWwindow* window
        = glfwCreateWindow(g_window_width, g_window_height, "APIC Wave Simulation (CPU)", nullptr, nullptr);
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

    if (!gladLoadGL(glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwTerminate();
        return 1;
    }

    GLuint program = create_shader_program();
    GLint mvp_loc = glGetUniformLocation(program, "mvp");
    GLint light_loc = glGetUniformLocation(program, "light_dir");

    // Create mesh
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

    // Allocate host arrays (double-buffered heights + mouse position)
    size_t heights_size = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
    std::vector<float> heights0(GRID_WIDTH * GRID_HEIGHT, 0.0f);
    std::vector<float> heights1(GRID_WIDTH * GRID_HEIGHT, 0.0f);
    float mouse_pos[2] = { -1000.0f, -1000.0f };
    int current_buffer = 0;

    printf("\nStarting simulation loop...\n");
    printf("  Left-click to create waves\n");
    printf("  Right-drag to rotate camera\n");
    printf("  Scroll to zoom\n\n");

    glEnable(GL_DEPTH_TEST);

    int frame = 0;
    double last_time = glfwGetTime();
    int fps_counter = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Compute mouse position in grid coordinates
        mouse_pos[0] = -1000.0f;
        mouse_pos[1] = -1000.0f;
        if (g_mouse_down) {
            mouse_pos[0] = static_cast<float>(g_mouse_x / g_window_width) * GRID_WIDTH;
            mouse_pos[1] = static_cast<float>(g_mouse_y / g_window_height) * GRID_HEIGHT;
        }

        // Set graph input parameters (host memory — just memcpy)
        float* h_current = (current_buffer == 0) ? heights0.data() : heights1.data();
        float* h_prev = (current_buffer == 0) ? heights1.data() : heights0.data();

        wp_apic_set_param(graph, "heights", h_current, heights_size);
        wp_apic_set_param(graph, "heights_prev", h_prev, heights_size);
        wp_apic_set_param(graph, "mouse_pos", mouse_pos, sizeof(mouse_pos));

        // Execute the graph on CPU
        if (!wp_apic_cpu_replay_graph(graph)) {
            fprintf(stderr, "CPU graph replay failed\n");
            break;
        }

        // Get output heights
        float* h_out = (current_buffer == 0) ? heights1.data() : heights0.data();
        float* h_prev_out = (current_buffer == 0) ? heights0.data() : heights1.data();
        wp_apic_get_param(graph, "heights_out", h_out, heights_size);
        wp_apic_get_param(graph, "heights_prev_out", h_prev_out, heights_size);
        current_buffer = 1 - current_buffer;

        // Update vertex positions from heights (CPU — simple loop)
        float* h_display = (current_buffer == 0) ? heights0.data() : heights1.data();
        update_vertices(vertices.data(), h_display, GRID_WIDTH, GRID_HEIGHT, GRID_SCALE);

        // Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());

        // Camera
        float cx = GRID_WIDTH * GRID_SCALE * 0.5f;
        float cy = 0.0f;
        float cz = GRID_HEIGHT * GRID_SCALE * 0.5f;
        float ax = g_camera_angle_x * 3.14159f / 180.0f;
        float ay = g_camera_angle_y * 3.14159f / 180.0f;
        float eye_x = cx + g_camera_distance * cosf(ax) * sinf(ay);
        float eye_y = cy + g_camera_distance * sinf(ax);
        float eye_z = cz + g_camera_distance * cosf(ax) * cosf(ay);

        float proj[16], view[16], mvp[16];
        mat4_perspective(proj, 60.0f * 3.14159f / 180.0f, (float)g_window_width / g_window_height, 0.1f, 100.0f);
        mat4_look_at(view, eye_x, eye_y, eye_z, cx, cy, cz, 0, 1, 0);
        mat4_multiply(mvp, proj, view);

        // Render
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program);
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp);
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
            snprintf(title, sizeof(title), "APIC Wave Simulation (CPU) - %d FPS", fps_counter);
            glfwSetWindowTitle(window, title);
            fps_counter = 0;
            last_time = current_time;
        }
        frame++;
    }

    // Cleanup
    printf("\nCleaning up...\n");
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteProgram(program);
    wp_apic_destroy_graph(graph);
    glfwDestroyWindow(window);
    glfwTerminate();

    printf("Done.\n");
    return 0;
}
