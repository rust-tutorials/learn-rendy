#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(early_fragment_tests) in;

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 color;

struct Light {
    vec3 pos;
    float pad;
    float intencity;
};

layout(std140, set = 0, binding = 0) uniform Args {
    mat4 proj;
    mat4 view;
};

layout(set = 0, binding = 1) uniform texture2D tex;
layout(set = 0, binding = 2) uniform sampler samp;

void main() {
    color = texture(sampler2D(tex, samp), frag_uv);
}