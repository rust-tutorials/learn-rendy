# Introduction

## What `rendy` does

Okay, so if `rendy` doesn't hide all the complexity of `hal` from the user, then
what is it good for after all? Well, as I said, it provides many conveniences
and helpers, all of them based around the central concept of `rendy`, the render
graph (also sometimes called a 'frame graph' in literature). The render graph is
the backbone of rendy, and its purpose is to manage the final *ordering*
(schedule) of rendering commands for a frame, as well as manage the *transitive
resources* needed to complete those rendering commands. The render graph should
allow the user to build this schedule in a manner which easy to reason about and
composable from the simplest to the most complex of renderers.

### Scheduling

One of the hardest parts of using Vulkan/DX12/Metal (and therefore `gfx-hal`) is
properly scheduling commands on the gpu and synchronizing resource usage. This
problem is compounded exponentially as you add more and more dependencies
between different pipelines, render passes, etc. as happens in a modern
rendering pipeline, but even for a relatively simple renderer, it can stil be a
source of much pain and debugging. And, when synchronization is over-done, it
can easily cause a Vulkan or DX12 application to perform worse than an OpenGL or
DX11 application, because graphics drivers for those APIs have gotten rather
good at figuring out scheduling. However, we have an advantage that those
graphics drivers do not: context about exactly how our application wants to use
specific resources and when. By building commands into the nodes of a graph and
defining dependencies between those nodes, `rendy` is able to automatically
create a (hopefully) optimal schdule of submissions and synchronization
commands, which is exactly what we want when using an api like `hal`.

### Transient resources

In order to build that schedule, `rendy` manages what are called 'transient'
resources. These are resources which are used within the process of rendering a
frame, but which are not based directly on inputs from the application (like,
for example, a texture or a model's vertex buffers) and which will not be
directly displayed to the user (say, a framebuffer containing a swapchain
image). Examples of transient resources would be shadow maps, an HDR image to be
tonemapped, a storage buffer which will be calculated in a compute shader then
used as information during a graphics draw call, etc.

### Memory allocation and non-transient resource management

`gfx-hal` provides only the raw memory allocation and resource creation
APIs that Vulkan does, which are not meant to be used directly, but rather as
the basis to build a memory allocator on top of. `rendy` provides a memory
allocator called `Heaps` as well as a resource management helper in the form
of its `Factory`. These build on top of the raw `hal`/Vulkan APIs and provide
a full memory allocator which is able to sub-allocate resources etc. as well as
convenience functions for creating buffers and images using higher level, more
expressive usage directives than those of `hal` which makes managing permanent
(non-transient) resources like textures, vertex buffers, uniform buffers, etc.
much easier and more ergonomic.

## Okay, on with it!

Alright, so now that you know what `rendy` provides, let's get on to actually using
it. First we'll go with the classic triangle example and then we'll immediately jump
to something more complex and pick up right where `learn-gfx-hal` left off with
drawing a set of instanced cubes.