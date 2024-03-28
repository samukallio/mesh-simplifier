# Overview

This is an efficient implementation of the mesh simplification algorithm presented in the paper
"Surface Simplification Using Quadric Error Metrics" by Michael Garland and Paul S. Heckbert.
The algorithm iteratively contracts pairs of vertices into a single point to remove one vertex
from the mesh at a time.  The pair to be contracted is chosen using a quadric error metric that
aims to minimize the error introduced by the resulting simplification.

The simplifier supports the Wavefront .OBJ format, a simple text-based 3D mesh format.

# Building

CMake is required to generate the build files for the project. The only library dependency is GLM,
which is bundled for convenience. To generate the build files, type the following:
```
mkdir build
cd build
cmake ..
```
After the build files have been generated, build the project using the platform compiler toolkit
(e.g. Visual Studio on Windows).

# Usage

```
> simplify.exe armadillo.obj armadillo_simple.obj

=== mesh simplification complete ===

input:
    vertices = 106289
    faces = 212574
output:
    vertices = 26568
    faces = 53056
time:
    total = 376ms
    pair find = 79ms
    vertex update = 9ms
    pair contract = 173ms
    face contract = 113ms
```
