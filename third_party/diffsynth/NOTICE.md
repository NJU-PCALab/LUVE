# Inference backend

This directory contains a selected inference subset of the DiffSynth-Studio
fork supplied by the maintainer, isolated under the `luve_diffsynth`
namespace. It contains no model weights, training data, caches, original
inference entry points, or server account information.

`MODIFICATIONS.md` records the local portability and integration changes. The package includes
Wan component definitions, prompt processing, a flow scheduler, VRAM
management, and the supplied 3D LIIF/VRT VLU implementation.

The source fork's accompanying `LICENSE.DiffSynth-Studio` file is retained
verbatim, together with the copyright and attribution headers present in
the supplied source files. This subset does not assign a new project or
model-weight license. The maintainer will finalize those release terms.
