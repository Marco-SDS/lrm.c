# lrm.c - Pure-C inference engine for LRM-family image-to-3D models.
# Fork of antirez/iris.c; see LRMengine.md for the full plan.
#
# Three peer backends, picked at build time:
#   make generic - Pure C, no deps (slow, for portability)
#   make blas    - OpenBLAS (Linux) / Accelerate (macOS), ~30x faster
#   make mps     - Apple Silicon Metal GPU (fastest, macOS arm64 only)

CC = gcc
# -I. lets sources under lrm/ include root headers (iris.h, iris_kernels.h).
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math -I.
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# =============================================================================
# Source files
# =============================================================================
# Root sources = generic infrastructure (kernels, image I/O, safetensors).
# lrm/ subdirectory = LRM-specific code (skeleton in Phase 3; populated in
# Phases 5-11). The binary is named ./lrmc to avoid the filesystem conflict
# between a regular file ./lrm and a directory ./lrm/ (project name is lrm.c,
# binary is lrmc, library is liblrmc.a).

INFRA_SRCS = iris.c iris_kernels.c iris_image.c jpeg.c iris_safetensors.c
LRM_SRCS   = lrm/lrm.c lrm/lrm_triposr.c lrm/lrm_vit_dino.c lrm/lrm_triplane_decoder.c lrm/lrm_triplane_upsample.c lrm/lrm_triplane_sample.c lrm/lrm_nerf_mlp.c lrm/lrm_density.c lrm/lrm_marching_cubes.c lrm/lrm_mesh_export.c lrm/lrm_bake_texture.c
SRCS       = $(INFRA_SRCS) $(LRM_SRCS)
OBJS       = $(SRCS:.c=.o)
MAIN       = main.c
TARGET     = lrmc
LIB        = liblrmc.a

DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test test-dino test-decoder test-upsample test-density test-density-sparse test-mc test-glb test-e2e test-e2e-tex help generic blas mps
.NOTPARALLEL: mps

# Default: show available targets
all: help

help:
	@echo "lrm.c - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make generic  - Pure C, no dependencies (slow, portable)"
	@echo "  make blas     - With BLAS acceleration"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make mps      - Apple Silicon with Metal GPU (fastest)"
endif
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test          - Build and run the kernel parity tests"
	@echo "  make test-dino     - Build and run the DINO ViT-B/16 parity test"
	@echo "  make test-decoder  - Build and run the triplane decoder parity test"
	@echo "  make test-upsample - Build and run the post-processor parity test"
	@echo "  make test-density  - Build and run the triplane-sample + NeRF MLP test"
	@echo "  make test-mc       - Build and run the marching cubes parity test"
	@echo "  make test-glb      - Build and run the GLB writer structural test"
	@echo "  make test-e2e      - End-to-end TripoSR inference at 64^3 (~50 s on Intel macOS CPU)"
	@echo "  make test-e2e-tex  - Textured end-to-end at 64^3 with 1024^2 UV atlas"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library ($(LIB))"
	@echo ""
	@echo "Phase 3 stub: ./$(TARGET) links lrm/ but the API is still empty;"
	@echo "real implementations land in Phases 5-11. See LRMengine.md."

# =============================================================================
# Backend: generic (pure C, no BLAS)
# =============================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C, no BLAS)."

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend."

# =============================================================================
# Backend: mps (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

mps: clean mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU)."

mps-build: $(SRCS:.c=.mps.o) iris_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

%.mps.o: %.c iris.h iris_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

# Embed Metal shader source as a C array (runtime compilation, no Metal
# toolchain needed at user-build time).
iris_shaders_source.h: iris_shaders.metal
	xxd -i $< > $@

iris_metal.o: iris_metal.m iris_metal.h iris_shaders_source.h
	$(CC) $(MPS_OBJCFLAGS) -c -o $@ $<

else
mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)."
	@exit 1
endif
else
mps:
	@echo "Error: MPS backend requires macOS."
	@exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c iris.h iris_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# =============================================================================
# Utilities
# =============================================================================
test:
	@echo "Building kernel parity tests..."
	@$(CC) $(CFLAGS_BASE) tests/model/test_kernels.c iris_kernels.c \
	    -lm -o /tmp/lrm_test_kernels
	@/tmp/lrm_test_kernels
	@rm -f /tmp/lrm_test_kernels

test-dino:
	@echo "Building DINO ViT-B/16 parity test..."
	@$(CC) $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	    tests/model/test_vit_dino.c lrm/lrm_vit_dino.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    -framework Accelerate -lm -o /tmp/lrm_test_dino
	@/tmp/lrm_test_dino
	@rm -f /tmp/lrm_test_dino

test-decoder:
	@echo "Building triplane decoder parity test..."
	@$(CC) $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	    tests/model/test_triplane_decoder.c lrm/lrm_triplane_decoder.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    -framework Accelerate -lm -o /tmp/lrm_test_decoder
	@/tmp/lrm_test_decoder
	@rm -f /tmp/lrm_test_decoder

test-upsample:
	@echo "Building triplane upsample parity test..."
	@$(CC) $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	    tests/model/test_triplane_upsample.c lrm/lrm_triplane_upsample.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    -framework Accelerate -lm -o /tmp/lrm_test_upsample
	@/tmp/lrm_test_upsample
	@rm -f /tmp/lrm_test_upsample

test-density:
	@echo "Building triplane-sample + NeRF MLP parity test..."
	@$(CC) $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	    tests/model/test_density_64.c \
	    lrm/lrm_triplane_sample.c lrm/lrm_nerf_mlp.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    -framework Accelerate -lm -o /tmp/lrm_test_density
	@/tmp/lrm_test_density
	@rm -f /tmp/lrm_test_density

test-density-sparse:
	@echo "Building sparse-vs-dense density parity test..."
	@$(CC) $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
	    tests/geometry/test_density_sparse.c \
	    lrm/lrm_density.c lrm/lrm_triplane_sample.c lrm/lrm_nerf_mlp.c \
	    lrm/lrm_marching_cubes.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    -framework Accelerate -lm -o /tmp/lrm_test_density_sparse
	@/tmp/lrm_test_density_sparse
	@rm -f /tmp/lrm_test_density_sparse

test-mc:
	@echo "Building marching cubes parity test..."
	@$(CC) $(CFLAGS_BASE) \
	    tests/geometry/test_marching_cubes.c lrm/lrm_marching_cubes.c iris.c \
	    -lm -o /tmp/lrm_test_mc
	@/tmp/lrm_test_mc
	@rm -f /tmp/lrm_test_mc

test-glb:
	@echo "Building GLB writer test..."
	@$(CC) $(CFLAGS_BASE) -D_DARWIN_C_SOURCE -D_GNU_SOURCE \
	    tests/geometry/test_glb.c lrm/lrm_mesh_export.c iris.c \
	    -lm -o /tmp/lrm_test_glb
	@/tmp/lrm_test_glb
	@if [ -x triposr_env/.venv/bin/python ]; then \
	    echo "round-trip via trimesh + raw glTF inspection ..."; \
	    triposr_env/.venv/bin/python tools/check_glb.py /tmp/lrm_test.glb; \
	fi
	@rm -f /tmp/lrm_test_glb /tmp/lrm_test.glb

test-e2e: blas
	@echo ""
	@echo "Running end-to-end TripoSR inference at 64^3 ..."
	@./lrmc infer triposr_env triposr_env/examples/robot.png \
	    -o /tmp/lrm_e2e.glb --mc-resolution 64
	@if [ -x triposr_env/.venv/bin/python ]; then \
	    echo ""; \
	    echo "Verifying GLB via check_glb.py:"; \
	    triposr_env/.venv/bin/python tools/check_glb.py /tmp/lrm_e2e.glb; \
	fi
	@echo ""
	@echo "PASS  end-to-end inference (/tmp/lrm_e2e.glb)"

test-e2e-tex: blas
	@echo ""
	@echo "Running textured end-to-end at 64^3 + 1024^2 atlas ..."
	@./lrmc infer triposr_env triposr_env/examples/robot.png \
	    -o /tmp/lrm_e2e_tex.glb --mc-resolution 64 \
	    --bake-texture --texture-resolution 1024
	@if [ -x triposr_env/.venv/bin/python ]; then \
	    echo ""; \
	    echo "Verifying GLB via check_glb.py:"; \
	    triposr_env/.venv/bin/python tools/check_glb.py /tmp/lrm_e2e_tex.glb; \
	fi
	@echo ""
	@echo "PASS  end-to-end textured inference (/tmp/lrm_e2e_tex.glb)"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 iris.h /usr/local/include/
	install -m 644 iris_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(SRCS:.c=.mps.o) iris_metal.o main.o main.mps.o $(TARGET) $(LIB)
	rm -f iris_shaders_source.h

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo ""
	@echo "Available backends for this platform:"
	@echo "  generic - Pure C (always available)"
ifeq ($(UNAME_S),Darwin)
	@echo "  blas    - Apple Accelerate"
ifeq ($(UNAME_M),arm64)
	@echo "  mps     - Metal GPU (recommended)"
endif
else
	@echo "  blas    - OpenBLAS (requires libopenblas-dev)"
endif

# =============================================================================
# Dependencies
# =============================================================================
iris.o: iris.c iris.h
iris_kernels.o: iris_kernels.c iris_kernels.h
iris_image.o: iris_image.c iris.h jpeg.h
iris_safetensors.o: iris_safetensors.c iris_safetensors.h
lrm/lrm.o: lrm/lrm.c lrm/lrm.h lrm/lrm_triposr.h iris.h
lrm/lrm_triposr.o: lrm/lrm_triposr.c lrm/lrm_triposr.h lrm/lrm.h iris.h iris_safetensors.h
lrm/lrm_vit_dino.o: lrm/lrm_vit_dino.c lrm/lrm_vit_dino.h iris.h iris_kernels.h iris_safetensors.h
lrm/lrm_triplane_decoder.o: lrm/lrm_triplane_decoder.c lrm/lrm_triplane_decoder.h iris.h iris_kernels.h iris_safetensors.h
lrm/lrm_triplane_upsample.o: lrm/lrm_triplane_upsample.c lrm/lrm_triplane_upsample.h iris.h iris_kernels.h iris_safetensors.h
lrm/lrm_triplane_sample.o: lrm/lrm_triplane_sample.c lrm/lrm_triplane_sample.h iris.h iris_kernels.h
lrm/lrm_nerf_mlp.o: lrm/lrm_nerf_mlp.c lrm/lrm_nerf_mlp.h iris.h iris_kernels.h iris_safetensors.h
lrm/lrm_density.o: lrm/lrm_density.c lrm/lrm_density.h lrm/lrm_triplane_sample.h lrm/lrm_nerf_mlp.h iris.h
lrm/lrm_marching_cubes.o: lrm/lrm_marching_cubes.c lrm/lrm_marching_cubes.h iris.h
lrm/lrm_mesh_export.o: lrm/lrm_mesh_export.c lrm/lrm_mesh_export.h lrm/lrm.h iris.h
lrm/lrm_bake_texture.o: lrm/lrm_bake_texture.c lrm/lrm_bake_texture.h lrm/lrm_triplane_sample.h lrm/lrm_nerf_mlp.h iris.h iris_kernels.h
main.o: main.c iris.h lrm/lrm.h lrm/lrm_triposr.h
