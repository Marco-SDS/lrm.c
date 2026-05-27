# lrm.c - Pure-C inference engine for LRM-family image-to-3D models (TripoSR).
# Fork of antirez/iris.c; see LRMengine.md for the full plan.
#
# Two CPU build pipelines produce the `lrmc` binary:
#   make generic   - Pure C, no deps (slow, fully portable)
#   make blas      - Accelerate (macOS) / OpenBLAS (Linux). Default, ~30x faster
#
# A Metal/GPU backend is planned (Phase 13) but the kernels are not yet
# implemented, so no GPU target is exposed here. The iris_metal*/iris_shaders
# sources are kept in the tree for that future work.
#
# Run end-to-end inference with the built binary (see `make run`):
#   make run IMAGE=img.png RES=256 TEX=0 OUT=/tmp/out.glb
#
# Validate numerics against the PyTorch reference with `make check`.

CC = gcc
# -I. lets sources under lrm/ include root headers (iris.h, iris_kernels.h).
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math -I.
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# BLAS flags, factored so both the `blas` build and the parity tests share
# them (Accelerate on macOS, OpenBLAS on Linux).
ifeq ($(UNAME_S),Darwin)
BLAS_CFLAGS  = -DUSE_BLAS -DACCELERATE_NEW_LAPACK
BLAS_LDFLAGS = -framework Accelerate
else
BLAS_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
BLAS_LDFLAGS = -lopenblas
endif

# =============================================================================
# Source files
# =============================================================================
# Root sources = generic infrastructure (kernels, image I/O, safetensors).
# lrm/ = ALL LRM-specific code. Binary is `lrmc` (avoids the ./lrm file vs
# lrm/ directory clash); static library is liblrmc.a.
INFRA_SRCS = iris.c iris_kernels.c iris_image.c jpeg.c iris_safetensors.c
LRM_SRCS   = lrm/lrm.c lrm/lrm_triposr.c lrm/lrm_vit_dino.c lrm/lrm_triplane_decoder.c lrm/lrm_triplane_upsample.c lrm/lrm_triplane_sample.c lrm/lrm_nerf_mlp.c lrm/lrm_density.c lrm/lrm_marching_cubes.c lrm/lrm_mesh_export.c lrm/lrm_bake_texture.c lrm/lrm_u2net.c
SRCS       = $(INFRA_SRCS) $(LRM_SRCS)
OBJS       = $(SRCS:.c=.o)
TARGET     = lrmc
LIB        = liblrmc.a

DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

# Parameters for `make run` (override on the command line).
MODEL  ?= triposr_env
IMAGE  ?= triposr_env/examples/robot.png
OUT    ?= /tmp/lrm_out.glb
RES    ?= 256
TEX    ?= 0
TEXRES ?= 2048
TEX_FLAGS = $(if $(filter 1 yes on true,$(TEX)),--bake-texture --texture-resolution $(TEXRES),)

.PHONY: all help generic blas run check debug lib install info clean \
        test test-dino test-decoder test-upsample test-density test-u2net \
        test-density-sparse test-mc test-glb

# Default: show available targets
all: help

help:
	@echo "lrm.c - image -> 3D (TripoSR), pure C, CPU"
	@echo ""
	@echo "Build (produces ./$(TARGET)):"
	@echo "  make blas     - CPU + BLAS (Accelerate/OpenBLAS). Default, fast."
	@echo "  make generic  - CPU, pure C, no deps. Portable, ~30x slower."
	@echo ""
	@echo "Run end-to-end inference with the built binary:"
	@echo "  make run [IMAGE=..] [RES=N] [TEX=0|1] [TEXRES=N] [OUT=..] [MODEL=..]"
	@echo "    IMAGE  input image           (default: $(IMAGE))"
	@echo "    RES    marching-cubes res     (default: $(RES); try 64/128/256/512)"
	@echo "    TEX    bake UV texture        (1 = textured, 0 = vertex colors; default 0)"
	@echo "    TEXRES texture atlas size     (default: $(TEXRES); only with TEX=1)"
	@echo "    OUT    output .glb path       (default: $(OUT))"
	@echo "    MODEL  model directory        (default: $(MODEL))"
	@echo ""
	@echo "Validate / develop:"
	@echo "  make check    - run all parity tests vs the PyTorch reference"
	@echo "  make clean    - remove build artifacts"
	@echo "  make lib      - build static library ($(LIB))"
	@echo "  make info     - show build configuration"
	@echo ""
	@echo "A Metal/GPU backend is planned (Phase 13); not yet available."

# =============================================================================
# Build pipelines (CPU)
# =============================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built ./$(TARGET) with GENERIC backend (pure C, no BLAS)."

blas: CFLAGS = $(CFLAGS_BASE) $(BLAS_CFLAGS)
blas: LDFLAGS += $(BLAS_LDFLAGS)
blas: clean $(TARGET)
	@echo ""
	@echo "Built ./$(TARGET) with BLAS backend."

$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c iris.h iris_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

lib: $(LIB)
$(LIB): $(OBJS)
	ar rcs $@ $^

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# =============================================================================
# Run end-to-end inference with the built binary
# =============================================================================
run:
	@test -x ./$(TARGET) || $(MAKE) blas
	@echo "lrmc infer: model=$(MODEL) image=$(IMAGE) res=$(RES) tex=$(TEX) out=$(OUT)"
	./$(TARGET) infer $(MODEL) $(IMAGE) -o $(OUT) --mc-resolution $(RES) $(TEX_FLAGS)
	@if [ -x triposr_env/.venv/bin/python ]; then \
	    echo ""; echo "Inspecting $(OUT):"; \
	    triposr_env/.venv/bin/python debug/debug_mesh.py $(OUT); \
	fi

# =============================================================================
# Parity tests (numerical correctness vs the PyTorch reference)
# =============================================================================
# `make check` runs the whole suite; individual targets are available for
# debugging a single stage.
check: test test-dino test-decoder test-upsample test-density test-u2net \
       test-density-sparse test-mc test-glb
	@echo ""
	@echo "PASS  all parity tests"

test:
	@echo "[kernels] LayerNorm/GELU/GEGLU/grid_sample vs NumPy ..."
	@$(CC) $(CFLAGS_BASE) tests/model/test_kernels.c iris_kernels.c \
	    -lm -o /tmp/lrm_test_kernels
	@/tmp/lrm_test_kernels
	@rm -f /tmp/lrm_test_kernels

test-dino:
	@echo "[dino] DINO ViT-B/16 forward parity ..."
	@$(CC) $(CFLAGS_BASE) $(BLAS_CFLAGS) \
	    tests/model/test_vit_dino.c lrm/lrm_vit_dino.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    $(BLAS_LDFLAGS) -lm -o /tmp/lrm_test_dino
	@/tmp/lrm_test_dino
	@rm -f /tmp/lrm_test_dino

test-decoder:
	@echo "[decoder] triplane decoder forward parity ..."
	@$(CC) $(CFLAGS_BASE) $(BLAS_CFLAGS) \
	    tests/model/test_triplane_decoder.c lrm/lrm_triplane_decoder.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    $(BLAS_LDFLAGS) -lm -o /tmp/lrm_test_decoder
	@/tmp/lrm_test_decoder
	@rm -f /tmp/lrm_test_decoder

test-upsample:
	@echo "[upsample] post-processor parity ..."
	@$(CC) $(CFLAGS_BASE) $(BLAS_CFLAGS) \
	    tests/model/test_triplane_upsample.c lrm/lrm_triplane_upsample.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    $(BLAS_LDFLAGS) -lm -o /tmp/lrm_test_upsample
	@/tmp/lrm_test_upsample
	@rm -f /tmp/lrm_test_upsample

test-density:
	@echo "[density] triplane-sample + NeRF MLP parity ..."
	@$(CC) $(CFLAGS_BASE) $(BLAS_CFLAGS) \
	    tests/model/test_density_64.c \
	    lrm/lrm_triplane_sample.c lrm/lrm_nerf_mlp.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    $(BLAS_LDFLAGS) -lm -o /tmp/lrm_test_density
	@/tmp/lrm_test_density
	@rm -f /tmp/lrm_test_density

test-u2net:
	@echo "[u2net] background-removal forward parity ..."
	@$(CC) $(CFLAGS_BASE) $(BLAS_CFLAGS) \
	    tests/model/test_u2net.c lrm/lrm_u2net.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    $(BLAS_LDFLAGS) -lm -o /tmp/lrm_test_u2net
	@/tmp/lrm_test_u2net
	@rm -f /tmp/lrm_test_u2net

test-density-sparse:
	@echo "[geometry] sparse-vs-dense density MC parity ..."
	@$(CC) $(CFLAGS_BASE) $(BLAS_CFLAGS) \
	    tests/geometry/test_density_sparse.c \
	    lrm/lrm_density.c lrm/lrm_triplane_sample.c lrm/lrm_nerf_mlp.c \
	    lrm/lrm_marching_cubes.c \
	    iris.c iris_kernels.c iris_safetensors.c \
	    $(BLAS_LDFLAGS) -lm -o /tmp/lrm_test_density_sparse
	@/tmp/lrm_test_density_sparse
	@rm -f /tmp/lrm_test_density_sparse

test-mc:
	@echo "[geometry] marching cubes structural parity + floater removal ..."
	@$(CC) $(CFLAGS_BASE) \
	    tests/geometry/test_marching_cubes.c lrm/lrm_marching_cubes.c iris.c \
	    -lm -o /tmp/lrm_test_mc
	@/tmp/lrm_test_mc
	@rm -f /tmp/lrm_test_mc

test-glb:
	@echo "[geometry] GLB writer structure ..."
	@$(CC) $(CFLAGS_BASE) -D_DARWIN_C_SOURCE -D_GNU_SOURCE \
	    tests/geometry/test_glb.c lrm/lrm_mesh_export.c iris.c \
	    -lm -o /tmp/lrm_test_glb
	@/tmp/lrm_test_glb
	@if [ -x triposr_env/.venv/bin/python ]; then \
	    echo "round-trip via trimesh + raw glTF inspection ..."; \
	    triposr_env/.venv/bin/python tools/check_glb.py /tmp/lrm_test.glb; \
	fi
	@rm -f /tmp/lrm_test_glb /tmp/lrm_test.glb

# =============================================================================
# Utilities
# =============================================================================
install: $(TARGET) $(LIB)
	install -d /usr/local/bin /usr/local/lib /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 iris.h iris_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) main.o $(TARGET) $(LIB)

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo "Backends: generic (pure C), blas ($(if $(filter Darwin,$(UNAME_S)),Accelerate,OpenBLAS))"
	@echo "Metal/GPU: planned (Phase 13), not yet implemented."

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
lrm/lrm_u2net.o: lrm/lrm_u2net.c lrm/lrm_u2net.h iris.h iris_kernels.h iris_safetensors.h
lrm/lrm_marching_cubes.o: lrm/lrm_marching_cubes.c lrm/lrm_marching_cubes.h iris.h
lrm/lrm_mesh_export.o: lrm/lrm_mesh_export.c lrm/lrm_mesh_export.h lrm/lrm.h iris.h
lrm/lrm_bake_texture.o: lrm/lrm_bake_texture.c lrm/lrm_bake_texture.h lrm/lrm_triplane_sample.h lrm/lrm_nerf_mlp.h iris.h iris_kernels.h
main.o: main.c iris.h lrm/lrm.h lrm/lrm_triposr.h
