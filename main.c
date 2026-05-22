/*
 * main.c - lrm.c CLI entry point.
 *
 * Phase 12 state: `infer` is now end-to-end. Loads the model + image,
 * runs the full TripoSR pipeline (preprocess -> DINO -> decoder ->
 * upsample -> density grid -> MC -> vertex color requery), writes a
 * GLB. `info` keeps doing tensor-tree diagnostics for safetensors
 * inspection.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iris.h"
#include "lrm/lrm.h"
#include "lrm/lrm_triposr.h"

static void print_version(void) {
    printf("lrm.c (fork of iris.c)\n");
    printf("  fork-target: LRM-family image-to-3D models (TripoSR first)\n");
    printf("  status:      Phase 12 - end-to-end TripoSR inference live\n");
}

static void print_help(void) {
    print_version();
    printf("\n");
    printf("usage: lrmc <command> [args]\n");
    printf("       lrmc --version|-v\n");
    printf("       lrmc --help|-h\n");
    printf("\n");
    printf("commands:\n");
    printf("  info  <model_dir|.safetensors>\n");
    printf("        Print the loaded tensor tree.\n");
    printf("\n");
    printf("  infer <model_dir> <image> -o <output.glb> [options]\n");
    printf("        Run end-to-end TripoSR inference. Default options:\n");
    printf("          --mc-resolution 256\n");
    printf("          --threshold     25.0\n");
    printf("\n");
    printf("More subcommands (download, convert) arrive in Phase 17.\n");
}

static int cmd_info(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: lrmc info <model_dir|.safetensors>\n");
        return 2;
    }
    lrm_model *m = lrm_load(argv[2]);
    if (m == NULL) {
        fprintf(stderr, "lrmc: %s\n", iris_get_error());
        return 1;
    }
    lrm_triposr_print_tree(m, stdout);
    lrm_free(m);
    return 0;
}

/* Parse a non-negative integer; returns -1 on error. */
static int parse_int(const char *s) {
    char *end;
    long v = strtol(s, &end, 10);
    if (*end != '\0' || v < 0 || v > 100000) return -1;
    return (int)v;
}
static float parse_float(const char *s) {
    char *end;
    float v = strtof(s, &end);
    if (*end != '\0') return -1.0f;
    return v;
}

static double clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int cmd_infer(int argc, char **argv) {
    const char *model_dir  = NULL;
    const char *image_path = NULL;
    const char *output     = NULL;
    int   mc_resolution = 256;
    float threshold     = 25.0f;
    int   positional = 0;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-o") == 0 || strcmp(a, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lrmc: -o requires a path\n"); return 2;
            }
            output = argv[++i];
        } else if (strcmp(a, "--mc-resolution") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lrmc: --mc-resolution requires a value\n"); return 2;
            }
            mc_resolution = parse_int(argv[++i]);
            if (mc_resolution < 8) {
                fprintf(stderr, "lrmc: --mc-resolution must be >= 8\n"); return 2;
            }
        } else if (strcmp(a, "--threshold") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lrmc: --threshold requires a value\n"); return 2;
            }
            threshold = parse_float(argv[++i]);
            if (threshold <= 0.0f) {
                fprintf(stderr, "lrmc: --threshold must be > 0\n"); return 2;
            }
        } else if (a[0] == '-') {
            fprintf(stderr, "lrmc: unknown option '%s'\n", a);
            return 2;
        } else {
            if (positional == 0)      model_dir  = a;
            else if (positional == 1) image_path = a;
            else {
                fprintf(stderr, "lrmc: unexpected positional '%s'\n", a);
                return 2;
            }
            positional++;
        }
    }
    if (!model_dir || !image_path || !output) {
        fprintf(stderr,
                "usage: lrmc infer <model_dir> <image> -o <output.glb> "
                "[--mc-resolution N] [--threshold V]\n");
        return 2;
    }

    fprintf(stderr, "lrmc: model=%s\n", model_dir);
    fprintf(stderr, "lrmc: image=%s\n", image_path);
    fprintf(stderr, "lrmc: output=%s  mc_res=%d  threshold=%.2f\n",
            output, mc_resolution, threshold);

    double t0 = clock_ms();

    iris_image *im = iris_image_load(image_path);
    if (im == NULL) {
        fprintf(stderr, "lrmc: failed to load image '%s'\n", image_path);
        return 1;
    }
    fprintf(stderr, "lrmc: loaded image %dx%d %dch\n",
            im->width, im->height, im->channels);

    double t_img = clock_ms();
    lrm_model *m = lrm_load(model_dir);
    if (m == NULL) {
        fprintf(stderr, "lrmc: %s\n", iris_get_error());
        iris_image_free(im);
        return 1;
    }
    double t_load = clock_ms();
    fprintf(stderr, "lrmc: model loaded in %.0f ms\n", t_load - t_img);

    lrm_infer_opts opts = LRM_INFER_OPTS_DEFAULT;
    opts.mc_resolution    = mc_resolution;
    opts.density_threshold = threshold;
    lrm_mesh *mesh = lrm_infer(m, im, &opts);
    double t_infer = clock_ms();
    if (mesh == NULL) {
        fprintf(stderr, "lrmc: %s\n", iris_get_error());
        lrm_free(m);
        iris_image_free(im);
        return 1;
    }
    fprintf(stderr, "lrmc: inference complete in %.1f s\n",
            (t_infer - t_load) / 1000.0);

    if (lrm_mesh_save_glb(mesh, output) != 0) {
        fprintf(stderr, "lrmc: %s\n", iris_get_error());
        lrm_mesh_free(mesh);
        lrm_free(m);
        iris_image_free(im);
        return 1;
    }
    double t_save = clock_ms();
    fprintf(stderr, "lrmc: wrote %s in %.0f ms\n", output, t_save - t_infer);

    fprintf(stderr, "lrmc: total %.1f s\n", (t_save - t0) / 1000.0);

    lrm_mesh_free(mesh);
    lrm_free(m);
    iris_image_free(im);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2) {
        if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0) {
            print_version();
            return 0;
        }
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_help();
            return 0;
        }
        if (strcmp(argv[1], "info") == 0) {
            return cmd_info(argc, argv);
        }
        if (strcmp(argv[1], "infer") == 0) {
            return cmd_infer(argc, argv);
        }
    }
    fprintf(stderr, "lrmc: no command given\n");
    fprintf(stderr, "      try 'lrmc --help' for usage.\n");
    return 1;
}
