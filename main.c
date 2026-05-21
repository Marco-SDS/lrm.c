/*
 * main.c - lrm.c CLI entry point.
 *
 * Phase 3 state: the CLI surface is still a placeholder, but it now wires
 * up the LRM API (lrm/lrm.h) so the linker actually pulls in lrm/lrm.o.
 * The real subcommand layer (download, convert, infer, info) lands in
 * Phase 12. For now we accept `--version`, `--help`, and a sentinel
 * `infer` that exercises lrm_load() to validate the linkage.
 */

#include <stdio.h>
#include <string.h>

#include "iris.h"
#include "lrm/lrm.h"

static void print_version(void) {
    printf("lrm.c (fork of iris.c)\n");
    printf("  fork-target: LRM-family image-to-3D models (TripoSR first)\n");
    printf("  status:      Phase 4 stub - kernels in place, LRM API not yet implemented\n");
}

static void print_help(void) {
    print_version();
    printf("\n");
    printf("usage: lrmc <command> [args]\n");
    printf("       lrmc --version|-v\n");
    printf("       lrmc --help|-h\n");
    printf("\n");
    printf("commands (Phase 3: all stubs, see LRMengine.md):\n");
    printf("  infer <model_dir> <image>   exercise lrm_load + lrm_infer\n");
    printf("\n");
    printf("Real subcommands (download, convert, info) land in Phase 12.\n");
}

static int cmd_infer(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: lrmc infer <model_dir> <image>\n");
        return 2;
    }
    const char *model_dir = argv[2];
    const char *image_path = argv[3];

    iris_image *im = iris_image_load(image_path);
    if (im == NULL) {
        fprintf(stderr, "lrmc: failed to load image '%s'\n", image_path);
        return 1;
    }

    lrm_model *m = lrm_load(model_dir);
    if (m == NULL) {
        fprintf(stderr, "lrmc: %s\n", iris_get_error());
        iris_image_free(im);
        return 1;
    }

    lrm_infer_opts opts = LRM_INFER_OPTS_DEFAULT;
    lrm_mesh *mesh = lrm_infer(m, im, &opts);
    if (mesh == NULL) {
        fprintf(stderr, "lrmc: %s\n", iris_get_error());
        lrm_free(m);
        iris_image_free(im);
        return 1;
    }

    /* Unreachable today (Phase 3 stubs always fail above); the code is here
     * so subsequent phases just plug in the real implementations. */
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
        if (strcmp(argv[1], "infer") == 0) {
            return cmd_infer(argc, argv);
        }
    }
    fprintf(stderr, "lrmc: no model loaded\n");
    fprintf(stderr, "      try 'lrmc --help' for usage.\n");
    return 1;
}
