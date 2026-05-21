/*
 * main.c - lrm.c CLI entry point.
 *
 * Phase 5 state: `info` is the first subcommand that does real work -
 * loads the safetensors and prints the full weight tree. `infer` still
 * fails at the lrm_infer stub, but the loader path is now exercised
 * end-to-end. The real subcommand layer (download, convert) lands in
 * Phase 12.
 */

#include <stdio.h>
#include <string.h>

#include "iris.h"
#include "lrm/lrm.h"
#include "lrm/lrm_triposr.h"

static void print_version(void) {
    printf("lrm.c (fork of iris.c)\n");
    printf("  fork-target: LRM-family image-to-3D models (TripoSR first)\n");
    printf("  status:      Phase 5 - safetensors loader live, forward pass to come\n");
}

static void print_help(void) {
    print_version();
    printf("\n");
    printf("usage: lrmc <command> [args]\n");
    printf("       lrmc --version|-v\n");
    printf("       lrmc --help|-h\n");
    printf("\n");
    printf("commands:\n");
    printf("  info  <model_dir|.safetensors>          print the loaded tensor tree\n");
    printf("  infer <model_dir> <image>               run end-to-end (Phases 6-10 wip)\n");
    printf("\n");
    printf("More subcommands (download, convert) arrive in Phase 12.\n");
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

    /* Unreachable today (lrm_infer is a stub); the code is here so subsequent
     * phases just plug in the real implementations. */
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
