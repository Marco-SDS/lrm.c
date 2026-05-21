/*
 * main.c - lrm.c CLI entry point (Phase 2 stub).
 *
 * After Phase 2 cleanup, this is intentionally a placeholder. The real CLI
 * (subcommands: download, convert, infer, info) will be built up in
 * Phase 12 once the LRM pipeline under lrm/ is functional.
 *
 * Currently it just prints a status line and exits, so the build system can
 * verify end-to-end that the stripped-down infrastructure compiles cleanly.
 */

#include <stdio.h>
#include <string.h>

static void print_version(void) {
    printf("lrm.c (fork of iris.c)\n");
    printf("  fork-target: LRM-family image-to-3D models (TripoSR first)\n");
    printf("  status:      Phase 2 stub - no model code linked yet\n");
}

int main(int argc, char **argv) {
    if (argc >= 2) {
        if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0) {
            print_version();
            return 0;
        }
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_version();
            printf("\n");
            printf("usage: lrm [--version|--help]\n");
            printf("       (model-specific subcommands land in Phase 12)\n");
            return 0;
        }
    }
    fprintf(stderr, "lrm: no model loaded\n");
    return 1;
}
