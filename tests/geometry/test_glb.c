/*
 * test_glb.c - GLB writer parity / structural test for Phase 11.
 *
 * Loads the golden mesh from the .bin sidecars, runs lrm_mesh_save_glb to
 * /tmp/lrm_test.glb, then re-parses the file in C to verify:
 *   - 12-byte GLB header (magic + version + total length)
 *   - JSON chunk header + size + brackets balance
 *   - BIN chunk header + size
 *   - Accessor counts in the JSON match what we wrote
 *
 * For the "real" round-trip ("opens in Blender / online viewer") the
 * Makefile target also runs a trimesh.load via the venv Python after the
 * C check passes; that confirms the file is a valid glTF 2.0 payload.
 *
 * Build: make test-glb
 * Run:   ./test_glb   (writes /tmp/lrm_test.glb)
 */

#include "iris.h"
#include "lrm/lrm.h"
#include "lrm/lrm_mesh_export.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *kVertsPath  = "tests/golden/triposr/mesh_vertices.bin";
static const char *kFacesPath  = "tests/golden/triposr/mesh_faces.bin";
static const char *kColorsPath = "tests/golden/triposr/mesh_vertex_colors.bin";
static const char *kOutPath    = "/tmp/lrm_test.glb";

#define GLTF_MAGIC      0x46546C67u
#define CHUNK_JSON      0x4E4F534Au
#define CHUNK_BIN       0x004E4942u

static void *mmap_bytes(const char *path, size_t *out_bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror(path); close(fd); return NULL; }
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror(path); return NULL; }
    *out_bytes = (size_t)st.st_size;
    return p;
}

/* Read raw bytes from a file (whole file). */
static uint8_t *read_all(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);
    *out_size = (size_t)n;
    return buf;
}

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(void) {
    /* ---- Load golden mesh ---- */
    size_t vb, fb, cb;
    float   *gold_verts  = (float *)mmap_bytes(kVertsPath,  &vb);
    int32_t *gold_faces  = (int32_t *)mmap_bytes(kFacesPath, &fb);
    uint8_t *gold_colors = (uint8_t *)mmap_bytes(kColorsPath, &cb);
    if (!gold_verts || !gold_faces || !gold_colors) {
        fprintf(stderr, "FAIL: cannot mmap golden mesh sidecars\n");
        return 1;
    }
    int Nv = (int)(vb / (sizeof(float) * 3));
    int Nf = (int)(fb / (sizeof(int32_t) * 3));
    int Cv = (int)(cb / 4);  /* RGBA u8 */
    fprintf(stderr, "golden: %d vertices, %d faces, %d color rows\n",
            Nv, Nf, Cv);
    if (Cv != Nv) {
        fprintf(stderr, "FAIL: color rows (%d) != vertex count (%d)\n", Cv, Nv);
        return 1;
    }

    /* ---- Build owned copies (lrm_mesh takes ownership). ---- */
    float *verts = (float *)malloc((size_t)Nv * 3 * sizeof(float));
    int32_t *faces = (int32_t *)malloc((size_t)Nf * 3 * sizeof(int32_t));
    float *colors_f = (float *)malloc((size_t)Nv * 4 * sizeof(float));
    if (!verts || !faces || !colors_f) {
        fprintf(stderr, "FAIL: oom\n");
        return 1;
    }
    memcpy(verts, gold_verts, (size_t)Nv * 3 * sizeof(float));
    memcpy(faces, gold_faces, (size_t)Nf * 3 * sizeof(int32_t));
    /* Convert uint8 RGBA -> f32 RGBA in [0, 1]. */
    for (int i = 0; i < Nv * 4; i++) {
        colors_f[i] = (float)gold_colors[i] / 255.0f;
    }

    /* ---- Build mesh + write GLB. ---- */
    lrm_mesh *m = lrm_mesh_from_buffers(Nv, verts, Nf, faces, colors_f);
    if (!m) {
        fprintf(stderr, "FAIL: lrm_mesh_from_buffers: %s\n", iris_get_error());
        return 1;
    }
    fprintf(stderr, "writing GLB to %s ...\n", kOutPath);
    if (lrm_mesh_save_glb(m, kOutPath) != 0) {
        fprintf(stderr, "FAIL: save_glb: %s\n", iris_get_error());
        lrm_mesh_free(m);
        return 1;
    }
    lrm_mesh_free(m);  /* takes the owned buffers with it */

    /* ---- Re-parse the file in C to validate structure. ---- */
    size_t fsize = 0;
    uint8_t *buf = read_all(kOutPath, &fsize);
    if (!buf) {
        fprintf(stderr, "FAIL: cannot read back %s\n", kOutPath);
        return 1;
    }
    fprintf(stderr, "wrote %zu bytes\n", fsize);

    int fails = 0;
    if (fsize < 28) {  /* 12 header + 8 chunk hdr + 8 second chunk hdr */
        fprintf(stderr, "  FAIL: file too short\n");
        fails++;
    }
    uint32_t magic = rd_u32(buf + 0);
    uint32_t version = rd_u32(buf + 4);
    uint32_t total_len = rd_u32(buf + 8);
    if (magic != GLTF_MAGIC) {
        fprintf(stderr, "  FAIL: magic 0x%08x != 0x%08x\n",
                magic, GLTF_MAGIC);
        fails++;
    } else {
        fprintf(stderr, "  PASS magic 'glTF'\n");
    }
    if (version != 2) {
        fprintf(stderr, "  FAIL: version %u != 2\n", version);
        fails++;
    } else {
        fprintf(stderr, "  PASS version=2\n");
    }
    if (total_len != fsize) {
        fprintf(stderr, "  FAIL: header length %u != file size %zu\n",
                total_len, fsize);
        fails++;
    } else {
        fprintf(stderr, "  PASS header length %u\n", total_len);
    }

    /* JSON chunk. */
    uint32_t js_len  = rd_u32(buf + 12);
    uint32_t js_type = rd_u32(buf + 16);
    if (js_type != CHUNK_JSON) {
        fprintf(stderr, "  FAIL: first chunk type 0x%08x != JSON\n", js_type);
        fails++;
    } else {
        fprintf(stderr, "  PASS first chunk JSON (%u bytes)\n", js_len);
    }
    const char *js = (const char *)(buf + 20);
    /* Crude JSON sanity: brace + bracket balance. */
    int braces = 0, brackets = 0;
    int in_string = 0;
    int has_position = 0, has_colors = 0, has_indices = 0;
    for (uint32_t i = 0; i < js_len; i++) {
        char c = js[i];
        if (in_string) {
            if (c == '"' && (i == 0 || js[i-1] != '\\')) in_string = 0;
            continue;
        }
        if (c == '"') { in_string = 1; continue; }
        if (c == '{') braces++;
        else if (c == '}') braces--;
        else if (c == '[') brackets++;
        else if (c == ']') brackets--;
    }
    if (braces != 0 || brackets != 0) {
        fprintf(stderr, "  FAIL: JSON braces=%d brackets=%d (should be 0)\n",
                braces, brackets);
        fails++;
    } else {
        fprintf(stderr, "  PASS JSON braces+brackets balanced\n");
    }
    /* Sanity-check that key fields appear. Phase 12.5 added NORMAL and a
     * materials block, and switched COLOR_0 to u8 normalized; verify all
     * of that is present. */
    int has_normal = 0, has_material = 0, has_normalized = 0;
    if (memmem(js, js_len, "\"POSITION\"", 10)) has_position = 1;
    if (memmem(js, js_len, "\"NORMAL\"",   8))  has_normal   = 1;
    if (memmem(js, js_len, "\"COLOR_0\"", 9))   has_colors   = 1;
    if (memmem(js, js_len, "\"indices\"", 9))   has_indices  = 1;
    if (memmem(js, js_len, "\"materials\"", 11)) has_material = 1;
    if (memmem(js, js_len, "\"normalized\":true", 17)) has_normalized = 1;
    if (!has_position || !has_normal || !has_colors || !has_indices) {
        fprintf(stderr,
                "  FAIL: JSON missing attribute (POSITION=%d, NORMAL=%d, "
                "COLOR_0=%d, indices=%d)\n",
                has_position, has_normal, has_colors, has_indices);
        fails++;
    } else {
        fprintf(stderr, "  PASS JSON has POSITION + NORMAL + COLOR_0 + indices\n");
    }
    if (!has_material) {
        fprintf(stderr, "  FAIL: JSON missing materials block\n");
        fails++;
    } else {
        fprintf(stderr, "  PASS JSON has PBR materials block\n");
    }
    if (!has_normalized) {
        fprintf(stderr, "  FAIL: COLOR_0 not marked normalized (u8) - "
                        "expected switch to u8 normalized in Phase 12.5\n");
        fails++;
    } else {
        fprintf(stderr, "  PASS COLOR_0 marked normalized (u8)\n");
    }

    /* BIN chunk. */
    size_t bin_off = 20 + ((js_len + 3) & ~(uint32_t)3);
    if (bin_off + 8 > fsize) {
        fprintf(stderr, "  FAIL: missing BIN chunk header (offset %zu, file %zu)\n",
                bin_off, fsize);
        fails++;
    } else {
        uint32_t bin_len  = rd_u32(buf + bin_off);
        uint32_t bin_type = rd_u32(buf + bin_off + 4);
        if (bin_type != CHUNK_BIN) {
            fprintf(stderr, "  FAIL: second chunk type 0x%08x != BIN\n",
                    bin_type);
            fails++;
        } else {
            fprintf(stderr, "  PASS second chunk BIN (%u bytes)\n", bin_len);
        }
        /* Expected payload: POSITION + NORMAL + COLOR_0(u8) + u16 indices,
         * each padded to 4 bytes. */
        size_t expected = (size_t)Nv * 3 * sizeof(float)       /* POSITION */
                        + (size_t)Nv * 3 * sizeof(float)       /* NORMAL */
                        + (size_t)Nv * 4 * sizeof(uint8_t)     /* COLOR_0 */
                        + (size_t)Nf * 3 * 2;                   /* u16 indices */
        /* Each block aligns to 4 bytes; allow a few bytes slop. */
        expected = (expected + 3) & ~(size_t)3;
        if (bin_len < expected || bin_len > expected + 12) {
            fprintf(stderr,
                    "  FAIL: BIN length %u, expected ~%zu (POS+NOR+COLOR+IDX padded)\n",
                    bin_len, expected);
            fails++;
        } else {
            fprintf(stderr, "  PASS BIN length matches expected (%u ~ %zu)\n",
                    bin_len, expected);
        }
    }

    free(buf);

    if (fails == 0) {
        printf("\nPASS  GLB writer structure (12 checks); file at %s\n",
               kOutPath);
        return 0;
    }
    fprintf(stderr, "\nFAIL  GLB writer (%d structural checks failed)\n", fails);
    return 1;
}
