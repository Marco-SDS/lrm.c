/*
 * lrm_marching_cubes.c - Lorensen-Cline marching cubes with edge dedup.
 *
 * Tables are the canonical 256-entry edge_table and tri_table from
 * Paul Bourke's polygonise reference (public domain;
 * https://paulbourke.net/geometry/polygonise/), with the standard
 * corner numbering:
 *
 *        v4 ----- v5
 *       /|       /|
 *     v7 ----- v6 |
 *     |  v0----|--v1
 *     | /     | /
 *     v3 ----- v2
 *
 *   Corners (X, Y, Z): 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
 *                      4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
 *   Edges connect (Bourke convention):
 *      0: 0->1   1: 1->2   2: 2->3   3: 3->0   (z=0 face, CCW)
 *      4: 4->5   5: 5->6   6: 6->7   7: 7->4   (z=1 face)
 *      8: 0->4   9: 1->5  10: 2->6  11: 3->7   (vertical)
 *
 * Mapping of (X, Y, Z) corner coords to our density indexing convention:
 *   X -> first axis (i), Y -> second axis (j), Z -> third axis (k).
 * So corner 5 of cube (a, b, c) is density[a+1, b, c+1].
 *
 * Vertex dedup: each cube edge corresponds to a unique global grid edge.
 * We index a [3 * R * R * R] table mapping (direction, i, j, k) to the
 * vertex index already emitted for that edge, or -1 if not yet emitted.
 */

#include "lrm_marching_cubes.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iris.h"

/* ========================================================================
 * Lorensen-Cline tables (Paul Bourke / public domain)
 * ======================================================================== */

/* Which of the 12 edges are crossed for each of the 256 cases.
 * Bit i is set if edge i is intersected by the surface. */
static const uint16_t kEdgeTable[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

/* For each of the 256 cube cases, up to 5 triangles defined as triples of
 * edge indices. -1 terminates. */
static const int8_t kTriTable[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1},
    { 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1},
    { 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1},
    { 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1},
    { 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1},
    { 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1},
    { 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1},
    { 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1},
    { 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1},
    { 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1},
    { 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1},
    { 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1},
    { 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1},
    { 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1},
    { 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1},
    { 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1},
    { 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1},
    { 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1},
    { 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1},
    {10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1},
    { 5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1},
    { 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1},
    { 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1},
    { 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1},
    { 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1},
    { 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1},
    { 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1},
    { 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1},
    { 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1},
    {11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1},
    { 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1},
    {11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1},
    {11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1},
    { 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1},
    { 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1},
    { 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1},
    { 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1},
    { 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1},
    { 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1},
    { 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1},
    { 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1},
    { 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1},
    { 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1},
    { 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1},
    { 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1},
    { 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1},
    { 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1},
    { 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1},
    { 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1},
    { 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1},
    { 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1},
    { 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1},
    { 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1},
    { 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1},
    {10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1},
    {10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1},
    { 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1},
    { 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1},
    { 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1},
    {10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1},
    { 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1},
    { 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1},
    { 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1},
    { 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1},
    { 3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1},
    { 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1},
    { 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1},
    {10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1},
    {10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1},
    { 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1},
    { 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1},
    { 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1},
    { 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1},
    { 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1},
    {11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1},
    { 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1},
    { 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1},
    { 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1},
    {10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1},
    { 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1},
    { 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1},
    { 7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1},
    { 2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1},
    { 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1},
    {10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1},
    {10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1},
    { 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1},
    { 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1},
    { 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1},
    { 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1},
    { 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1},
    { 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1},
    {10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1},
    { 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1},
    { 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1},
    { 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1},
    { 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1},
    {10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1},
    { 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1},
    {10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1},
    { 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1},
    { 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1},
    { 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1},
    { 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1},
    { 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1},
    { 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1},
    { 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1},
    { 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1},
    { 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1},
    { 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1},
    { 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1},
    { 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1},
    { 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1},
    { 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1},
    { 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1},
    { 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1},
    { 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1},
    {11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1},
    { 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1},
    { 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1},
    { 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1},
    { 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1},
    { 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1},
    {10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1},
    { 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1},
    { 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1},
    {10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1},
    {11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1},
    { 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1},
    { 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1},
    { 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1},
    { 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1},
    { 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1},
    { 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1},
    { 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1},
    { 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1},
    { 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1},
    { 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1},
    { 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1},
    {10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1},
    { 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1},
    { 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1},
    { 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1},
    { 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1},
    { 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1},
    { 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1},
    { 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1},
    { 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1},
    { 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1},
    { 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1},
    { 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1},
    { 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1},
    { 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1},
    {11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1},
    {11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1},
    { 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1},
    { 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1},
    { 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1},
    { 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1},
    { 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1},
    { 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1},
    { 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1},
    { 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1},
    { 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

/* Per-edge corner endpoints (which two corners each edge connects). */
static const int kEdgeCorners[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {4,5}, {5,6}, {6,7}, {7,4},
    {0,4}, {1,5}, {2,6}, {3,7}
};

/* Corner offsets in (X, Y, Z) = (i, j, k). */
static const int kCornerOffset[8][3] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
};

/* ========================================================================
 * Helpers
 * ======================================================================== */

static int set_err(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    iris_set_error(buf);
    return -1;
}

/* Per-edge (direction, di, dj, dk) lookup, filled lazily at first call. */
static int g_edge_grid[12][4];
static int g_edge_grid_init = 0;

static void init_edge_grid(void) {
    if (g_edge_grid_init) return;
    /* edge 0 corners (0,0,0)-(1,0,0): X dir at (0,0,0) */
    int data[12][4] = {
        {0, 0, 0, 0},  /* edge 0 X at (i, j, k) */
        {1, 1, 0, 0},  /* edge 1 Y at (i+1, j, k) */
        {0, 0, 1, 0},  /* edge 2 X at (i, j+1, k) */
        {1, 0, 0, 0},  /* edge 3 Y at (i, j, k) */
        {0, 0, 0, 1},  /* edge 4 X at (i, j, k+1) */
        {1, 1, 0, 1},  /* edge 5 Y at (i+1, j, k+1) */
        {0, 0, 1, 1},  /* edge 6 X at (i, j+1, k+1) */
        {1, 0, 0, 1},  /* edge 7 Y at (i, j, k+1) */
        {2, 0, 0, 0},  /* edge 8 Z at (i, j, k) */
        {2, 1, 0, 0},  /* edge 9 Z at (i+1, j, k) */
        {2, 1, 1, 0},  /* edge 10 Z at (i+1, j+1, k) */
        {2, 0, 1, 0},  /* edge 11 Z at (i, j+1, k) */
    };
    memcpy(g_edge_grid, data, sizeof(data));
    g_edge_grid_init = 1;
}

static inline int edge_global_id(int R, int edge, int a, int b, int c) {
    int dir = g_edge_grid[edge][0];
    int i = a + g_edge_grid[edge][1];
    int j = b + g_edge_grid[edge][2];
    int k = c + g_edge_grid[edge][3];
    /* Single id space: direction * R^3 + i * R^2 + j * R + k. */
    return ((dir * R + i) * R + j) * R + k;
}

/* Grow a dynamic float buffer; returns 0 on success. */
static int grow_floats(float **buf, int *cap, int need) {
    if (*cap >= need) return 0;
    int new_cap = (*cap > 0) ? *cap : 1024;
    while (new_cap < need) new_cap *= 2;
    float *p = (float *)realloc(*buf, (size_t)new_cap * sizeof(float));
    if (!p) return set_err("marching_cubes: oom (vertices)");
    *buf = p;
    *cap = new_cap;
    return 0;
}
static int grow_ints(int32_t **buf, int *cap, int need) {
    if (*cap >= need) return 0;
    int new_cap = (*cap > 0) ? *cap : 1024;
    while (new_cap < need) new_cap *= 2;
    int32_t *p = (int32_t *)realloc(*buf, (size_t)new_cap * sizeof(int32_t));
    if (!p) return set_err("marching_cubes: oom (faces)");
    *buf = p;
    *cap = new_cap;
    return 0;
}

/* ========================================================================
 * Main extraction
 * ======================================================================== */

int lrm_marching_cubes_extract(const float *volume, int R,
                               float threshold,
                               float world_min, float world_max,
                               lrm_mc_mesh *mesh) {
    if (!volume || !mesh || R < 2) {
        return set_err("marching_cubes: bad arguments (R=%d)", R);
    }
    init_edge_grid();
    memset(mesh, 0, sizeof(*mesh));

    /* Edge-id -> vertex-index map. */
    size_t edge_table_n = (size_t)3 * (size_t)R * (size_t)R * (size_t)R;
    int *edge_to_vertex = (int *)malloc(edge_table_n * sizeof(int));
    if (!edge_to_vertex) {
        return set_err("marching_cubes: oom (edge table, %zu ints)",
                       edge_table_n);
    }
    memset(edge_to_vertex, 0xff, edge_table_n * sizeof(int));  /* -1 */

    float   *verts     = NULL;
    int32_t *faces     = NULL;
    int      verts_cap = 0, verts_n = 0;
    int      faces_cap = 0, faces_n = 0;

    /* Normalization helpers. Vertex output is mapped from grid coords
     * [0, R-1] to normalized [0, 1] (divide by R-1), then linearly to
     * [world_min, world_max]. We do this once per emitted vertex.
     * The TripoSR [2, 1, 0] axis swap is applied at write time: the
     * grid (i, j, k) becomes vertex (k_pos, j_pos, i_pos).
     *
     *   norm = grid / (R - 1)
     *   world = norm * (world_max - world_min) + world_min
     */
    const float inv_rm1 = 1.0f / (float)(R - 1);
    const float scale   = world_max - world_min;
    const float bias    = world_min;

    /* Inner-loop precomputed corner strides (i, j, k -> linear offset). */
    /* density[i, j, k] in our convention = volume[i*R*R + j*R + k]. */
    int corner_offset[8];
    for (int c = 0; c < 8; c++) {
        int di = kCornerOffset[c][0];
        int dj = kCornerOffset[c][1];
        int dk = kCornerOffset[c][2];
        corner_offset[c] = di * R * R + dj * R + dk;
    }

    for (int a = 0; a < R - 1; a++) {
        for (int b = 0; b < R - 1; b++) {
            int base = a * R * R + b * R;
            for (int c = 0; c < R - 1; c++) {
                int origin = base + c;

                /* Classify the 8 corners against threshold. The MC
                 * convention emits a surface where density - threshold
                 * crosses 0; corners "inside" (density > threshold)
                 * contribute their bit to cube_index. */
                int cube_index = 0;
                float dens[8];
                for (int k = 0; k < 8; k++) {
                    dens[k] = volume[origin + corner_offset[k]];
                    if (dens[k] > threshold) cube_index |= (1 << k);
                }
                uint16_t edge_mask = kEdgeTable[cube_index];
                if (edge_mask == 0) continue;

                /* Resolve / emit vertices for the crossed edges of this cube. */
                int vert_ids[12];
                for (int e = 0; e < 12; e++) vert_ids[e] = -1;
                for (int e = 0; e < 12; e++) {
                    if (!(edge_mask & (1 << e))) continue;
                    int gid = edge_global_id(R, e, a, b, c);
                    int existing = edge_to_vertex[gid];
                    if (existing >= 0) {
                        vert_ids[e] = existing;
                        continue;
                    }
                    /* Linear interpolation along the edge.
                     * t = (threshold - dens_a) / (dens_b - dens_a). */
                    int ca = kEdgeCorners[e][0];
                    int cb = kEdgeCorners[e][1];
                    float va = dens[ca];
                    float vb = dens[cb];
                    float denom = vb - va;
                    float t = (denom == 0.0f) ? 0.5f
                                              : (threshold - va) / denom;
                    if (t < 0.0f) t = 0.0f;
                    if (t > 1.0f) t = 1.0f;

                    /* Corner positions in grid coords. */
                    float pa[3] = {
                        (float)(a + kCornerOffset[ca][0]),
                        (float)(b + kCornerOffset[ca][1]),
                        (float)(c + kCornerOffset[ca][2])
                    };
                    float pb[3] = {
                        (float)(a + kCornerOffset[cb][0]),
                        (float)(b + kCornerOffset[cb][1]),
                        (float)(c + kCornerOffset[cb][2])
                    };
                    float vp_i = pa[0] + t * (pb[0] - pa[0]);
                    float vp_j = pa[1] + t * (pb[1] - pa[1]);
                    float vp_k = pa[2] + t * (pb[2] - pa[2]);

                    /* Normalize to [0, 1] then scale to world. */
                    float ni = vp_i * inv_rm1;
                    float nj = vp_j * inv_rm1;
                    float nk = vp_k * inv_rm1;
                    float wi = ni * scale + bias;
                    float wj = nj * scale + bias;
                    float wk = nk * scale + bias;

                    /* Append vertex in (i, j, k)-axis-natural order,
                     * which after the (0, 1) -> (world_min, world_max)
                     * scale gives world position with axis 0 = density's
                     * first axis. This matches TripoSR's final mesh
                     * coords: torchmcubes outputs (k, j, i) (since it
                     * treats vol as [depth=z, height=y, width=x]) and
                     * TripoSR applies [2, 1, 0] to swap back to
                     * (i, j, k). We bypass that round-trip. */
                    if (grow_floats(&verts, &verts_cap,
                                    (verts_n + 1) * 3) != 0) goto fail;
                    verts[verts_n * 3 + 0] = wi;
                    verts[verts_n * 3 + 1] = wj;
                    verts[verts_n * 3 + 2] = wk;
                    edge_to_vertex[gid] = verts_n;
                    vert_ids[e] = verts_n;
                    verts_n++;
                }

                /* Emit triangles for this cube. */
                const int8_t *row = kTriTable[cube_index];
                for (int t = 0; row[t] >= 0; t += 3) {
                    int e0 = row[t];
                    int e1 = row[t + 1];
                    int e2 = row[t + 2];
                    int v0 = vert_ids[e0];
                    int v1 = vert_ids[e1];
                    int v2 = vert_ids[e2];
                    if (v0 < 0 || v1 < 0 || v2 < 0) {
                        /* Should not happen if edge_table is consistent
                         * with tri_table - guard against bad case rows. */
                        continue;
                    }
                    if (grow_ints(&faces, &faces_cap,
                                  (faces_n + 1) * 3) != 0) goto fail;
                    /* Emit with reversed winding (v0, v2, v1). Our cube_index
                     * sets a corner bit when density > threshold ("inside"),
                     * which is the opposite of Paul Bourke's val < isolevel
                     * convention the tri_table is authored for; without this
                     * swap every triangle faces inward (negative signed
                     * volume). Reversing makes the mesh canonically
                     * CCW-outward, so consumers that recompute normals from
                     * the winding (and our analytic gradient normals) agree. */
                    faces[faces_n * 3 + 0] = v0;
                    faces[faces_n * 3 + 1] = v2;
                    faces[faces_n * 3 + 2] = v1;
                    faces_n++;
                }
            }
        }
    }

    free(edge_to_vertex);
    /* Trim to exact size. */
    if (verts_n > 0) {
        float *p = (float *)realloc(verts, (size_t)verts_n * 3 * sizeof(float));
        if (p) verts = p;
    }
    if (faces_n > 0) {
        int32_t *p = (int32_t *)realloc(faces, (size_t)faces_n * 3 * sizeof(int32_t));
        if (p) faces = p;
    }
    mesh->n_vertices = verts_n;
    mesh->n_faces    = faces_n;
    mesh->vertices   = verts;
    mesh->faces      = faces;
    return 0;

fail:
    free(edge_to_vertex);
    free(verts);
    free(faces);
    memset(mesh, 0, sizeof(*mesh));
    return -1;
}

/* ========================================================================
 * Connected-component floater removal
 * ======================================================================== */

static int uf_find(int *parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  /* path halving */
        x = parent[x];
    }
    return x;
}

static void uf_union(int *parent, int a, int b) {
    int ra = uf_find(parent, a), rb = uf_find(parent, b);
    if (ra != rb) parent[ra] = rb;
}

int lrm_mc_remove_small_components(lrm_mc_mesh *mesh, float min_fraction) {
    if (!mesh) return set_err("remove_components: NULL mesh");
    if (min_fraction <= 0.0f || mesh->n_vertices <= 0 || mesh->n_faces <= 0) {
        return 0;  /* disabled or empty */
    }
    const int Nv = mesh->n_vertices;
    const int Nf = mesh->n_faces;

    int *parent = (int *)malloc((size_t)Nv * sizeof(int));
    int *fcount = (int *)calloc((size_t)Nv, sizeof(int));  /* faces per root */
    if (!parent || !fcount) {
        free(parent); free(fcount);
        return set_err("remove_components: oom (union-find)");
    }
    for (int v = 0; v < Nv; v++) parent[v] = v;
    for (int f = 0; f < Nf; f++) {
        int a = mesh->faces[f*3+0], b = mesh->faces[f*3+1], c = mesh->faces[f*3+2];
        uf_union(parent, a, b);
        uf_union(parent, b, c);
    }
    /* Tally triangles per component root and find the largest. */
    int max_faces = 0;
    for (int f = 0; f < Nf; f++) {
        int r = uf_find(parent, mesh->faces[f*3+0]);
        fcount[r]++;
        if (fcount[r] > max_faces) max_faces = fcount[r];
    }
    int min_faces = (int)(min_fraction * (float)max_faces);
    if (min_faces < 1) min_faces = 1;

    /* Build a compacted vertex remap for vertices in kept components. */
    int *vmap = (int *)malloc((size_t)Nv * sizeof(int));
    if (!vmap) {
        free(parent); free(fcount);
        return set_err("remove_components: oom (vmap)");
    }
    for (int v = 0; v < Nv; v++) vmap[v] = -1;

    float   *nv = (float *)malloc((size_t)Nv * 3 * sizeof(float));
    int32_t *nf = (int32_t *)malloc((size_t)Nf * 3 * sizeof(int32_t));
    if (!nv || !nf) {
        free(parent); free(fcount); free(vmap); free(nv); free(nf);
        return set_err("remove_components: oom (compacted buffers)");
    }

    int out_nv = 0, out_nf = 0, removed_comp = 0;
    for (int f = 0; f < Nf; f++) {
        int r = uf_find(parent, mesh->faces[f*3+0]);
        if (fcount[r] < min_faces) continue;  /* drop floater face */
        int32_t out_idx[3];
        for (int k = 0; k < 3; k++) {
            int v = mesh->faces[f*3+k];
            if (vmap[v] < 0) {
                vmap[v] = out_nv;
                nv[out_nv*3+0] = mesh->vertices[v*3+0];
                nv[out_nv*3+1] = mesh->vertices[v*3+1];
                nv[out_nv*3+2] = mesh->vertices[v*3+2];
                out_nv++;
            }
            out_idx[k] = vmap[v];
        }
        nf[out_nf*3+0] = out_idx[0];
        nf[out_nf*3+1] = out_idx[1];
        nf[out_nf*3+2] = out_idx[2];
        out_nf++;
    }
    /* Count dropped components for diagnostics. */
    for (int v = 0; v < Nv; v++) {
        if (parent[v] == v && fcount[v] > 0 && fcount[v] < min_faces) {
            removed_comp++;
        }
    }

    free(parent); free(fcount); free(vmap);

    if (out_nf == 0) {
        /* min_fraction too aggressive - keep the original mesh untouched. */
        free(nv); free(nf);
        return set_err("remove_components: would empty the mesh");
    }

    /* Swap in the compacted buffers (trim to exact size). */
    float *nv2 = (float *)realloc(nv, (size_t)out_nv * 3 * sizeof(float));
    if (nv2) nv = nv2;
    int32_t *nf2 = (int32_t *)realloc(nf, (size_t)out_nf * 3 * sizeof(int32_t));
    if (nf2) nf = nf2;

    free(mesh->vertices);
    free(mesh->faces);
    mesh->vertices   = nv;
    mesh->faces      = nf;
    mesh->n_vertices = out_nv;
    mesh->n_faces    = out_nf;

    if (getenv("LRM_TIMING")) {
        fprintf(stderr,
                "lrmc:   (floaters: dropped %d components, V %d->%d F %d->%d)\n",
                removed_comp, Nv, out_nv, Nf, out_nf);
    }
    return 0;
}

void lrm_mc_mesh_free(lrm_mc_mesh *mesh) {
    if (!mesh) return;
    free(mesh->vertices);
    free(mesh->faces);
    mesh->vertices = NULL;
    mesh->faces    = NULL;
    mesh->n_vertices = 0;
    mesh->n_faces    = 0;
}
