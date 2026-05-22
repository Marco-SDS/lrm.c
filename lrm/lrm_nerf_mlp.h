/*
 * lrm_nerf_mlp.h - NeRF-style decoder MLP.
 *
 * Mirrors TripoSR's NeRFMLP (tsr/models/network_utils.py): 10 Linear layers
 * with SiLU between, no activation after the final layer (activations are
 * applied externally per channel: density vs color).
 *
 * Layer structure (in_channels=120, n_neurons=64, n_hidden_layers=9):
 *
 *   layers.0       Linear 120 -> 64    + SiLU
 *   layers.2..16   Linear  64 -> 64    + SiLU      (8 blocks)
 *   layers.18      Linear  64 -> 4                  (no activation here)
 *
 * Output channels [0:1] = raw density, [1:4] = raw color features.
 * The caller applies density_activation = exp(raw + density_bias)
 * and color_activation = sigmoid(features) (see lrm_query_triplane).
 *
 * Weights are borrowed from the safetensors mmap (no owned allocations).
 */

#ifndef LRM_LRM_NERF_MLP_H
#define LRM_LRM_NERF_MLP_H

#include <stddef.h>

#include "iris_safetensors.h"

#define LRM_NERF_MLP_MAX_LAYERS 16  /* generous upper bound */

typedef struct lrm_nerf_mlp {
    int in_channels;       /* 120 */
    int hidden_dim;        /* 64  */
    int out_channels;      /* 4   */
    int num_layers;        /* 10  */
    /* layer_idx[i] holds the index inside the nn.Sequential that the
     * i-th Linear sits at. For TripoSR: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18. */
    int seq_index[LRM_NERF_MLP_MAX_LAYERS];
    const float *weight[LRM_NERF_MLP_MAX_LAYERS]; /* shapes vary */
    const float *bias  [LRM_NERF_MLP_MAX_LAYERS];

    /* Density / color post-processing. */
    float density_bias;     /* -1.0 */
} lrm_nerf_mlp;

/* Bind weights from the safetensors. Returns 0 on success. */
int lrm_nerf_mlp_init(lrm_nerf_mlp *mlp, const safetensors_file_t *sf);

/* Workspace required for N_points queries. */
size_t lrm_nerf_mlp_workspace_bytes(const lrm_nerf_mlp *mlp, int N_points);

/*
 * Forward pass with density/color activations applied to the outputs.
 *
 *   features  : [N_points, 120]  borrowed input (the triplane sample output)
 *   density   : [N_points]       writes post-activation density (exp(raw - 1))
 *   color     : [N_points, 3]    writes post-activation color (sigmoid(raw))
 *   work      : scratch >= lrm_nerf_mlp_workspace_bytes(mlp, N_points)
 *
 * Returns 0 on success.
 */
int lrm_nerf_mlp_forward(const lrm_nerf_mlp *mlp,
                         const float *features,
                         int N_points,
                         float *density,
                         float *color,
                         float *work);

#endif /* LRM_LRM_NERF_MLP_H */
