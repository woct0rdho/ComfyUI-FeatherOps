# gfx1151 FP16 NT HHS Input Shapes

## Existing TensileLite Data

Generated from `~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/`.
Exact shape rows are `[m, n, batch, k]`. All extracted rows have `batch=1`.

### Key Counts

- Current NT HHS exact table: 470 unique shapes.
- Current NT HHS AuxH exact table: 470 unique shapes. Identical to non-AuxH.
- Tuned TN HHS main large grid: 9679 entries, 9679 unique shapes.
- Tuned NN HHS main large grid: 9766 entries, 9677 unique shapes, 89 duplicate entries.
- Large-grid target union from tuned HHS/BBS TN/NN sources: 9681 unique shapes.
- Large-grid target shapes missing from current NT HHS: 9400.
- Current NT HHS shapes not present in the large-grid target union: 189.
- Overlap between current NT HHS and large-grid target union: 281.

### Files

Saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/`:
- `current_nt_hhs_shapes.csv`: current NT HHS unique exact shapes.
- `current_nt_hhs_aux_shapes.csv`: current NT HHS AuxH unique exact shapes.
- `large_grid_tn_hhs_shapes_ordered.csv`: ordered exact rows from the tuned TN HHS main GridBased YAML.
- `large_grid_nn_hhs_shapes_ordered_with_duplicates.csv`: ordered exact rows from the tuned NN HHS main GridBased YAML, preserving duplicate rows.
- `large_grid_target_union_shapes.csv`: unique target large-grid union from tuned HHS/BBS TN/NN sources.
- `large_grid_target_union_decomposition.json`: exact non-overlapping product-block decomposition of `large_grid_target_union_shapes.csv`; 35 direct-product blocks reconstruct all 9,681 shapes with zero missing or extra rows.
- `large_grid_target_missing_from_current_nt_hhs.csv`: target large-grid shapes that current NT HHS does not cover.
- `current_nt_hhs_not_in_large_grid_target.csv`: current NT HHS shapes outside the large-grid target union.
- `current_nt_hhs_intersection_large_grid_target.csv`: overlap between current NT HHS and target union.
- `per_k_summary.csv`: per-K compact summary for current NT, target union, and TN HHS.
- `manifest.json`: source-file metadata, dimensions, counts-by-K, and duplicate details.

### Large-Grid Target Dimension Sets

- `M`: [1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1216, 1344, 1472, 1536, 1600, 1728, 1856, 1984, 2048, 2112, 2368, 2624, 2880, 3072, 3136, 3392, 3648, 3904, 4096, 8192]
- `N`: [1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1216, 1280, 1344, 1472, 1536, 1600, 1728, 1792, 1856, 1984, 2048, 2112, 2304, 2368, 2560, 2624, 2816, 2880, 3072, 3136, 3328, 3392, 3584, 3648, 3840, 3904, 4096, 8192]
- `K`: [1, 2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192]

### Possible M/N Asymmetry Explanation

No authoritative source was found that explicitly documents why the large-grid `M` and `N` dimension sets differ. The most likely explanation is that the gfx1151 TN HHS large grid inherited a workload-oriented Navi48/Navi31 grid rather than being freshly designed as a symmetric GEMM grid.

Evidence:
- PR #3112 is the origin point for the gfx1151 large-grid asymmetry. Its description says the grid was made denser, inspired by the Navi48 grid, and that kernels/indexes were taken from Navi31.
- Current gfx1151 TN HHS, gfx1201 TN HHS, and navi31 TN HHS have the same `N`-superset asymmetry. gfx1200 TN HHS uses symmetric `M`/`N` sets, so this is not a generic gfx12 requirement.
- The `N`-only values are `[1280, 1792, 2304, 2560, 2816, 3328, 3584, 3840]`.
- These `N`-only values appear only with `M in {1,16,32,3072}` and `K in {1,16,32,3072}`. That looks like a targeted feature/output-dimension slice, not an accidental full Cartesian grid.

Interpretation: for TN forward-linear style workloads, `M` is commonly token/batch rows and `N` is commonly output/features. It is therefore plausible for `N` to include extra model-dimension or tensor-parallel shard sizes that are not equally useful as `M`.

AITER configs provide circumstantial support for this pattern, not proof that they were the source. Several `N`-only values map to model dimensions after tensor-parallel splitting, for example `5120 / 4 = 1280`, `7168 / 4 = 1792`, `14336 / 4 = 3584`, `18432 / 8 = 2304`, `10240 / 4 = 2560`, and `53248 / 16 = 3328`. Direct AITER evidence was not found for every `N`-only value, notably `2816` and `3840`.

No AITER-style hipBLASLt target model config was found under `~/rocm-libraries/`. The closest matches are in Composable Kernel, including FMHA testing data that explicitly references production configs from AITER `model_shapes.json`, but hipBLASLt appears to keep the resulting target shapes embedded in generated Tensile logic YAMLs rather than in a separate model-shape JSON.

### Notes

- The tuned HHS/BBS TN/NN large grids are nearly identical but not byte-for-byte identical. The union is safest for future NT target planning.
- Current NT HHS and TT HHS use the same small 470-shape grid. NT HHS has not been tuned on the large grid used by the newer TN/NN work.
- PR #7468 says the TN large-grid work improved `N<1000, M>500, K>500` by over 30% average on grid points. This is a useful priority slice within the saved union.

## PyTorch Workloads

### Row-Major PyTorch To Column-Major hipBLASLt

PyTorch tensors and default `torch.mm`/`torch.addmm`/`torch.nn.functional.linear` outputs are row-major. hipBLASLt and TensileLite use column-major GEMM conventions. PyTorch bridges this by using `(A @ B).T = B.T @ A.T`: for a default row-major output, the BLAS operands are swapped at the backend boundary.

For logical PyTorch `torch.mm(lhs, rhs)` with `lhs.shape == [m, k]`, `rhs.shape == [k, n]`, and default row-major output `[m, n]`, the hipBLASLt/TensileLite problem is:
- hipBLASLt `A` data comes from PyTorch `rhs`.
- hipBLASLt `B` data comes from PyTorch `lhs`.
- TensileLite exact shape is `[M=n, N=m, batch=1, K=k]`.

The transpose layout must be derived from the PyTorch tensor strides after this swap. In hipBLASLt stride terms, a row-major contiguous PyTorch 2D tensor is `T` layout, while a column-major-like tensor, such as a `.T` view of a contiguous tensor, is `N` layout. With a default row-major PyTorch output, the mapping is:

| PyTorch `lhs` stride layout | PyTorch `rhs` stride layout | TensileLite layout |
| --- | --- | --- |
| `T` row-major | `T` row-major | `NN` |
| `T` row-major | `N` column-major/view-transposed | `TN` |
| `N` column-major/view-transposed | `T` row-major | `NT` |
| `N` column-major/view-transposed | `N` column-major/view-transposed | `TT` |

This is the same convention encoded in `benchmark_mm_hipblaslt_fp16.py`: the benchmark remaps PyTorch providers with `TT -> NN`, `TN -> TN`, `NT -> NT`, and `NN -> TT` before applying `.T` views to the Python tensors.

TensileLite layout filenames map as follows:
- `NN`: `Cijk_Ailk_Bljk`, `TransposeA=False`, `TransposeB=False`.
- `NT`: `Cijk_Ailk_Bjlk`, `TransposeA=False`, `TransposeB=True`.
- `TN`: `Cijk_Alik_Bljk`, `TransposeA=True`, `TransposeB=False`.
- `TT`: `Cijk_Alik_Bjlk`, `TransposeA=True`, `TransposeB=True`.

### F.linear Mapping

`torch.nn.functional.linear(input, weight, bias)` computes `input @ weight.t()`, with `weight.shape == [out_features, in_features]`. PyTorch's native implementation uses `addmm(bias, input, weight.t())` for the 2D fused path and also flattens contiguous nD input for the likely-fusable bias path.

For default contiguous `input` and default contiguous `weight`:
- Let `B = prod(input.shape[:-1])`, the flattened row/token count.
- Let `K = input.shape[-1] = in_features`.
- Let `O = weight.shape[0] = out_features`.
- The logical PyTorch output is `[B, O]` after flattening, then reshaped back to `input.shape[:-1] + [O]`.
- The TensileLite exact shape is `[M=O, N=B, batch=1, K=K]`.
- The TensileLite layout is `TN` because PyTorch passes `rhs = weight.t()`, which is column-major-like, and `lhs = input`, which is row-major contiguous.
- The PyTorch bias length is `O`; for this default mapped problem it corresponds to the TensileLite `M` dimension.

For non-default strides, explicit `out=` tensors, missing/non-fusable bias paths, or higher-rank `matmul` paths that become batched GEMMs, re-derive the shape from the actual emitted 2D/batched GEMM. Do not assume the visible Python argument order is the TensileLite argument order.

### Direct hipBLASLt Wrapper

The local `mm_hipblaslt_fp16` wrapper is not PyTorch's row-major-facing `mm` path. It allocates a column-major output and calls hipBLASLt directly with `M=a.shape[0]`, `N=b.shape[1]`, and `K=a.shape[1]`. Its layout labels come directly from tensor strides, so do not apply the PyTorch A/B swap when mapping shapes for this wrapper.

### SDXL

Sources checked:
- ComfyUI inference/model code under `~/ComfyUI/comfy/ldm/modules/attention.py`, `~/ComfyUI/comfy/ldm/modules/diffusionmodules/openaimodel.py`, `~/ComfyUI/comfy/ops.py`, `~/ComfyUI/comfy/supported_models.py`, and `~/ComfyUI/comfy/model_detection.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/checkpoints/sdxl_anime/WAI-Illustrious-v17.safetensors`.

Assumptions for these shapes:
- Batch size is 1.
- Image sizes are `1024x1024` and `1536x1536`; SDXL latent sizes are therefore `128x128` and `192x192`.
- Text context length is 16 tokens.
- LoRA rank is 16.
- Counts and `K=B` activation-row dimensions are per single batch element and per single UNet/text-encoder call. Classifier-free-guidance batching or larger training batches multiply the flattened activation rows.

ComfyUI SDXL config and checkpoint facts:
- SDXL UNet config is `model_channels=320`, `channel_mult=[1,2,4]`, `num_res_blocks=[2,2,2]`, `context_dim=2048`, `num_head_channels=64`, `use_linear_in_transformer=True`.
- Transformer depth is `[0,0,2,2,10,10]` in input blocks, `10` in the middle block, and `[0,0,0,2,2,2,10,10,10]` in output blocks.
- Image-dependent UNet transformer blocks: 10 blocks at width 640 and latent downsample `ds=2`; 60 blocks at width 1280 and latent downsample `ds=4`.
- The checkpoint has UNet transformer linears with shapes/counts: `[640,640] x70`, `[5120,640] x10`, `[640,2560] x10`, `[640,2048] x20`, `[1280,1280] x372`, `[10240,1280] x60`, `[1280,5120] x60`, `[1280,2048] x120`. Counts include `SpatialTransformer` `proj_in/proj_out` for the square image-token linears.

Forward-pass NT check:
- No default PyTorch/ComfyUI SDXL forward NT GEMMs were found for the UNet transformer linears, CLIP text linears, or LoRA forward linears.
- `operations.Linear` ultimately uses `torch.nn.functional.linear`, so default contiguous weights map to TensileLite `TN`, not `NT`.
- LoRA forward is two default linear calls, down then up, so it also maps to `TN` for both matmuls.
- Matmuls inside the attention kernel are excluded from this SDXL shape set.
- Convolutions and VAE paths are not counted here as hipBLASLt/TensileLite GEMMs.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/sdxl_gemm_shapes.csv`, with assumptions and source paths in `sdxl_gemm_shapes_manifest.json`.

Summary of image-dependent UNet rows:
- Forward transformer linears are `TN`. For `1024x1024`, image-token activation rows are 4096 at width 640 and 1024 at width 1280. For `1536x1536`, they are 9216 and 2304. Cross-attention K/V linears use 16 text-token rows.
- Full-weight training creates `NT` parameter-gradient GEMMs for each trainable `Linear`. For a default `Linear(out,in)` with activation rows `B`, the mapped exact shape is `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors. The down factor maps to `[M=in, N=16, batch=1, K=B]`; the up factor maps to `[M=16, N=out, batch=1, K=B]`.

Summary of CLIP/text rows:
- CLIP-L has `[768,768] x48`, `[3072,768] x12`, and `[768,3072] x12` linears in the checkpoint.
- CLIP-G/OpenCLIP has combined qkv `in_proj_weight [3840,1280] x32`, plus `[1280,1280] x32`, `[5120,1280] x32`, and `[1280,5120] x32` linears.
- Their forward linears are also `TN`, not `NT`. If training CLIP weights or rank-16 CLIP LoRA, the exact 16-token `NT` parameter-gradient rows are in the saved CSV.

### Anima

Sources checked:
- ComfyUI inference/model code under `~/ComfyUI/comfy/ldm/anima/model.py`, `~/ComfyUI/comfy/ldm/cosmos/predict2.py`, `~/ComfyUI/comfy/ops.py`, `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, and `~/ComfyUI/comfy/model_base.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors`.

Assumptions match the SDXL section unless noted: batch size 1, image sizes `1024x1024` and `1536x1536`, 16 prompt tokens, LoRA rank 16, and attention-kernel matmuls excluded.

Anima config and checkpoint facts:
- The checkpoint is the 2048-channel Anima DiT variant: `model_channels=2048`, `num_blocks=28`, `num_heads=16`, `crossattn_emb_channels=1024`, `patch_spatial=2`, `patch_temporal=1`, `in_channels=16`, `out_channels=16`, `use_adaln_lora=True`, `adaln_lora_dim=256`.
- `x_embedder.proj.1.weight` is `[2048,68]`, matching 16 latent channels plus one padding-mask channel over a `2x2` spatial patch.
- For `1024x1024`, latent size is `128x128` and patch rows are 4096. For `1536x1536`, latent size is `192x192` and patch rows are 9216.
- The LLM adapter is part of the diffusion checkpoint. With 16 prompt tokens, it processes 16 target/source rows, then Anima pads its output to 512 tokens before the main DiT cross-attention consumes it.
- The external Qwen text encoder used by the workflow is a separate checkpoint and is not included in this Anima diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI Anima forward NT GEMMs were found for the main DiT linears, LLM-adapter linears, or LoRA forward linears.
- As with SDXL, `operations.Linear` uses default PyTorch row-major `linear`, so forward linears map to `TN`, not `NT`.
- Matmuls inside the attention kernel are excluded from this shape set.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/anima_gemm_shapes.csv`, with assumptions and source paths in `anima_gemm_shapes_manifest.json`.

Summary of main DiT rows:
- Main image-token linears use 4096 activation rows at `1024x1024` and 9216 rows at `1536x1536`.
- Main DiT cross-attention K/V linears use 512 context rows because of the LLM-adapter pad.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.

Summary of LLM-adapter rows:
- The adapter has six transformer blocks plus an output projection. Its Linear weights are `[1024,1024] x49`, `[4096,1024] x6`, and `[1024,4096] x6`. The `[32128,1024]` embedding is not a GEMM and is excluded.
- Adapter forward linears are `TN`; adapter full-weight or rank-16 LoRA training produces 16-token `NT` parameter-gradient rows saved in the CSV.

### Qwen

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/qwen_image/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/qwen_image.py`.
- Detection/loading/LoRA paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/lora.py`.
- Latent-shape sources under `~/ComfyUI/comfy/sample.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy_extras/nodes_sd3.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/diffusion_models/qwen_image_2512_bf16.safetensors`.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens, LoRA rank 16, no reference latents, and attention-kernel matmuls excluded.

Qwen config and checkpoint facts:
- The diffusion checkpoint is a 60-layer dual-stream Qwen Image DiT with inner width 3072, 24 attention heads, head dimension 128, and text input width 3584.
- Qwen uses the Wan21 latent format in ComfyUI. `EmptySD3LatentImage` creates a `/8` spatial latent with 16 channels; Qwen then patches by `2x2`, so `img_in.weight` is `[3072,64]`.
- For `1024x1024`, latent size is `128x128` and patch rows are 4096. For `1536x1536`, latent size is `192x192` and patch rows are 9216.
- The checkpoint has 846 2D Linear/GEMM weights: `[3072,3072] x481`, `[12288,3072] x120`, `[3072,12288] x120`, `[18432,3072] x120`, plus one each of `[3072,64]`, `[6144,3072]`, `[64,3072]`, `[3072,256]`, and `[3072,3584]`.
- The external Qwen text encoder checkpoint is not included in this diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI Qwen forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- As with SDXL and Anima, `operations.Linear` uses default PyTorch row-major `linear`, so forward linears map to `TN`, not `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/qwen_gemm_shapes.csv`, with assumptions and source paths in `qwen_gemm_shapes_manifest.json`.

Summary of Qwen rows:
- Per image size, the forward Linear rows sum to all 846 checkpoint Linear weights and use `TN`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.
- Image-stream rows use activation rows 4096 at `1024x1024` and 9216 at `1536x1536`; text-stream rows use 16 activation rows under the short-prompt assumption; timestep/modulation/final conditioning rows use one activation row.

### Z-Image

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/lumina/model.py`, which implements the Z-Image path through Lumina2/NextDiT.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/z_image.py`.
- Detection/loading/latent-format/omni-conditioning paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy_extras/nodes_zimage.py`.
- Remote safetensors header from `https://huggingface.co/Comfy-Org/z_image/blob/main/split_files/diffusion_models/z_image_bf16.safetensors`, fetched with HTTP Range requests only; tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens before model padding, LoRA rank 16, no omni/reference latents, no SigLIP reference features, external Qwen3-4B text encoder weights excluded, pixel-space Z-Image variant excluded, and attention-kernel matmuls excluded.

Z-Image config and checkpoint facts:
- The checkpoint is the latent-space Z-Image Lumina2/NextDiT path: hidden width 3840, 30 main transformer layers, 2 context-refiner layers, 2 noise-refiner layers, 30 attention heads, head dimension 128, FFN width 10240, text feature width 2560, and Z-Image modulation enabled.
- Z-Image uses the Flux latent format in ComfyUI: 16 latent channels at `/8` spatial downsample, then patches by `2x2`, so `x_embedder.weight` is `[3840,64]`.
- For `1024x1024`, image rows are 4096. For `1536x1536`, image rows are 9216. The checkpoint has `cap_pad_token` and `x_pad_token`, so Lumina2 pads the 16-token text context to 32 rows; main joint sequence rows are therefore 4128 and 9248.
- The remote BF16 checkpoint header has 453 tensors. It has 208 2D Linear/GEMM weights; `cap_pad_token [1,3840]` and `x_pad_token [1,3840]` are parameter rows, not GEMM weights, and are excluded.
- The included Linear/GEMM weight shapes are `[10240,3840] x68`, `[3840,10240] x34`, `[3840,3840] x34`, `[11520,3840] x34`, `[15360,256] x32`, plus one each of `[3840,64]`, `[3840,2560]`, `[1024,256]`, `[256,1024]`, `[3840,256]`, and `[64,3840]`.

Forward-pass NT check:
- No default PyTorch/ComfyUI Z-Image forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`; full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/z_image_gemm_shapes.csv`, with assumptions and source paths in `z_image_gemm_shapes_manifest.json`.

Summary of Z-Image rows:
- Per image size, the forward Linear rows sum to all 208 checkpoint Linear/GEMM weights and use `TN`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.
- `cap_embedder` rows use 16 activation rows; context-refiner rows use 32 padded text rows; noise-refiner rows use image rows 4096/9216; main DiT and final rows use padded text+image rows 4128/9248; timestep/AdaLN rows use one activation row.

### Krea 2

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/krea2/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/krea2.py`.
- Detection/loading/LoRA/latent-format paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy/utils.py`.
- Remote safetensors header from `https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors`, fetched with HTTP Range requests only; tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 text tokens after the Krea 2 template strip, LoRA rank 16, external Qwen3-VL-4B text encoder weights excluded, temporal/video path excluded, and attention-kernel matmuls excluded.

Krea 2 config and checkpoint facts:
- The diffusion checkpoint is a 28-layer single-stream MMDiT with hidden width 6144, 48 attention heads, 12 KV heads, head dimension 128, FFN width 16384, timestep width 256, and sigmoid-gated GQA attention.
- Krea 2 conditioning uses a Qwen3-VL-4B 12-layer tap flattened as `(B, seq, 12*2560)`, then unpacked and fused inside the DiT by `txtfusion`.
- The `txtfusion` adapter has 2 layerwise blocks over the 12 tapped hidden-state layers, a `[1,12]` projector, and 2 refiner blocks over text tokens. TextFusion width is 2560, has 20 heads, and uses FFN width 6912.
- Krea 2 uses ComfyUI's Wan21 latent format for this image path: 16 latent channels at `/8` spatial downsample, then patches by `2x2`, so `first.weight` is `[6144,64]`.
- For `1024x1024`, image rows are 4096 and main joint sequence rows are `16 + 4096 = 4112`. For `1536x1536`, image rows are 9216 and main joint sequence rows are `16 + 9216 = 9232`.
- TextFusion layerwise Linear rows flatten the 16 text tokens and 12 tapped layers to 192 activation rows. The TextFusion projector applies `Linear(12,1)` across `16 * 2560 = 40960` rows.
- The remote checkpoint header has 430 tensors: 256 BF16 tensors and 174 F32 tensors. It has 265 2D tensors; after excluding `last.modulation.lin [2,6144]` as an AdaLN parameter table, 264 2D Linear/GEMM weights remain.
- The included Linear/GEMM weight shapes are `[6144,6144] x86`, `[1536,6144] x56`, `[16384,6144] x56`, `[6144,16384] x28`, `[2560,2560] x20`, `[6912,2560] x8`, `[2560,6912] x4`, plus one each of `[6144,64]`, `[64,6144]`, `[6144,256]`, `[36864,6144]`, `[1,12]`, and `[6144,2560]`.

Forward-pass NT check:
- No default PyTorch/ComfyUI Krea 2 forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`; full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/krea2_gemm_shapes.csv`, with assumptions and source paths in `krea2_gemm_shapes_manifest.json`.

Summary of Krea 2 rows:
- Per image size, the forward Linear rows sum to all 264 checkpoint Linear/GEMM weights and use `TN`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.
- Main DiT rows use activation rows 4112/9232; image patch rows use 4096/9216; text refiner/text MLP rows use 16; TextFusion layerwise rows use 192; TextFusion projector rows use 40960; timestep/AdaLN rows use one activation row.

### Ideogram 4

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/ideogram4/model.py` and the shared Lumina `FeedForward` under `~/ComfyUI/comfy/ldm/lumina/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/ideogram4.py`.
- Detection/loading/latent-format paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/latent_formats.py`.
- Remote safetensors header from `https://huggingface.co/Comfy-Org/Ideogram-4/blob/main/diffusion_models/ideogram4_fp8_scaled.safetensors`, fetched with HTTP Range requests only; tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens, LoRA rank 16, external Qwen3-VL text encoder weights excluded, and attention-kernel matmuls excluded. Conditional packed text/image rows and image-only rows are alternative forward paths, not additive within a single diffusion call.

Ideogram 4 config and checkpoint facts:
- The diffusion checkpoint is a 34-layer single-stream Ideogram 4 DiT with hidden width 4608, 18 attention heads, head dimension 256, FFN width 12288, AdaLN width 512, and Qwen3-VL tapped text feature width 53248.
- Ideogram 4 uses the Flux2 latent format in ComfyUI: 128 packed latent channels at `/16` spatial downsample, representing a `2x2` patch over 32 VAE channels.
- For `1024x1024`, image rows are 4096; conditional packed sequence rows are `16 + 4096 = 4112`. For `1536x1536`, image rows are 9216; conditional packed sequence rows are `16 + 9216 = 9232`.
- The remote fp8/e4m3 checkpoint header has 880 tensors with `model_type=ideogram4_cond`. It has 212 2D weight tensors; after excluding `embed_image_indicator.weight [2,4608]` as an embedding table, 211 2D Linear/GEMM weights remain.
- The included Linear/GEMM weight shapes are `[4608,4608] x36`, `[12288,4608] x68`, `[4608,12288] x34`, `[13824,4608] x34`, `[18432,512] x34`, plus one each of `[4608,128]`, `[4608,53248]`, `[512,4608]`, `[4608,512]`, and `[128,4608]`.

Forward-pass NT check:
- No default PyTorch/ComfyUI Ideogram 4 forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`; full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/ideogram_4_gemm_shapes.csv`, with assumptions and source paths in `ideogram_4_gemm_shapes_manifest.json`.

Summary of Ideogram 4 rows:
- Per image size, the conditional forward Linear rows sum to 211 checkpoint Linear/GEMM weights and use `TN`; image-only rows sum to 210 because `llm_cond_proj` is skipped when `context is None`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.
- Main-sequence rows use activation rows 4112/9232 for conditional packed text/image calls and 4096/9216 for image-only calls; text-projection rows use 16 activation rows; timestep/AdaLN rows use one activation row.

### Klein 9B

Sources checked:
- ComfyUI Flux2 diffusion-model code under `~/ComfyUI/comfy/ldm/flux/model.py` and `~/ComfyUI/comfy/ldm/flux/layers.py`.
- Klein text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/flux.py`.
- Detection/loading/LoRA paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/lora.py`.
- Flux2 latent-shape source under `~/ComfyUI/comfy/latent_formats.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/diffusion_models/flux-2-klein-9b-bf16.safetensors`.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, LoRA rank 16, no reference latents in the primary CSV rows, and attention-kernel matmuls excluded.

Text-row note:
- The requested short-prompt simplification would be 16 tokens, but it is not reasonable for Klein 9B in ComfyUI. `KleinTokenizer8B` uses `Qwen3Tokenizer8B` with `min_length=512`, and `Flux2.extra_conds` also pads shorter cross-attention to 512 tokens.
- The Klein 9B rows therefore use 512 text activation rows.

Klein 9B config and checkpoint facts:
- The diffusion checkpoint is Flux2 with hidden width 4096, 8 double-stream blocks, 24 single-stream blocks, 32 attention heads, head dimension 128, context input width 12288, global modulation, `mlp_ratio=3.0`, and SiLU-gated MLP input projections.
- Flux2 latent format uses 128 latent channels at `/16` spatial downsample and `patch_size=1`.
- For `1024x1024`, latent/image rows are 4096 and single-block rows are `512 + 4096 = 4608`. For `1536x1536`, latent/image rows are 9216 and single-block rows are `512 + 9216 = 9728`.
- The checkpoint has 121 2D Linear/GEMM weights: `[36864,4096] x24`, `[4096,16384] x24`, `[24576,4096] x18`, `[4096,4096] x17`, `[12288,4096] x17`, `[4096,12288] x17`, plus one each of `[8192,4096]`, `[128,4096]`, `[4096,128]`, and `[4096,256]`.
- The external Qwen text encoder checkpoint is not included in this diffusion-model shape set.
- The local Klein image-edit script passes a reference latent. The primary CSV rows exclude reference latents; one same-size reference latent would add that reference token count to the image-stream and single-block activation rows.

Forward-pass NT check:
- No default PyTorch/ComfyUI Klein 9B forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`; full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/klein_9b_gemm_shapes.csv`, with assumptions and source paths in `klein_9b_gemm_shapes_manifest.json`.

Summary of Klein 9B rows:
- Per image size, the forward Linear rows sum to all 121 checkpoint Linear weights and use `TN`.
- Full-weight training creates 121 `NT` parameter-gradient GEMM rows per image size with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates 242 `NT` parameter-gradient rows per image size because both LoRA factors are included: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.

### Wan

Sources checked:
- ComfyUI Wan diffusion-model code under `~/ComfyUI/comfy/ldm/wan/model.py`.
- Wan text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/wan.py`.
- Detection/loading/LoRA paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/lora.py`.
- Latent-shape sources under `~/ComfyUI/comfy/latent_formats.py` and `~/ComfyUI/comfy_extras/nodes_hunyuan.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/diffusion_models/Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors`.

Assumptions: batch size 1, video sizes `640x480x40` and `1280x720x80`, LoRA rank 16, fp16-equivalent GEMM shapes despite the fp8/e5m2 checkpoint storage, Conv3d patch embedding excluded, fp8 scale tensors excluded, and attention-kernel matmuls excluded.

Text-row note:
- The requested short-prompt simplification would be 16 tokens, but it is not reasonable for Wan in ComfyUI. `UMT5XXlTokenizer` uses `min_length=512`.
- The Wan rows therefore use 512 text activation rows.

Wan config and checkpoint facts:
- The checkpoint has `in_dim=16`, `out_dim=16`, `patch_size=(1,2,2)`, hidden width 5120, 40 layers, 40 heads, head dimension 128, text width 4096, and FFN width 13824.
- Although the filename is Wan2.2, this low-noise checkpoint shape path is the 16-channel Wan latent path rather than the 48-channel Wan22 latent format.
- Wan latent rows use spatial `/8` and temporal `((frames - 1) // 4) + 1`, then patch by `(1,2,2)`.
- Exact video token rows are 12000 for `640x480x40` and 72000 for `1280x720x80`. The saved rows round these up to GEMM-friendly multiples of 256: 12032 and 72192.
- The checkpoint has 406 2D Linear/GEMM weights: `[5120,5120] x322`, `[13824,5120] x40`, `[5120,13824] x40`, plus one each of `[64,5120]`, `[5120,4096]`, `[5120,256]`, and `[30720,5120]`.
- The external UMT5 text encoder checkpoint is not included in this diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI Wan forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`; full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/wan_gemm_shapes.csv`, with assumptions and source paths in `wan_gemm_shapes_manifest.json`.

Summary of Wan rows:
- Per video size, the forward Linear rows sum to all 406 checkpoint Linear weights and use `TN`.
- Full-weight training creates 406 `NT` parameter-gradient GEMM rows per video size with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates 812 `NT` parameter-gradient rows per video size because both LoRA factors are included: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.

### LTX 2.3

Sources checked:
- ComfyUI Lightricks diffusion-model code under `~/ComfyUI/comfy/ldm/lightricks/model.py`, `~/ComfyUI/comfy/ldm/lightricks/av_model.py`, `~/ComfyUI/comfy/ldm/lightricks/embeddings_connector.py`, and `~/ComfyUI/comfy/ldm/lightricks/symmetric_patchifier.py`.
- LTX text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/lt.py`.
- Detection/loading/LoRA paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/lora.py`.
- Latent-shape source under `~/ComfyUI/comfy/latent_formats.py` and `~/ComfyUI/comfy_extras/nodes_lt.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/diffusion_models/ltx-2.3-22b-dev_transformer_only_bf16.safetensors`.

Assumptions: batch size 1, video sizes `640x480x40` and `1280x720x80`, no audio, LoRA rank 16, external text encoder weights excluded, and attention-kernel matmuls excluded.

Text-row note:
- The requested short-prompt simplification would be 16 tokens, but it is not reasonable for LTX 2.3 video context in ComfyUI. The video embeddings connector adds learnable register tokens up to at least 1024 context rows.
- The LTX 2.3 rows therefore use 1024 text/context activation rows.

LTX 2.3 config and checkpoint facts:
- The checkpoint metadata identifies `model_version=2.3.0` and `_class_name=AVTransformer3DModel`. It is detected as `ltxav`, but this shape set documents the requested no-audio video workload.
- No-audio rows exclude audio-only modules, AV cross-attention modules, `audio_embeddings_connector.*`, and `video_embeddings_connector.learnable_registers` because the latter is a parameter table rather than a Linear/GEMM weight.
- The video branch has hidden width 4096, 48 transformer blocks, 32 attention heads, head dimension 128, FFN width 16384, caption projection width 4096, and 8 video-connector transformer blocks.
- LTX latents use 128 channels, spatial `/32`, and temporal `((frames - 1) // 8) + 1`; patch size is 1.
- Exact video token rows before GEMM rounding are 1500 for `640x480x40` and 8800 for `1280x720x80` using Comfy's integer-division latent sizing. The 720p case uses 22 latent height rows because 720 is not divisible by 32. The saved rows round up to GEMM-friendly multiples of 256: 1536 and 8960.
- The no-audio checkpoint subset has 640 2D Linear/GEMM weights: `[4096,4096] x418`, `[32,4096] x104`, `[4096,16384] x56`, `[16384,4096] x56`, plus one each of `[128,4096]`, `[4096,128]`, `[8192,4096]`, `[36864,4096]`, and two `[4096,256]` timestep linears.

Forward-pass NT check:
- No default PyTorch/ComfyUI LTX 2.3 forward NT GEMMs were found for the no-audio diffusion-model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`; full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/ltx_2_3_gemm_shapes.csv`, with assumptions and source paths in `ltx_2_3_gemm_shapes_manifest.json`.

Summary of LTX 2.3 rows:
- Per video size, the forward Linear rows sum to all 640 included no-audio Linear weights and use `TN`.
- Full-weight training creates 640 `NT` parameter-gradient GEMM rows per video size with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates 1280 `NT` parameter-gradient rows per video size because both LoRA factors are included: down maps to `[M=in, N=16, batch=1, K=B]`; up maps to `[M=16, N=out, batch=1, K=B]`.

## Recommended NT HHS Tuning Plan

### Normalized Workload CSV Schema

All model workload CSVs in `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhs/shape_data/*_gemm_shapes.csv` use the same columns:

`dataset,component,phase,op_kind,input_size,activation_rows_kind,layout,m,n,batch,k,count,weight_out,weight_in,activation_rows,rank,source,notes`

The exact TensileLite shape columns remain `m,n,batch,k` in `[M,N,batch,K]` order. The `count` column is the number of model weights represented by that aggregate row.

### Planning Summary

The ComfyUI workloads analyzed here produce no default forward `NT` GEMMs for the model Linears; forward Linears are `TN`. The `NT` tuning target is therefore mainly full-weight and LoRA parameter-gradient GEMMs.

Aggregating `NT` rows from SDXL, Anima, Qwen, Z-Image, Krea 2, Ideogram 4, Klein 9B, Wan, and LTX 2.3 gives:
- 637 normalized CSV rows.
- 406 unique exact `NT` shapes.
- 25,275 weighted `NT` shape occurrences after applying each row's `count`.
- Current NT HHS exact table covers only 6 of those unique shapes, or 1,060 weighted occurrences.
- The large-grid target union covers 62 of those unique shapes, or 5,828 weighted occurrences.

Important workload dimensions not covered by the current large-grid dimension sets:
- Extra `M` values: `[68, 1280, 2560, 3584, 3840, 4608, 5120, 6144, 6912, 10240, 12288, 13824, 16384, 53248]`.
- Extra `N` values: `[4608, 5120, 6144, 6912, 10240, 11520, 12288, 13824, 15360, 16384, 18432, 24576, 30720, 36864]`.
- Extra `K` activation-row values: `[192, 1536, 2304, 4112, 4128, 4608, 8960, 9216, 9232, 9248, 9728, 12032, 40960, 72192]`.

The `M` extensions matter for `NT` specifically. Values like `1280` exist as `N` in the inherited large grid, but `NT` parameter gradients often put feature/input dimensions on `M`, so an `N`-only value is not enough.

### Recommended Shape Set

Tune NT HHS on the union of these tiers:

1. Base large grid: all 9,681 shapes from `large_grid_target_union_shapes.csv`. This brings NT HHS to the same broad coverage already used by the tuned HHS/BBS TN/NN work.
2. Exact workload misses: all unique `NT` shapes from the normalized `*_gemm_shapes.csv` files that are not already in the large-grid target union. This adds 344 shapes and gives exact coverage for the two analyzed sizes per model.
3. Banded workload anchors: add K-banded variants of the workload shapes so nearby image/video sizes can choose a sensible tuned solution instead of falling back to a distant exact shape. Use this K mapping for the workload-derived rows: keep `1`, `16`, `32`, `192`, `512`, `1024`, `1536`, `2304`, `4096`, `9216`, and `40960`; map `4112`, `4128`, and `4608 -> 4096`; map `8960`, `9232`, `9248`, and `9728 -> 9216`; map `12032 -> 12288`; map `72192 -> 73728`.

The combined recommended set has 10,072 unique exact shapes: the 9,681-shape large grid plus 391 workload/banded additions.

Do not expand the workload extensions into a full Cartesian product of every observed `M`, `N`, and `K`. Keep the workload `M,N` pairs from the CSV rows and only add the K-banded variants above. The existing large grid already provides broad canonical coverage; the workload extension is for high-value missing feature pairs and token-row bands.

### Priority If Tuning Budget Is Limited

Minimum viable NT HHS update:
- Tune the 9,681-shape large-grid target union first.
- Add the high-weight workload misses with `K in {1536, 2304, 4096, 4112, 4128, 9216, 9232, 9248, 12288, 73728}`.
- Add skinny LoRA rank-16 shapes where `M=16` or `N=16` for feature sizes `[1280, 1536, 2048, 2560, 3072, 3840, 4096, 4608, 5120, 6144, 6912, 10240, 11520, 12288, 13824, 15360, 16384, 18432, 24576, 36864]`.

High-priority K bands:
- `K=1536`: LTX 2.3 small-video rows; especially `(M,N)` pairs `(4096,16)`, `(16,4096)`, `(4096,4096)`, `(4096,32)`, `(16,32)`, `(16384,4096)`, and `(4096,16384)`.
- `K=2304`: SDXL `1536x1536` mid-resolution rows; especially `(16,1280)`, `(1280,16)`, `(1280,1280)`, `(1280,10240)`, and `(5120,1280)`.
- `K=4096`/`K=4112`/`K=4128`: Ideogram 4, Z-Image, and Krea 2 `1024x1024` rows; especially `(16,3840)`, `(3840,16)`, `(3840,3840)`, `(3840,10240)`, `(10240,3840)`, `(3840,11520)`, `(16,4608)`, `(4608,4608)`, `(12288,4608)`, `(16,6144)`, `(6144,6144)`, `(6144,16384)`, and `(16384,6144)`.
- `K=9216`/`K=9232`/`K=9248`: high-resolution image rows, nearby LTX/Klein rows, and Ideogram 4/Z-Image/Krea 2 conditional rows; especially `(16,3072)`, `(3072,16)`, `(3072,3072)`, `(4096,4096)`, `(4608,4608)`, `(12288,4608)`, `(3840,3840)`, `(10240,3840)`, `(6144,6144)`, and `(16384,6144)`.
- `K=12288`: Wan `640x480x40` video band; especially `(5120,16)`, `(16,5120)`, `(5120,5120)`, `(5120,13824)`, and `(13824,5120)`.
- `K=73728`: Wan `1280x720x80` video band; use the same Wan feature pairs as `K=12288`.

Lower-priority or deferrable items:
- Exact `K=4112`/`K=9232` Ideogram 4 and Krea 2 rows, `K=4128`/`K=9248` Z-Image rows, and `K=4608`/`K=9728` Klein rows can be covered by nearby `K=4096` and `K=9216` anchors if tuning budget is tight.
- Small-token Z-Image context-refiner rows with `K=32`, Krea 2 TextFusion layer-stack rows with `K=192`, Krea 2 projector rows with `K=40960`, and timestep/global rows with `K=1` are low arithmetic work or very skinny. Keep them in the full recommended set, but deprioritize them in a constrained tuning run after the larger token-row bands are covered.
- Very low-count odd feature sizes such as `M=68`, `M=3584`, `M=53248`, `N=30720`, and exact patch/unpatch one-offs can be deferred unless profiling shows them hot.
