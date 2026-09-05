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

Saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/`:
- `current_nt_hhs_shapes.csv`: current NT HHS unique exact shapes.
- `current_nt_hhs_aux_shapes.csv`: current NT HHS AuxH unique exact shapes.
- `large_grid_tn_hhs_shapes_ordered.csv`: ordered exact rows from the tuned TN HHS main GridBased YAML.
- `large_grid_nn_hhs_shapes_ordered_with_duplicates.csv`: ordered exact rows from the tuned NN HHS main GridBased YAML, preserving duplicate rows.
- `large_grid_target_union_shapes.csv`: unique target large-grid union from tuned HHS/BBS TN/NN sources.
- `large_grid_target_union_decomposition.json`: exact non-overlapping product-block decomposition of `large_grid_target_union_shapes.csv`. 35 direct-product blocks reconstruct all 9,681 shapes with zero missing or extra rows.
- `large_grid_target_missing_from_current_nt_hhs.csv`: target large-grid shapes that current NT HHS does not cover.
- `current_nt_hhs_not_in_large_grid_target.csv`: current NT HHS shapes outside the large-grid target union.
- `current_nt_hhs_intersection_large_grid_target.csv`: overlap between current NT HHS and target union.
- `per_k_summary.csv`: per-K compact summary for current NT, target union, and TN HHS.
- `manifest.json`: source-file metadata, dimensions, counts-by-K, and duplicate details.
- `h3_gemm_shapes.csv`: normalized H3 forward, full-weight-gradient, and theoretical rank-16 LoRA rows.
- `h3_gemm_shapes_manifest.json`: H3 geometry, pruned-AdaLN assumptions, checkpoint shape counts, and layout notes.
- `tn_gemm_shapes.csv` and `tn_gemm_shapes_manifest.json`: derived base and unmerged rank-16 LoRA forward TN rows, with training-layout overrides.
- `nn_gemm_shapes.csv` and `nn_gemm_shapes_manifest.json`: derived base and rank-16 LoRA input-gradient NN rows, including the Krea 2 batched alternate.
- `qwen_image_1_gemm_shapes.csv` and `qwen_image_1_gemm_shapes_manifest.json`: the legacy Qwen-Image 1 dataset, renamed from `qwen_gemm_shapes.csv`.
- `qwen_image_21_gemm_shapes.csv` and `qwen_image_21_gemm_shapes_manifest.json`: Qwen-Image 2.1 forward, prefix-cache, full-weight-gradient, and theoretical rank-16 LoRA rows.
- `moss_image_gemm_shapes.csv` and `moss_image_gemm_shapes_manifest.json`: MOSS-Image forward (unfused and fused-`qkv`), TT in-projection, full-weight-gradient, and theoretical rank-16 LoRA rows.

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
- The PyTorch bias length is `O`. For this default mapped problem it corresponds to the TensileLite `M` dimension.

For non-default strides, explicit `out=` tensors, missing/non-fusable bias paths, or higher-rank `matmul` paths that become batched GEMMs, re-derive the shape from the actual emitted 2D/batched GEMM. Do not assume the visible Python argument order is the TensileLite argument order.

### Direct hipBLASLt Wrapper

The local `mm_hipblaslt_fp16` wrapper is not PyTorch's row-major-facing `mm` path. It allocates a column-major output and calls hipBLASLt directly with `M=a.shape[0]`, `N=b.shape[1]`, and `K=a.shape[1]`. Its layout labels come directly from tensor strides, so do not apply the PyTorch A/B swap when mapping shapes for this wrapper.

### SDXL

Sources checked:
- ComfyUI inference/model code under `~/ComfyUI/comfy/ldm/modules/attention.py`, `~/ComfyUI/comfy/ldm/modules/diffusionmodules/openaimodel.py`, `~/ComfyUI/comfy/ops.py`, `~/ComfyUI/comfy/supported_models.py`, and `~/ComfyUI/comfy/model_detection.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/checkpoints/sdxl_anime/WAI-Illustrious-v17.safetensors`.

Assumptions for these shapes:
- Batch size is 1.
- Image sizes are `1024x1024` and `1536x1536`. SDXL latent sizes are therefore `128x128` and `192x192`.
- Text context length is 16 tokens.
- LoRA rank is 16.
- Counts and `K=B` activation-row dimensions are per single batch element and per single UNet/text-encoder call. Classifier-free-guidance batching or larger training batches multiply the flattened activation rows.

ComfyUI SDXL config and checkpoint facts:
- SDXL UNet config is `model_channels=320`, `channel_mult=[1,2,4]`, `num_res_blocks=[2,2,2]`, `context_dim=2048`, `num_head_channels=64`, `use_linear_in_transformer=True`.
- Transformer depth is `[0,0,2,2,10,10]` in input blocks, `10` in the middle block, and `[0,0,0,2,2,2,10,10,10]` in output blocks.
- Image-dependent UNet transformer blocks: 10 blocks at width 640 and latent downsample `ds=2`. 60 blocks at width 1280 and latent downsample `ds=4`.
- The checkpoint has UNet transformer linears with shapes/counts: `[640,640] x70`, `[5120,640] x10`, `[640,2560] x10`, `[640,2048] x20`, `[1280,1280] x372`, `[10240,1280] x60`, `[1280,5120] x60`, `[1280,2048] x120`. Counts include `SpatialTransformer` `proj_in/proj_out` for the square image-token linears.

Forward-pass NT check:
- No default PyTorch/ComfyUI SDXL forward NT GEMMs were found for the UNet transformer linears, CLIP text linears, or LoRA forward linears.
- `operations.Linear` ultimately uses `torch.nn.functional.linear`, so default contiguous weights map to TensileLite `TN`, not `NT`.
- LoRA forward is two default linear calls, down then up, so it also maps to `TN` for both matmuls.
- Matmuls inside the attention kernel are excluded from this SDXL shape set.
- Convolutions and VAE paths are not counted here as hipBLASLt/TensileLite GEMMs.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/sdxl_gemm_shapes.csv`, with assumptions and source paths in `sdxl_gemm_shapes_manifest.json`.

Summary of image-dependent UNet rows:
- Forward transformer linears are `TN`. For `1024x1024`, image-token activation rows are 4096 at width 640 and 1024 at width 1280. For `1536x1536`, they are 9216 and 2304. Cross-attention K/V linears use 16 text-token rows.
- Full-weight training creates `NT` parameter-gradient GEMMs for each trainable `Linear`. For a default `Linear(out,in)` with activation rows `B`, the mapped exact shape is `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors. The down factor maps to `[M=in, N=16, batch=1, K=B]`. The up factor maps to `[M=16, N=out, batch=1, K=B]`.

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
- The external Qwen3-VL text encoder used by the workflow is a separate checkpoint and is not included in this Anima diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI Anima forward NT GEMMs were found for the main DiT linears, LLM-adapter linears, or LoRA forward linears.
- As with SDXL, `operations.Linear` uses default PyTorch row-major `linear`, so forward linears map to `TN`, not `NT`.
- Matmuls inside the attention kernel are excluded from this shape set.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/anima_gemm_shapes.csv`, with assumptions and source paths in `anima_gemm_shapes_manifest.json`.

Summary of main DiT rows:
- Main image-token linears use 4096 activation rows at `1024x1024` and 9216 rows at `1536x1536`.
- Main DiT cross-attention K/V linears use 512 context rows because of the LLM-adapter pad.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.

Summary of LLM-adapter rows:
- The adapter has six transformer blocks plus an output projection. Its Linear weights are `[1024,1024] x49`, `[4096,1024] x6`, and `[1024,4096] x6`. The `[32128,1024]` embedding is not a GEMM and is excluded.
- Adapter forward linears are `TN`. Adapter full-weight or rank-16 LoRA training produces 16-token `NT` parameter-gradient rows saved in the CSV.

### Qwen-Image 1

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/qwen_image/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/qwen_image.py`.
- Detection/loading/LoRA paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/lora.py`.
- Latent-shape sources under `~/ComfyUI/comfy/sample.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy_extras/nodes_sd3.py`.
- Checkpoint tensor shapes from `~/ComfyUI/models/diffusion_models/qwen_image_2512_bf16.safetensors`.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens, LoRA rank 16, no reference latents, and attention-kernel matmuls excluded.

Qwen-Image 1 config and checkpoint facts:
- The diffusion checkpoint is a 60-layer dual-stream Qwen-Image 1 DiT with inner width 3072, 24 attention heads, head dimension 128, and text input width 3584.
- Qwen-Image 1 uses the Wan21 latent format in ComfyUI. `EmptySD3LatentImage` creates a `/8` spatial latent with 16 channels. Qwen-Image 1 then patches by `2x2`, so `img_in.weight` is `[3072,64]`.
- For `1024x1024`, latent size is `128x128` and patch rows are 4096. For `1536x1536`, latent size is `192x192` and patch rows are 9216.
- The checkpoint has 846 2D Linear/GEMM weights: `[3072,3072] x481`, `[12288,3072] x120`, `[3072,12288] x120`, `[18432,3072] x120`, plus one each of `[3072,64]`, `[6144,3072]`, `[64,3072]`, `[3072,256]`, and `[3072,3584]`.
- The external Qwen2.5-VL-7B text encoder checkpoint is not included in this diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI Qwen-Image 1 forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- As with SDXL and Anima, `operations.Linear` uses default PyTorch row-major `linear`, so forward linears map to `TN`, not `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/qwen_image_1_gemm_shapes.csv`, with assumptions and source paths in `qwen_image_1_gemm_shapes_manifest.json`.

Summary of Qwen-Image 1 rows:
- Per image size, the forward Linear rows sum to all 846 checkpoint Linear weights and use `TN`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.
- Image-stream rows use activation rows 4096 at `1024x1024` and 9216 at `1536x1536`. Text-stream rows use 16 activation rows under the short-prompt assumption. Timestep/modulation/final conditioning rows use one activation row.

### Qwen-Image 2.1

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/qwen_image21/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/qwen_image21.py`.
- Detection/loading/caching paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy_extras/nodes_qwen.py`.
- Latent construction under `~/ComfyUI/comfy/sample.py` and `~/ComfyUI/nodes.py`.
- Remote safetensors headers from `https://huggingface.co/Comfy-Org/Qwen-Image-2.1`, fetched with HTTP Range requests only. Tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens, LoRA rank 16, no reference latents, no Fun-Control stream, and attention-kernel matmuls excluded.

Qwen-Image 2.1 config and checkpoint facts:
- The diffusion checkpoint is a 32-layer single-stream DiT with inner width 4096, 32 attention heads, head dimension 128, text input width 4096, and `mlp_ratio=3`.
- The latent format is 64 channels at `/16`. `EmptyLatentImage` emits `[1,4,H/8,W/8]` and `fix_empty_latent_channels` repeats the channels to 64 and rescales the spatial axes by `8/16`, so the DiT sees `[1,64,H/16,W/16]`. The model consumes that latent directly with no extra patching, so `img_in.weight` is `[4096,64]`.
- For `1024x1024` the latent is `64x64` and image rows are 4096. For `1536x1536` the latent is `96x96` and image rows are 9216. Native 2K is also supported: `2048x2048` gives a `128x128` latent and 16384 image rows.
- The checkpoint has 200 2D Linear/GEMM weights: `[4096,4096] x132`, `[24576,4096] x32`, `[4096,12288] x32`, plus one each of `[4096,64]`, `[16384,4096]`, `[64,4096]`, and `[4096,256]`.
- The 32 blocks contribute 192 of those weights: `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, the fused `img_mlp.gate_up = Linear(4096,24576)`, and `img_mlp.out = Linear(12288,4096)`. The model uses one shared `modulation = Sequential(SiLU, Linear(4096,16384))` outside the blocks rather than per-block modulation linears.
- The repackaged `qwen_image_2.1_int8_convrot.safetensors` variant stores the same 200 logical shapes but quantizes 192 of them as per-row-scaled INT8 with `comfy_quant` metadata. Its rows are recorded as logical dequantized BF16-equivalent shapes.
- Transparent RGBA is supported end to end, but it does not add a DiT channel. The released VAE is `AutoencoderKLQwenImage21` with `in_channels=4`, `out_channels=4`, `z_dim=64`, and `scale_factor_spatial=16`, so the alpha plane rides inside the same 64 latent channels that `img_in` and `proj_out` already consume. ComfyUI detects it as `Qwen Image 2.1 VAE: Wan 2.2 layout, temporal kernel 1, no patchify, RGBA` with `pad_channel_value=1.0`, so a 3-channel input is padded with opaque alpha and decode returns 4 channels for an RGBA PNG. See the RGBA note below.
- The external Qwen3-VL-8B text encoder checkpoint is not included in this diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI Qwen-Image 2.1 forward NT GEMMs were found. Every Linear receives either a contiguous packed activation (`TN`) or, for `img_in`, the `flatten(2).transpose(1,2)` token view, which is column-major-like and therefore `TT`.
- `img_in` and an unmerged rank-16 `lora_down` on it stay `TT` in inference and training, including when the weight requires gradients. This was verified with a `torch` dispatch probe. That case is documented in `input_shapes_tt.md`.
- Attention is block-causal over the packed sequence: each block issues a 16-token causal text segment and a target segment whose keys span the whole prefix. Fused-attention rows are documented in the attention shape note.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/qwen_image_21_gemm_shapes.csv`, with assumptions, the attention segment table, and source paths in `qwen_image_21_gemm_shapes_manifest.json`.

Summary of Qwen-Image 2.1 rows:
- Per image size, the forward Linear rows sum to all 200 checkpoint Linear weights: 199 `TN` weights and the `TT` `img_in`.
- Full-weight training creates 200 `NT` parameter-gradient GEMM rows per image size.
- Rank-16 LoRA on every logical Linear creates 400 `NT` factor-gradient rows per image size: down maps to `[M=in, N=16, batch=1, K=rows]` and up maps to `[M=16, N=out, batch=1, K=rows]`. A block-scoped target pattern such as a `QwenImage21TransformerBlock` walk would cover 192 of them.
- Block attention and MLP rows use the packed activation rows `16+4096=4112` at `1024x1024` and `16+9216=9232` at `1536x1536`. Text projections use 16 rows, the two timestep-embedder linears and the shared modulation use 2 rows, `norm_out.linear` uses 1 row, and `proj_out` uses the image rows.
- Per image size the `NT` rows collapse to 26 unique exact shapes and 600 weighted occurrences, so 1200 weighted occurrences across both sizes. Of the 39 unique shapes, 12 already exist in the large-grid target union. The other 27 are outside it, and 21 of those are new relative to the whole workload union. None are covered by the current NT HHS table.
- Two extra non-additive `TN` alternate rows per block Linear come from the default prefix K/V cache, which splits the first sampling step into a 16-row prefix pass and an image-row target pass. Those are recorded in `tn_gemm_shapes.csv` and `input_shapes_tn.md`.

RGBA impact:
- No GEMM or fused-attention shape changes. The DiT keeps `in_channels=64` and `out_channels=64`, so `img_in = [4096,64]`, `proj_out = [64,4096]`, the `[1,64,H/16,W/16]` latent, every block row band, and the attention extents are identical for RGB and RGBA. Transparency is a latent channel, not an extra mask or input plane, so nothing about the 200 Linear weights changes.
- What does change is outside the DiT: the VAE is 4-channel in and out, and ComfyUI pads a 3-channel image with alpha `1.0` before encoding, so RGB and RGBA share one pixel domain.
- Row counts only move because the recommended RGBA prompt is longer. The transparency recipe is a prompt wrapper, not a separate input mode, so transparency is produced by text conditioning. The ComfyUI-visible text span tokenizes to exactly 8 fixed tokens plus the prompt (measured with the released tokenizer: 8 rows for an empty prompt, 10 for `a cat`, and 32 for the documented RGBA wrapper around `a cat`). A wrapper like that therefore moves the text rows from the 16-token convention to roughly 30-40 and grows the packed rows and the attention text segment by the same delta.
- Reference images may carry alpha. The vision tower sees the alpha composited over white, while the VAE keeps all four channels. Reference latents use the same 64-channel latent and the same row geometry, and they are excluded from this no-reference workload anyway.

### Z-Image

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/lumina/model.py`, which implements the Z-Image path through Lumina2/NextDiT.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/z_image.py`.
- Detection/loading/latent-format/omni-conditioning paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy_extras/nodes_zimage.py`.
- Remote safetensors header from `https://huggingface.co/Comfy-Org/z_image/blob/main/split_files/diffusion_models/z_image_bf16.safetensors`, fetched with HTTP Range requests only. Tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens before model padding, LoRA rank 16, no omni/reference latents, no SigLIP reference features, external Qwen3-4B text encoder weights excluded, pixel-space Z-Image variant excluded, and attention-kernel matmuls excluded.

Z-Image config and checkpoint facts:
- The checkpoint is the latent-space Z-Image Lumina2/NextDiT path: hidden width 3840, 30 main transformer layers, 2 context-refiner layers, 2 noise-refiner layers, 30 attention heads, head dimension 128, FFN width 10240, text feature width 2560, and Z-Image modulation enabled.
- Z-Image uses the Flux latent format in ComfyUI: 16 latent channels at `/8` spatial downsample, then patches by `2x2`, so `x_embedder.weight` is `[3840,64]`.
- For `1024x1024`, image rows are 4096. For `1536x1536`, image rows are 9216. The checkpoint has `cap_pad_token` and `x_pad_token`, so Lumina2 pads the 16-token text context to 32 rows. Main joint sequence rows are therefore 4128 and 9248.
- The remote BF16 checkpoint header has 453 tensors. It has 208 2D Linear/GEMM weights. `cap_pad_token [1,3840]` and `x_pad_token [1,3840]` are parameter rows, not GEMM weights, and are excluded.
- The included Linear/GEMM weight shapes are `[10240,3840] x68`, `[3840,10240] x34`, `[3840,3840] x34`, `[11520,3840] x34`, `[15360,256] x32`, plus one each of `[3840,64]`, `[3840,2560]`, `[1024,256]`, `[256,1024]`, `[3840,256]`, and `[64,3840]`.

Forward-pass NT check:
- No default PyTorch/ComfyUI Z-Image forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`. Full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/z_image_gemm_shapes.csv`, with assumptions and source paths in `z_image_gemm_shapes_manifest.json`.

Summary of Z-Image rows:
- Per image size, the forward Linear rows sum to all 208 checkpoint Linear/GEMM weights and use `TN`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.
- `cap_embedder` rows use 16 activation rows. Context-refiner rows use 32 padded text rows. Noise-refiner rows use image rows 4096/9216. Main DiT and final rows use padded text+image rows 4128/9248. Timestep/AdaLN rows use one activation row.

### Krea 2

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/krea2/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/krea2.py`.
- Detection/loading/LoRA/latent-format paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy/utils.py`.
- Remote safetensors header from `https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors`, fetched with HTTP Range requests only. Tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 text tokens after the Krea 2 template strip, LoRA rank 16, external Qwen3-VL-4B text encoder weights excluded, temporal/video path excluded, and attention-kernel matmuls excluded.

Krea 2 config and checkpoint facts:
- The diffusion checkpoint is a 28-layer single-stream MMDiT with hidden width 6144, 48 attention heads, 12 KV heads, head dimension 128, FFN width 16384, timestep width 256, and sigmoid-gated GQA attention.
- Krea 2 conditioning uses a Qwen3-VL-4B 12-layer tap flattened as `(B, seq, 12*2560)`, then unpacked and fused inside the DiT by `txtfusion`.
- The `txtfusion` adapter has 2 layerwise blocks over the 12 tapped hidden-state layers, a `[1,12]` projector, and 2 refiner blocks over text tokens. TextFusion width is 2560, has 20 heads, and uses FFN width 6912.
- Krea 2 uses ComfyUI's Wan21 latent format for this image path: 16 latent channels at `/8` spatial downsample, then patches by `2x2`, so `first.weight` is `[6144,64]`.
- For `1024x1024`, image rows are 4096 and main joint sequence rows are `16 + 4096 = 4112`. For `1536x1536`, image rows are 9216 and main joint sequence rows are `16 + 9216 = 9232`.
- TextFusion layerwise Linear rows flatten the 16 text tokens and 12 tapped layers to 192 activation rows. The TextFusion projector applies `Linear(12,1)` across `16 * 2560 = 40960` rows.
- The remote checkpoint header has 430 tensors: 256 BF16 tensors and 174 F32 tensors. It has 265 2D tensors. After excluding `last.modulation.lin [2,6144]` as an AdaLN parameter table, 264 2D Linear/GEMM weights remain.
- The included Linear/GEMM weight shapes are `[6144,6144] x86`, `[1536,6144] x56`, `[16384,6144] x56`, `[6144,16384] x28`, `[2560,2560] x20`, `[6912,2560] x8`, `[2560,6912] x4`, plus one each of `[6144,64]`, `[64,6144]`, `[6144,256]`, `[36864,6144]`, `[1,12]`, and `[6144,2560]`.

Forward-pass NT check:
- No default PyTorch/ComfyUI Krea 2 forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`. Full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/krea2_gemm_shapes.csv`, with assumptions and source paths in `krea2_gemm_shapes_manifest.json`.

Summary of Krea 2 rows:
- Per image size, the forward Linear rows sum to all 264 checkpoint Linear/GEMM weights and use `TN`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.
- Main DiT rows use activation rows 4112/9232. Image patch rows use 4096/9216. Text refiner/text MLP rows use 16. TextFusion layerwise rows use 192. TextFusion projector rows use 40960. Timestep/AdaLN rows use one activation row.

### Ideogram 4

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/ideogram4/model.py` and the shared Lumina `FeedForward` under `~/ComfyUI/comfy/ldm/lumina/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/ideogram4.py`.
- Detection/loading/latent-format paths under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, and `~/ComfyUI/comfy/latent_formats.py`.
- Remote safetensors header from `https://huggingface.co/Comfy-Org/Ideogram-4/blob/main/diffusion_models/ideogram4_fp8_scaled.safetensors`, fetched with HTTP Range requests only. Tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 diffusion text tokens, LoRA rank 16, external Qwen3-VL text encoder weights excluded, and attention-kernel matmuls excluded. Conditional packed text/image rows and image-only rows are alternative forward paths, not additive within a single diffusion call.

Ideogram 4 config and checkpoint facts:
- The diffusion checkpoint is a 34-layer single-stream Ideogram 4 DiT with hidden width 4608, 18 attention heads, head dimension 256, FFN width 12288, AdaLN width 512, and Qwen3-VL tapped text feature width 53248.
- Ideogram 4 uses the Flux2 latent format in ComfyUI: 128 packed latent channels at `/16` spatial downsample, representing a `2x2` patch over 32 VAE channels.
- For `1024x1024`, image rows are 4096. Conditional packed sequence rows are `16 + 4096 = 4112`. For `1536x1536`, image rows are 9216. Conditional packed sequence rows are `16 + 9216 = 9232`.
- The remote fp8/e4m3 checkpoint header has 880 tensors with `model_type=ideogram4_cond`. It has 212 2D weight tensors. After excluding `embed_image_indicator.weight [2,4608]` as an embedding table, 211 2D Linear/GEMM weights remain.
- The included Linear/GEMM weight shapes are `[4608,4608] x36`, `[12288,4608] x68`, `[4608,12288] x34`, `[13824,4608] x34`, `[18432,512] x34`, plus one each of `[4608,128]`, `[4608,53248]`, `[512,4608]`, `[4608,512]`, and `[128,4608]`.

Forward-pass NT check:
- No default PyTorch/ComfyUI Ideogram 4 forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`. Full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/ideogram_4_gemm_shapes.csv`, with assumptions and source paths in `ideogram_4_gemm_shapes_manifest.json`.

Summary of Ideogram 4 rows:
- Per image size, the conditional forward Linear rows sum to 211 checkpoint Linear/GEMM weights and use `TN`. Image-only rows sum to 210 because `llm_cond_proj` is skipped when `context is None`.
- Full-weight training creates `NT` parameter-gradient GEMMs with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates `NT` parameter-gradient GEMMs for both LoRA factors: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.
- Main-sequence rows use activation rows 4112/9232 for conditional packed text/image calls and 4096/9216 for image-only calls. Text-projection rows use 16 activation rows. Timestep/AdaLN rows use one activation row.

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
- The external Qwen3-8B text encoder checkpoint is not included in this diffusion-model shape set.
- The local Klein image-edit script passes a reference latent. The primary CSV rows exclude reference latents. One same-size reference latent would add that reference token count to the image-stream and single-block activation rows.

Forward-pass NT check:
- No default PyTorch/ComfyUI Klein 9B forward NT GEMMs were found for the diffusion model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`. Full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/klein_9b_gemm_shapes.csv`, with assumptions and source paths in `klein_9b_gemm_shapes_manifest.json`.

Summary of Klein 9B rows:
- Per image size, the forward Linear rows sum to all 121 checkpoint Linear weights and use `TN`.
- Full-weight training creates 121 `NT` parameter-gradient GEMM rows per image size with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates 242 `NT` parameter-gradient rows per image size because both LoRA factors are included: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.

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
- Forward Linears map to `TN`, not `NT`. Full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/wan_gemm_shapes.csv`, with assumptions and source paths in `wan_gemm_shapes_manifest.json`.

Summary of Wan rows:
- Per video size, the forward Linear rows sum to all 406 checkpoint Linear weights and use `TN`.
- Full-weight training creates 406 `NT` parameter-gradient GEMM rows per video size with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates 812 `NT` parameter-gradient rows per video size because both LoRA factors are included: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.

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
- LTX latents use 128 channels, spatial `/32`, and temporal `((frames - 1) // 8) + 1`. Patch size is 1.
- Exact video token rows before GEMM rounding are 1500 for `640x480x40` and 8800 for `1280x720x80` using Comfy's integer-division latent sizing. The 720p case uses 22 latent height rows because 720 is not divisible by 32. The saved rows round up to GEMM-friendly multiples of 256: 1536 and 8960.
- The no-audio checkpoint subset has 640 2D Linear/GEMM weights: `[4096,4096] x418`, `[32,4096] x104`, `[4096,16384] x56`, `[16384,4096] x56`, plus one each of `[128,4096]`, `[4096,128]`, `[8192,4096]`, `[36864,4096]`, and two `[4096,256]` timestep linears.

Forward-pass NT check:
- No default PyTorch/ComfyUI LTX 2.3 forward NT GEMMs were found for the no-audio diffusion-model Linears or LoRA forward linears.
- Forward Linears map to `TN`, not `NT`. Full-weight and rank-16 LoRA parameter-gradient rows map to `NT`.
- The saved `m,n,batch,k` columns use TensileLite exact-shape order `[M,N,batch,K]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/ltx_2_3_gemm_shapes.csv`, with assumptions and source paths in `ltx_2_3_gemm_shapes_manifest.json`.

Summary of LTX 2.3 rows:
- Per video size, the forward Linear rows sum to all 640 included no-audio Linear weights and use `TN`.
- Full-weight training creates 640 `NT` parameter-gradient GEMM rows per video size with exact pattern `[M=in, N=out, batch=1, K=B]`.
- Rank-16 LoRA training creates 1280 `NT` parameter-gradient rows per video size because both LoRA factors are included: down maps to `[M=in, N=16, batch=1, K=B]`. Up maps to `[M=16, N=out, batch=1, K=B]`.

### H3

Sources checked:
- ComfyUI H3 model and conditioning code under `~/ComfyUI/comfy/ldm/minimax/model.py`, `~/ComfyUI/comfy/text_encoders/minimax.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/latent_formats.py`, and `~/ComfyUI/comfy_extras/nodes_minimax_h3.py`.
- Dequantized logical tensor shapes from `~/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`.
- Theoretical training and LoRA target paths under `~/musubi-tuner/src/musubi_tuner/minimax_h3/model.py`, `~/musubi-tuner/src/musubi_tuner/minimax_h3/packing.py`, `~/musubi-tuner/src/musubi_tuner/minimax_h3_train_network.py`, and `~/musubi-tuner/src/musubi_tuner/networks/lora_minimax_h3.py`.

Assumptions: batch size 1, requested video sizes `640x480x40` and `1280x720x80`, a short 16-token prompt, no first/last keyframes or reference inputs, LoRA rank 16, external Qwen3-VL-32B text-encoder weights excluded, and attention-kernel matmuls excluded. The local INT8 ConvRot weights are treated as ordinary dequantized BF16 matrices. Quantization scales and ConvRot metadata are not GEMMs.

H3 geometry and packed rows:
- H3's valid video frame grid is `17*n+5`. ComfyUI rounds the requested 40 and 80 frame inputs upward to 56 and 90 frames respectively. Musubi's dataset path rounds source clips downward instead, so use the effective H3 frame count shown below when reproducing these rows in training.
- H3 uses 24 video latent channels, spatial `/16`, temporal latent frames `2 + 5*((frames-5)//17)`, and video patch size `(1,2,2)`. The effective video token grid is therefore spatially `/32`.
- The target latent uses the requested width and height directly. ComfyUI's 768-short-edge canvas adaptation is only used for reference videos, so it does not change these T2VA target rows.
- For `1280x720x80`, the 720-pixel height produces 45 latent rows. ComfyUI pads that latent axis to 46 before 2x2 patchification.

| Requested input | Effective H3 frames | Latent grid `[T,H,W]` | Video rows exact / used | Audio rows exact / used | Packed rows exact / used |
| --- | ---: | --- | ---: | ---: | ---: |
| `640x480x40` | 56 | `[17,30,40]` | 5100 / 5120 | 186 / 256 | 5302 / 5376 |
| `1280x720x80` | 90 | `[27,46,80]` | 24840 / 25088 | 300 / 512 | 25156 / 25344 |

Audio rows are `2 * audio_latent_frames` at 40 Hz. Packed rows are `16 text + audio + video`. The saved GEMM rows round video, audio, and packed activation rows independently up to multiples of 256.

H3 config and checkpoint facts:
- H3 is a single-stream audio-video transformer with hidden width 5376, 50 main DiT blocks, 2 text-token refiner blocks, 56 attention heads, head dimension 128, attention inner width 7168, FFN width 14336, text input width 5120, video patch width 96, and audio patch width 32.
- The pruned checkpoint has 264 logical 2D Linear/GEMM weights: `[21504,5376] x52`, `[5376,7168] x52`, `[28672,5376] x52`, `[5376,14336] x52`, `[96768,8] x50`, plus one each of `[5376,96]`, `[5376,32]`, `[5376,5120]`, `[10752,8]`, `[96,5376]`, and `[32,5376]`.
- `adaln_t_table [1025,8]` is an interpolation table, not a GEMM weight. The pruned AdaLN projections therefore use input width 8. This is different from the unpruned full-AdaLN H3 architecture and is intentional for the supplied checkpoint.
- H3 packs one text span, audio rows, and video rows into one main sequence. Keyframe rows would add one spatial video frame per supplied keyframe. Ref2VA reference rows are input-dependent and are excluded from this fixed baseline.
- The external Qwen3-VL-32B conditioning checkpoint is not included in this diffusion-model shape set.

Forward-pass layout check:
- No default H3 forward `NT` GEMMs were found. Contiguous video/text/packed-stream inputs use `TN`.
- In the no-reference ComfyUI T2VA path, `pack_audio` returns a `[2*audio_latent_frames,32]` channel-major view with stride `(1,2*audio_latent_frames)`. `audio_patch_proj` therefore has confirmed `TT` rows. H3's TT cases are recorded in `input_shapes_tt.md`. Condition/reference audio assembly and the Musubi training `torch.cat` path materialize contiguous inputs and use `TN`.
- Full-weight gradients use `NT` `[M=in, N=out, batch=1, K=B]`. The theoretical rank-16 LoRA factor gradients use down `[M=in, N=16, batch=1, K=B]` and up `[M=16, N=out, batch=1, K=B]`.

Exact machine-readable rows are saved in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/h3_gemm_shapes.csv`, with geometry, checkpoint, and training notes in `h3_gemm_shapes_manifest.json`.

Summary of H3 rows:
- Per requested video size, the forward rows cover all 264 logical checkpoint Linear weights: 263 `TN` weights and one `TT` audio input projection.
- Theoretical full-weight training creates 264 `NT` parameter-gradient GEMM rows per video size.
- The documented rank-16 LoRA scope follows the Musubi default target pattern: only the 50 main blocks' `qkv_proj`, `out_proj`, `fc1`, and `fc2` are adapted. That is 200 target weights and 400 factor-gradient rows per size. An all-linear theoretical LoRA expansion would add the remaining 64 weights, but is not included in the saved LoRA rows.

### MOSS-Image

Sources checked:
- ComfyUI diffusion-model code under `~/ComfyUI/comfy/ldm/moss_image/model.py`.
- Text-conditioning wrapper under `~/ComfyUI/comfy/text_encoders/moss_image.py`.
- Detection, loading, and LoRA mapping under `~/ComfyUI/comfy/supported_models.py`, `~/ComfyUI/comfy/model_detection.py`, `~/ComfyUI/comfy/model_base.py`, `~/ComfyUI/comfy/latent_formats.py`, `~/ComfyUI/comfy/lora.py`, `~/ComfyUI/comfy/lora_convert.py`, and `~/ComfyUI/comfy_extras/nodes_moss_image.py`.
- Training implementation under `~/sd-scripts/library/moss_image_models.py`, `~/sd-scripts/library/moss_image_utils.py`, and `~/sd-scripts/docs/moss_image_train_network.md`.
- Remote safetensors headers from `https://huggingface.co/Comfy-Org/MOSS-Image`, fetched with HTTP Range requests only. Tensor payloads were not downloaded.

Assumptions: batch size 1, image sizes `1024x1024` and `1536x1536`, short prompt treated as 16 text tokens, LoRA rank 16, no color/depth control adapter, and attention-kernel matmuls excluded.

MOSS-Image config and checkpoint facts:
- The diffusion model is a 42-block single-stream DiT with width 3072, FFN width 8192, 24 attention heads of dimension 128, text input width 4096, frequency width 256, and 128 latent channels.
- The latent format is `Flux2`, 128 channels at `/16`. For `1024x1024` the latent is `64x64` and image rows are 4096. For `1536x1536` it is `96x96` and image rows are 9216. Native 2K `2048x2048` gives a `128x128` latent and 16384 image rows.
- The FFN uses a SiTU-GLU on the two halves of a fused `ffn_up_gate = Linear(3072,16384)` output, followed by `ffn_down = Linear(8192,3072)`. One shared `shared_block_modulation = Linear(3072,18432)` supplies the six per-token modulation vectors for all 42 blocks, and `video_head_modulation = Linear(3072,6144)` supplies the final shift/scale.
- The released ComfyUI bf16 files store **185 fused-view 2D Linear weights**: `[3072,3072] x45`, `[3072,8192] x44`, `[9216,3072] x44`, `[16384,3072] x42`, `[8192,3072] x4`, plus one each of `[128,3072]`, `[18432,3072]`, `[3072,4096]`, `[3072,256]`, `[3072,128]`, and `[6144,3072]`. The other 194 tensors are 89 width-3072 RMSNorm scales, 88 head-dimension-128 RMSNorm scales, `rope.bands`, and biases.
- `moss_image_sft_int8_convrot.safetensors` keeps the same logical shapes but stores the 168 main-block Linears as per-row-scaled INT8 with `comfy_quant` metadata. Its rows are recorded as logical dequantized BF16-equivalent shapes.
- The sd-scripts training model and the official diffusers shards keep `proj.q`, `proj.k`, and `proj.v` separate, so the training model has **269 logical Linears**. `moss_image_utils.split_fused_qkv` splits the ComfyUI fused file on load.
- The external Qwen3.5-9B text encoder checkpoint is not included in this diffusion-model shape set.

Forward-pass NT check:
- No default PyTorch/ComfyUI MOSS-Image forward NT GEMMs were found. The main blocks, the text refiner, the timestep/modulation Linears, and `final_linear_video` all receive contiguous activations and are `TN`.
- `video_embedder` is the exception: ComfyUI passes `x.flatten(2).transpose(1,2)` directly, which is `TT`. A `torch` dispatch probe on the real module confirms the column-major operand pair. sd-scripts calls `.contiguous()` on the same view, so the training forward is `TN`. That case is documented in `input_shapes_tt.md`.
- The sequence is image tokens followed by text tokens, with full bidirectional attention over the packed result. Each of the 42 main blocks issues exactly one fused-attention call. The two text-refiner blocks issue their own 16-token calls. There is no block-causal split and no prefix K/V cache. Fused-attention rows are documented in the attention shape note.
- `final_linear_video` runs on `x[:, :H*W]`, so the text rows are sliced off before the final modulation and projection.

Fused versus unfused parameter gradients:
- The 108 saved `NT` rows follow the sd-scripts training model: 18 module kinds per image size with 269 full-weight parameter-gradient GEMMs and 538 rank-16 factor-gradient GEMMs.
- Training the fused ComfyUI repack directly would instead produce one `[3072,9216,1,rows]` qkv parameter gradient per block. That variant is not saved, because ComfyUI has no MOSS-Image training loop and the released files are inference repacks.
- A rank-16 LoRA on `proj.qkv` in ComfyUI is mapped onto the fused weight slice by slice, so its factor GEMMs are the same per-projection `[3072,rows,1,16]` / `[16,rows,1,3072]` rows as the unfused model.

Summary of MOSS-Image rows:
- Per image size the forward Linear rows sum to all 269 logical Linears, or the 185 fused-view weights when the fused `qkv` row is counted instead of three `q/k/v` rows.
- Full-weight training creates 18 `NT` rows covering 269 parameter-gradient GEMMs per image size.
- Rank-16 LoRA on every logical Linear creates 538 factor-gradient rows per image size: 269 down maps `[M=in, N=16, batch=1, K=rows]` and 269 up maps `[M=16, N=out, batch=1, K=rows]`. The sd-scripts default target set is the 252 main-block modules. The 11 text-embedder modules need `train_text_embedder=True`.
- Block Linears use the packed rows `16+4096=4112` and `16+9216=9232`. The text embedder and text refiner use 16 rows, the timestep and modulation Linears use 1 row, `video_embedder` and `final_linear_video` use the image rows.
- Per image size the `NT` rows collapse to 33 unique exact shapes and 807 weighted occurrences, so 46 unique shapes and 1614 weighted occurrences across both sizes.
- Of the 46 unique shapes, 14 are already in the large-grid target union (69 weighted occurrences). The other 32 are outside it: 14 are the packed `K=4112`/`9232` block pairs, 9 are the `K=4096`/`9216` in/out pairs, and 9 are the small `K=1`/`K=16` modulation and text-embedder rows.

## Recommended NT HHS Tuning Plan

### Normalized Workload CSV Schema

All model workload CSVs in `~/tmp/feather_ops/tensile_fp16_nt_hhs/shape_data/*_gemm_shapes.csv` use the same columns:

`dataset,component,phase,op_kind,input_size,activation_rows_kind,layout,m,n,batch,k,count,weight_out,weight_in,activation_rows,rank,factor,source,notes`

The exact TensileLite shape columns remain `m,n,batch,k` in `[M,N,batch,K]` order. The `count` column is the number of model weights represented by that aggregate row. `phase` and `op_kind` use a fixed vocabulary (`forward`, `forward_lora_down`, `forward_fused_inference`, `forward_cache_prefix`, `forward_cache_target`, `backward_full_weight_grad`, `backward_lora_down_weight_grad`, `backward_lora_up_weight_grad`. `linear`, `linear_weight_grad`, `lora_down_linear`, `lora_up_linear`, `lora_down_weight_grad`, `lora_up_weight_grad`), and `factor` carries the explicit `down`/`up` label for LoRA rows.

### Planning Summary

The ComfyUI workloads analyzed here produce no default forward `NT` GEMMs for the model Linears. Ordinary forward Linears are `TN`, with confirmed strided `TT` exceptions documented separately. Training input gradients use `NN` for the corresponding base and LoRA factor matrices. Exact TN and NN datasets are documented in `input_shapes_tn.md` and `input_shapes_nn.md`. The `NT` tuning target here remains full-weight and LoRA parameter-gradient GEMMs.

Aggregating `NT` rows from SDXL, Anima, Qwen-Image 1, Qwen-Image 2.1, Z-Image, Krea 2, Ideogram 4, Klein 9B, Wan, LTX 2.3, H3, and MOSS-Image gives:
- 857 normalized CSV rows.
- 488 unique exact `NT` shapes.
- 29,417 weighted `NT` shape occurrences after applying each row's `count`.
- Current NT HHS exact table covers only 6 of those unique shapes, or 1,060 weighted occurrences.
- The large-grid target union covers 68 of those unique shapes, or 5,921 weighted occurrences.

Important workload dimensions not covered by the current large-grid dimension sets:
- Extra `M` values: `[68, 96, 1280, 2560, 3584, 3840, 4608, 5120, 5376, 6144, 6912, 7168, 10240, 12288, 13824, 14336, 16384, 53248]`.
- Extra `N` values: `[96, 4608, 5120, 5376, 6144, 6912, 10240, 10752, 11520, 12288, 13824, 15360, 16384, 18432, 21504, 24576, 28672, 30720, 36864, 96768]`.
- Extra `K` activation-row values: `[192, 1536, 2304, 4112, 4128, 4608, 5120, 5376, 8960, 9216, 9232, 9248, 9728, 12032, 25088, 25344, 40960, 72192]`.

The `M` extensions matter for `NT` specifically. Values like `1280` exist as `N` in the inherited large grid, but `NT` parameter gradients often put feature/input dimensions on `M`, so an `N`-only value is not enough.

### Recommended Shape Set

Tune NT HHS on the union of these tiers:
- Base large grid: all 9,681 shapes from `large_grid_target_union_shapes.csv`. This brings NT HHS to the same broad coverage already used by the tuned HHS/BBS TN/NN work.
- Exact workload misses: all unique `NT` shapes from the normalized `*_gemm_shapes.csv` files that are not already in the large-grid target union. This adds 420 shapes and gives exact coverage for the two analyzed sizes per model.
- Banded workload anchors: add K-banded variants of the workload shapes so nearby image/video sizes can choose a sensible tuned solution instead of falling back to a distant exact shape. Use this K mapping for the workload-derived rows: keep `1`, `16`, `32`, `192`, `512`, `1024`, `1536`, `2304`, `4096`, `5120`, `9216`, `25088`, and `40960`. Map `4112`, `4128`, and `4608 -> 4096`. Map `5376 -> 5120`. Map `8960`, `9232`, `9248`, and `9728 -> 9216`. Map `12032 -> 12288`. Map `25344 -> 25088`. Map `72192 -> 73728`.

The combined recommended set has 10,172 unique exact shapes: the 9,681-shape large grid plus 491 workload/banded additions.

Do not expand the workload extensions into a full Cartesian product of every observed `M`, `N`, and `K`. Keep the workload `M,N` pairs from the CSV rows and only add the K-banded variants above. The existing large grid already provides broad canonical coverage. The workload extension is for high-value missing feature pairs and token-row bands.

### Priority If Tuning Budget Is Limited

Minimum viable NT HHS update:
- Tune the 9,681-shape large-grid target union first.
- Add the high-weight workload misses with `K in {1536, 2304, 4096, 4112, 4128, 5120, 5376, 9216, 9232, 9248, 12288, 25088, 25344, 73728}`.
- Add skinny LoRA rank-16 shapes where `M=16` or `N=16` for feature sizes `[1280, 1536, 2048, 2560, 3072, 3840, 4096, 4608, 5120, 5376, 6144, 6912, 7168, 10240, 11520, 12288, 13824, 14336, 15360, 16384, 18432, 21504, 24576, 28672, 36864]`.

High-priority K bands:
- `K=1536`: LTX 2.3 small-video rows. Especially `(M,N)` pairs `(4096,16)`, `(16,4096)`, `(4096,4096)`, `(4096,32)`, `(16,32)`, `(16384,4096)`, and `(4096,16384)`.
- `K=2304`: SDXL `1536x1536` mid-resolution rows. Especially `(16,1280)`, `(1280,16)`, `(1280,1280)`, `(1280,10240)`, and `(5120,1280)`.
- `K=4096`/`K=4112`/`K=4128`: Ideogram 4, Z-Image, and Krea 2 `1024x1024` rows. Especially `(16,3840)`, `(3840,16)`, `(3840,3840)`, `(3840,10240)`, `(10240,3840)`, `(3840,11520)`, `(16,4608)`, `(4608,4608)`, `(12288,4608)`, `(16,6144)`, `(6144,6144)`, `(6144,16384)`, and `(16384,6144)`.
- `K=9216`/`K=9232`/`K=9248`: high-resolution image rows, nearby LTX/Klein rows, and Ideogram 4/Z-Image/Krea 2 conditional rows. Especially `(16,3072)`, `(3072,16)`, `(3072,3072)`, `(4096,4096)`, `(4608,4608)`, `(12288,4608)`, `(3840,3840)`, `(10240,3840)`, `(6144,6144)`, and `(16384,6144)`.
- `K=12288`: Wan `640x480x40` video band. Especially `(5120,16)`, `(16,5120)`, `(5120,5120)`, `(5120,13824)`, and `(13824,5120)`.
- `K=73728`: Wan `1280x720x80` video band. Use the same Wan feature pairs as `K=12288`.
- `K=5120`/`K=5376`: H3 `640x480x40` packed/video bands. Especially `(5376,16)`, `(16,5376)`, `(5376,21504)`, `(7168,5376)`, `(5376,28672)`, and `(14336,5376)`.
- `K=25088`/`K=25344`: H3 `1280x720x80` packed/video bands. Use the same H3 feature pairs as `K=5120`/`K=5376`.
- `K=4112` and `K=9232`: Qwen-Image 2.1 packed text+image bands. Especially `(16,4096)`, `(4096,16)`, `(4096,4096)`, `(4096,24576)`, `(16,24576)`, `(12288,16)`, and `(12288,4096)`.
- `K=4112` and `K=9232`: MOSS-Image packed text+image bands. Especially `(3072,3072)`, `(3072,16384)`, `(8192,3072)`, `(16,3072)`, `(3072,16)`, `(16,16384)`, and `(8192,16)`.
- `K=2`: Qwen-Image 2.1 shared timestep/modulation rows. Especially `(4096,4096)`, `(4096,16384)`, `(16,16384)`, `(256,4096)`, `(4096,16)`, and `(256,16)`. These are caught by the base large grid only for `K=2` itself, not for the `(M,N)` pairs.

Lower-priority or deferrable items:
- Exact `K=4112`/`K=9232` Ideogram 4 and Krea 2 rows, `K=4128`/`K=9248` Z-Image rows, and `K=4608`/`K=9728` Klein rows can be covered by nearby `K=4096` and `K=9216` anchors if tuning budget is tight.
- Small-token Z-Image context-refiner rows with `K=32`, Krea 2 TextFusion layer-stack rows with `K=192`, Krea 2 projector rows with `K=40960`, and timestep/global rows with `K=1` are low arithmetic work or very skinny. Keep them in the full recommended set, but deprioritize them in a constrained tuning run after the larger token-row bands are covered.
- H3's pruned-AdaLN `K=2` gradients and tiny audio/patch projection rows are low arithmetic work. Keep them in the full recommended set but prioritize the packed main-block rows first.
- Very low-count odd feature sizes such as `M=68`, `M=96`, `M=3584`, `M=53248`, `N=96`, `N=30720`, and exact patch/unpatch one-offs can be deferred unless profiling shows them hot.
- MOSS-Image's `K=16` text-embedder rows `(4096,3072)`, `(3072,9216)`, `(3072,8192)`, `(8192,3072)`, `(16,9216)`, and `(16,8192)` are small but cheap to add, and its `K=4096`/`K=9216` rows `(128,3072)`, `(3072,128)`, `(16,128)`, and `(128,16)` are the only feature-128 pairs in the set.
- Qwen-Image 2.1's `K=1`/`K=2`/`K=16` timestep and text-projection rows are low arithmetic work, and its `(4096,64)`/`(64,4096)` `K=4096`/`K=9216` rows are already inside the large grid. The packed `K=4112`/`K=9232` rows are the ones worth exact coverage, and `K=9216` with `(64,4096)`/`(4096,64)` is the cheapest way to extend the image band beyond `K=4096`.
