#!/usr/bin/env python3

import argparse
import json
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def run_binary(cmd: list[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def find_tool(candidates: list[str]) -> str:
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    for c in candidates:
        if pathlib.Path(c).exists():
            return c
    raise RuntimeError(f"Unable to find tool from candidates: {candidates}")


def extract_code_objects(so_path: pathlib.Path, out_dir: pathlib.Path, llvm_objcopy: str) -> list[pathlib.Path]:
    fatbin = out_dir / "hip_fatbin.bin"
    run([llvm_objcopy, "--dump-section", f".hip_fatbin={fatbin}", str(so_path)])

    data = fatbin.read_bytes()
    offsets = [m.start() for m in re.finditer(b"\\x7fELF", data)]
    if not offsets:
        raise RuntimeError("No embedded ELF found in .hip_fatbin")

    codeobjs: list[pathlib.Path] = []
    for i, off in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(data)
        blob = data[off:end]
        p = out_dir / f"hip_codeobj_{i}.elf"
        p.write_bytes(blob)
        codeobjs.append(p)
    return codeobjs


@dataclass
class KernelTemplate:
    warps_m: int
    warps_n: int
    unroll_k: int
    stages: int
    repeat_m: int
    repeat_n: int
    check_bounds: bool
    contig_fastpath: bool


def parse_template_key(key: str) -> KernelTemplate:
    m = re.fullmatch(
        r"ILi(-?\d+)ELi(-?\d+)ELi(-?\d+)ELi(-?\d+)ELi(-?\d+)ELi(-?\d+)ELb([01])ELb([01])",
        key,
    )
    if not m:
        raise RuntimeError(f"Bad template key: {key}")
    return KernelTemplate(
        warps_m=int(m.group(1)),
        warps_n=int(m.group(2)),
        unroll_k=int(m.group(3)),
        stages=int(m.group(4)),
        repeat_m=int(m.group(5)),
        repeat_n=int(m.group(6)),
        check_bounds=(m.group(7) == "1"),
        contig_fastpath=(m.group(8) == "1"),
    )


def find_kernel_symbol(codeobj: pathlib.Path, template_key: str) -> str:
    out = run(["nm", "-D", "--defined-only", str(codeobj)]).stdout.splitlines()
    hits = []
    for line in out:
        if "scaled_mm_kernel_wmma_k0mk1" not in line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        sym = parts[-1]
        if template_key in sym:
            hits.append(sym)
    if not hits:
        raise RuntimeError(f"No kernel symbol found for template key {template_key}")
    return hits[0]


def disassemble_symbol(codeobj: pathlib.Path, symbol: str, llvm_objdump: str, out_path: pathlib.Path) -> str:
    txt = run([llvm_objdump, "-d", f"--disassemble-symbols={symbol}", str(codeobj)]).stdout
    out_path.write_text(txt)
    return txt


def summarize_instructions(disasm: str) -> dict[str, int]:
    lines = disasm.splitlines()
    inst_lines = [ln for ln in lines if re.match(r"^\s*[A-Za-z][A-Za-z0-9_\.]+\b", ln)]
    text = "\n".join(inst_lines)

    def c(pattern: str) -> int:
        return len(re.findall(pattern, text))

    return {
        "total_instructions": len(inst_lines),
        "global_load": c(r"\bglobal_load_"),
        "global_store": c(r"\bglobal_store_"),
        "buffer_load": c(r"\bbuffer_load_"),
        "buffer_store": c(r"\bbuffer_store_"),
        "ds_read": c(r"\bds_read_"),
        "ds_store": c(r"\bds_store_"),
        "wmma": c(r"\bv_wmma_"),
        "mfma": c(r"\bv_mfma_"),
        "s_waitcnt": c(r"\bs_waitcnt\b"),
        "s_waitcnt_vmcnt": c(r"\bs_waitcnt\s+vmcnt\("),
        "s_waitcnt_vmcnt0": c(r"\bs_waitcnt\s+vmcnt\(0\)"),
        "s_waitcnt_lgkmcnt": c(r"\bs_waitcnt\s+lgkmcnt\("),
        "s_waitcnt_lgkmcnt0": c(r"\bs_waitcnt\s+lgkmcnt\(0\)"),
        "s_barrier": c(r"\bs_barrier\b"),
    }


def parse_torch_tokens(torch_db: pathlib.Path) -> dict[str, object]:
    if not torch_db.exists():
        return {"kernel_name": "", "present_tokens": {}, "missing_db": True}

    query = "select name from top_kernels where name like 'Cijk_%' order by total_duration desc limit 1;"
    out = run(["sqlite3", str(torch_db), query]).stdout.strip()
    expected = [
        "MT128x128x32",
        "WG32_4_1",
        "WS32",
        "PGR1",
        "SIA3",
        "1LDSB1",
        "SU32",
        "SUS256",
        "WSGRA1",
        "WSGRB1",
        "LRVW16",
    ]
    present = {t: (t in out) for t in expected}
    return {"kernel_name": out, "present_tokens": present, "missing_db": False}


def hip_token_projection(t: KernelTemplate) -> dict[str, object]:
    macro_m = 16 * t.warps_m * t.repeat_m
    macro_n = 16 * t.warps_n * t.repeat_n
    depth_u = 16 * t.unroll_k
    # wg_size = 32 * t.warps_m * t.warps_n

    projected = {
        "MT": f"MT{macro_m}x{macro_n}x{depth_u}",
        "WG": f"WG32_{t.warps_m * t.warps_n}_1",
        "WS": "WS32",
        "1LDSB1_like": int(t.stages == t.unroll_k),
        "PGR1_like": int(t.stages >= 2 * t.unroll_k),
        "WSGRA_like": 0,
        "WSGRB_like": 0,
    }
    return projected


def build_recommendations(inst: dict[str, int], t: KernelTemplate) -> list[str]:
    rec: list[str] = []
    if inst["s_waitcnt_vmcnt0"] > 0:
        rec.append("Avoid forcing extra vmcnt(0) fences in hot path unless paired with measured benchmark gain; prior steps showed profile-only wins and benchmark loss.")
    if inst["s_waitcnt"] > (inst["wmma"] // 2 + 20):
        rec.append("Wait density is high relative to WMMA count; prioritize asm-parity scheduling for GR/LW/LR cadence instead of more C++ micro-edits.")
    if t.stages == t.unroll_k:
        rec.append("Current config behaves like 1LDSB1/no-overlap path; focus on no-overlap issue order and LDS lifetime parity.")
    if t.stages < 2 * t.unroll_k:
        rec.append("PGR1-like overlap is structurally unavailable with current stages; prioritize mainloop asm parity rather than overlap hints.")
    return rec


def write_summary_md(out_dir: pathlib.Path, report: dict[str, object]) -> None:
    p = out_dir / "isa_parity_summary.md"
    lines = []
    lines.append("# ISA Parity Summary")
    lines.append("")
    lines.append(f"- template: `{report['template_key']}`")
    lines.append(f"- selected symbol: `{report['hip_symbol']}`")
    lines.append(f"- torch kernel: `{report['torch']['kernel_name']}`")
    lines.append("")
    lines.append("## Instruction Counts (HIP kernel symbol)")
    for k, v in report["instruction_counts"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Token View")
    lines.append(f"- torch tokens: `{report['torch']['present_tokens']}`")
    lines.append(f"- hip projection: `{report['hip_projection']}`")
    lines.append("")
    lines.append("## Recommendations")
    for r in report["recommendations"]:
        lines.append(f"- {r}")
    p.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline HIP ISA parity checker for Step41")
    ap.add_argument(
        "--so",
        default="kernel/hip/build/scaled_mm_hip_ext/scaled_mm_hip_ext.so",
        help="HIP extension .so path",
    )
    ap.add_argument(
        "--template-key",
        default="ILi2ELi2ELi2ELi2ELi4ELi4ELb0ELb1",
        help="Template key fragment for active kernel",
    )
    ap.add_argument(
        "--torch-db",
        default="rocprof_torch_8192_steady/torch8192_steady_results.db",
        help="Torch steady rocprof sqlite db",
    )
    ap.add_argument(
        "--out-dir",
        default="rocprof_hip_8192_step41_isa_parity",
        help="Output directory",
    )
    args = ap.parse_args()

    root = pathlib.Path.cwd()
    so_path = (root / args.so).resolve()
    torch_db = (root / args.torch_db).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    llvm_objcopy = find_tool(
        [
            "llvm-objcopy",
            "~/venv_torch/lib/python3.13/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-objcopy",
            "~/venv_torch/lib/python3.13/site-packages/_rocm_sdk_core/lib/llvm/bin/llvm-objcopy",
        ]
    )
    llvm_objdump = find_tool(
        [
            "llvm-objdump",
            "~/venv_torch/lib/python3.13/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-objdump",
            "~/venv_torch/lib/python3.13/site-packages/_rocm_sdk_core/lib/llvm/bin/llvm-objdump",
        ]
    )

    codeobjs = extract_code_objects(so_path, out_dir, llvm_objcopy)
    selected = codeobjs[0]
    (out_dir / "selected_codeobj.txt").write_text(str(selected) + "\n")

    symbol = find_kernel_symbol(selected, args.template_key)
    (out_dir / "hip_symbol.txt").write_text(symbol + "\n")

    disasm = disassemble_symbol(selected, symbol, llvm_objdump, out_dir / "hip_kernel_disasm.s")
    inst = summarize_instructions(disasm)

    t = parse_template_key(args.template_key)
    torch = parse_torch_tokens(torch_db)
    proj = hip_token_projection(t)
    rec = build_recommendations(inst, t)

    report = {
        "template_key": args.template_key,
        "hip_symbol": symbol,
        "so_path": str(so_path),
        "selected_codeobj": str(selected),
        "instruction_counts": inst,
        "torch": torch,
        "hip_projection": proj,
        "recommendations": rec,
    }

    (out_dir / "isa_parity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary_md(out_dir, report)

    print(f"[OK] wrote {out_dir / 'isa_parity_report.json'}")
    print(f"[OK] wrote {out_dir / 'isa_parity_summary.md'}")


if __name__ == "__main__":
    main()
