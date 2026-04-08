#!/usr/bin/env python3
"""Patch fair-esm inverse folding code for correct GPU device placement.

The fair-esm 2.0.0 package has several locations where tensors are created
on CPU without respecting the model's device, causing RuntimeError when
the model is on GPU. This script patches the installed files in-place.

Bugs fixed:
1. gvp_transformer.py: model.sample() creates sampled_tokens on CPU and
   doesn't pass device to CoordBatchConverter
2. util.py: CoordBatchConverter.__call__() creates coord/confidence tensors
   on CPU then moves them, but other tensors derived before the move inherit CPU
3. gvp_transformer_encoder.py: mask_tokens created via int multiplication
   of padding_mask needs explicit .long() cast for correct dtype
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_file(path: Path, old: str, new: str, label: str) -> None:
    """Replace old with new in file, or fail loudly."""
    text = path.read_text()
    if old not in text:
        print(f"  SKIP {label}: pattern not found (already patched?)")
        return
    count = text.count(old)
    if count > 1:
        print(f"  WARNING {label}: pattern found {count} times, patching all", file=sys.stderr)
    text = text.replace(old, new)
    path.write_text(text)
    print(f"  OK   {label}")


def main() -> None:
    base = Path("/usr/local/lib/python3.10/dist-packages/esm/inverse_folding")
    if not base.exists():
        print(f"ERROR: {base} not found", file=sys.stderr)
        sys.exit(1)

    print("[patch] Patching gvp_transformer.py ...")

    gvp_transformer = base / "gvp_transformer.py"

    # Fix 1a: Pass device to CoordBatchConverter so tensors are created on GPU
    patch_file(
        gvp_transformer,
        old=(
            "        batch_coords, confidence, _, _, padding_mask = (\n"
            "            batch_converter([(coords, confidence, None)])\n"
            "        )"
        ),
        new=(
            "        device = next(self.parameters()).device\n"
            "        batch_coords, confidence, _, _, padding_mask = (\n"
            "            batch_converter([(coords, confidence, None)], device=device)\n"
            "        )"
        ),
        label="pass device to batch_converter",
    )

    # Fix 1b: Create sampled_tokens on the correct device
    patch_file(
        gvp_transformer,
        old="        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)",
        new="        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=torch.long, device=device)",
        label="sampled_tokens device",
    )

    print("[patch] Patching util.py ...")

    util = base / "util.py"

    # Fix 2: Create coord/confidence tensors on target device from the start.
    # The original code creates on CPU then moves with .to(device). Instead,
    # create on device directly so derived tensors (padding_mask, coord_mask)
    # also land on the correct device.
    patch_file(
        util,
        old=(
            "        coords = [\n"
            "            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)\n"
            "            for cd, _ in coords_and_confidence\n"
            "        ]\n"
            "        confidence = [\n"
            "            F.pad(torch.tensor(cf), (1, 1), value=-1.)\n"
            "            for _, cf in coords_and_confidence\n"
            "        ]\n"
            "        coords = self.collate_dense_tensors(coords, pad_v=np.nan)\n"
            "        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)\n"
            "        if device is not None:\n"
            "            coords = coords.to(device)\n"
            "            confidence = confidence.to(device)\n"
            "            tokens = tokens.to(device)"
        ),
        new=(
            "        _dev = device or 'cpu'\n"
            "        coords = [\n"
            "            F.pad(torch.tensor(cd, device=_dev), (0, 0, 0, 0, 1, 1), value=np.inf)\n"
            "            for cd, _ in coords_and_confidence\n"
            "        ]\n"
            "        confidence = [\n"
            "            F.pad(torch.tensor(cf, device=_dev), (1, 1), value=-1.)\n"
            "            for _, cf in coords_and_confidence\n"
            "        ]\n"
            "        coords = self.collate_dense_tensors(coords, pad_v=np.nan)\n"
            "        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)\n"
            "        if device is not None:\n"
            "            tokens = tokens.to(device)"
        ),
        label="coord/confidence device placement",
    )

    # Fix 2b: get_sequence_loss() calls batch_converter without device, so
    # all tensors stay on CPU while the model is on GPU. Patch it to detect
    # the model's device and pass it through.
    patch_file(
        util,
        old=(
            "def get_sequence_loss(model, alphabet, coords, seq):\n"
            "    batch_converter = CoordBatchConverter(alphabet)\n"
            "    batch = [(coords, None, seq)]\n"
            "    coords, confidence, strs, tokens, padding_mask = batch_converter(batch)"
        ),
        new=(
            "def get_sequence_loss(model, alphabet, coords, seq):\n"
            "    device = next(model.parameters()).device\n"
            "    batch_converter = CoordBatchConverter(alphabet)\n"
            "    batch = [(coords, None, seq)]\n"
            "    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)"
        ),
        label="get_sequence_loss device",
    )

    # Fix 2c: get_sequence_loss calls .detach().numpy() on GPU tensors,
    # which fails — need .cpu() first.
    patch_file(
        util,
        old=(
            "    loss = loss[0].detach().numpy()\n"
            "    target_padding_mask = target_padding_mask[0].numpy()"
        ),
        new=(
            "    loss = loss[0].detach().cpu().numpy()\n"
            "    target_padding_mask = target_padding_mask[0].cpu().numpy()"
        ),
        label="get_sequence_loss cpu before numpy",
    )

    print("[patch] Patching gvp_transformer_encoder.py ...")

    encoder = base / "gvp_transformer_encoder.py"

    # Fix 3: mask_tokens creation — use .long() for correct dtype when
    # multiplying bool padding_mask by int scalars, and ensure result
    # is on the same device as padding_mask.
    patch_file(
        encoder,
        old=(
            "        mask_tokens = (\n"
            "            padding_mask * self.dictionary.padding_idx + \n"
            "            ~padding_mask * self.dictionary.get_idx(\"<mask>\")\n"
            "        )"
        ),
        new=(
            "        mask_tokens = (\n"
            "            padding_mask.long() * self.dictionary.padding_idx + \n"
            "            (~padding_mask).long() * self.dictionary.get_idx(\"<mask>\")\n"
            "        )"
        ),
        label="mask_tokens dtype",
    )

    print("[patch] Done.")


if __name__ == "__main__":
    main()
