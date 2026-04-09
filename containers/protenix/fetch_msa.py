#!/usr/bin/env python3
"""Fetch MSAs from a ColabFold server and convert to Protenix format.

Two phases:
  1. Submit all protein sequences to the ColabFold server as a single
     paired query, poll until completion, and download the result tarball.
  2. Parse the ColabFold combined a3m output and split into per-chain
     pairing.a3m and non_pairing.a3m files in Protenix's expected layout.

The ColabFold server API:
  - POST {url}/ticket/pair  with  q=>101\\nSEQ1\\n>102\\nSEQ2\\n  &mode=env
  - GET  {url}/ticket/{id}  -> JSON with status field
  - GET  {url}/result/download/{id}  -> tar.gz with a3m files

Adapted from Protenix's scripts/colabfold_msa.py (A3MProcessor).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_MSA_SERVER = "https://api.colabfold.com"
POLL_INTERVAL = 5  # seconds
MAX_POLL_TIME = 3600  # 1 hour max wait
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds between retries on transient errors


# ---------------------------------------------------------------------------
# ColabFold server interaction (stdlib only — no requests dependency)
# ---------------------------------------------------------------------------


def submit_msa_query(server_url: str, sequences: list[str]) -> str:
    """Submit protein sequences to the ColabFold server for paired MSA search.

    Args:
        server_url: Base URL of the ColabFold server.
        sequences: List of protein amino acid sequences.

    Returns:
        Ticket ID for polling.
    """
    # Build query: >101\nSEQ1\n>102\nSEQ2\n (ColabFold 101-based indexing)
    query_lines = []
    for i, seq in enumerate(sequences):
        query_lines.append(f">{101 + i}")
        query_lines.append(seq)
    query_str = "\n".join(query_lines) + "\n"

    data = urlencode({"q": query_str, "mode": "env"}).encode()
    url = f"{server_url}/ticket/pair"
    req = Request(url, data=data, method="POST")

    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                ticket_id = result["id"]
                print(f"[fetch_msa] Submitted {len(sequences)} sequence(s), ticket: {ticket_id}")
                return ticket_id
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[fetch_msa] Submit attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to submit MSA query after {MAX_RETRIES} attempts: {e}")

    raise RuntimeError("Unreachable")  # pragma: no cover


def poll_until_complete(server_url: str, ticket_id: str) -> None:
    """Poll the ColabFold server until the job completes or errors.

    Args:
        server_url: Base URL of the ColabFold server.
        ticket_id: Ticket ID from submission.

    Raises:
        RuntimeError: On job error, rate limit, maintenance, or timeout.
    """
    start = time.monotonic()
    consecutive_errors = 0

    while True:
        elapsed = time.monotonic() - start
        if elapsed > MAX_POLL_TIME:
            raise RuntimeError(f"MSA server query timed out after {MAX_POLL_TIME}s")

        try:
            url = f"{server_url}/ticket/{ticket_id}"
            with urlopen(url, timeout=30) as resp:
                result = json.loads(resp.read().decode())
            consecutive_errors = 0
        except (HTTPError, URLError, TimeoutError) as e:
            consecutive_errors += 1
            if consecutive_errors >= MAX_RETRIES:
                raise RuntimeError(f"Lost connection to MSA server after {MAX_RETRIES} errors: {e}")
            print(f"[fetch_msa] Poll error ({consecutive_errors}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
            continue

        status = result.get("status", "UNKNOWN")

        if status == "COMPLETE":
            print(f"[fetch_msa] Job completed in {elapsed:.0f}s")
            return
        elif status == "ERROR":
            raise RuntimeError(f"MSA server returned error: {result}")
        elif status == "RATELIMIT":
            print("[fetch_msa] Rate limited, waiting 60s...")
            time.sleep(60)
        elif status == "MAINTENANCE":
            raise RuntimeError("MSA server is under maintenance. Try again later.")
        else:
            # PENDING, RUNNING, UNKNOWN
            time.sleep(POLL_INTERVAL)


def download_result(server_url: str, ticket_id: str) -> bytes:
    """Download the completed MSA result tarball.

    Returns:
        Raw bytes of the tar.gz file.
    """
    url = f"{server_url}/result/download/{ticket_id}"

    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(url, timeout=120) as resp:
                data = resp.read()
                print(f"[fetch_msa] Downloaded result ({len(data)} bytes)")
                return data
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[fetch_msa] Download attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to download result after {MAX_RETRIES} attempts: {e}")

    raise RuntimeError("Unreachable")  # pragma: no cover


def extract_a3m_from_tarball(tarball_bytes: bytes) -> dict[str, str]:
    """Extract a3m files from the ColabFold result tarball.

    Returns:
        Dict mapping filename to file content string.
    """
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    # Use just the filename, not the full path
                    name = Path(member.name).name
                    files[name] = f.read().decode()
    return files


# ---------------------------------------------------------------------------
# A3M conversion — adapted from Protenix's scripts/colabfold_msa.py
# ---------------------------------------------------------------------------


class A3MProcessor:
    """Convert ColabFold combined a3m to Protenix per-chain paired/unpaired format.

    ColabFold's combined a3m format:
      - Header line: #L1,L2,...\\tN1,N2,...  (chain lengths, oligomeric counts)
      - Paired entries: sequences concatenated across chains
      - Per-chain unpaired entries

    Protenix expects per-chain directories:
      - msa/<chain_idx>/pairing.a3m (taxonomy-keyed for cross-chain pairing)
      - msa/<chain_idx>/non_pairing.a3m (chain-specific MSA hits)
    """

    def __init__(self, sequences: list[str]):
        self.sequences = sequences
        self.num_chains = len(sequences)

    def process(self, a3m_content: str, output_dir: Path) -> dict[int, dict[str, Path]]:
        """Process combined a3m into per-chain MSA files.

        Args:
            a3m_content: Combined a3m content from ColabFold.
            output_dir: Base directory to write per-chain MSAs.

        Returns:
            Dict mapping chain index to {"pairing": Path, "non_pairing": Path}.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.num_chains == 1:
            return self._process_monomer(a3m_content, output_dir)
        return self._process_multimer(a3m_content, output_dir)

    def _process_monomer(
        self, a3m_content: str, output_dir: Path
    ) -> dict[int, dict[str, Path]]:
        """Process monomer a3m — full MSA as non_pairing, query-only as pairing."""
        lines = a3m_content.strip().split("\n")

        # Skip header line if present
        start_idx = 0
        if lines and lines[0].startswith("#"):
            start_idx = 1

        # Find query sequence (first sequence after any header)
        query_seq = ""
        for i in range(start_idx, len(lines)):
            if not lines[i].startswith(">"):
                query_seq = lines[i]
                break

        chain_dir = output_dir / "0"
        chain_dir.mkdir(exist_ok=True)

        # non_pairing.a3m = full MSA with "query" as first header
        non_pairing_lines = [">query", query_seq]
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if line.startswith(">"):
                # Skip the first query entry (we already wrote it)
                if i == start_idx:
                    continue
                non_pairing_lines.append(line)
            elif i > start_idx + 1:  # Skip query seq line
                non_pairing_lines.append(line)

        non_pairing_path = chain_dir / "non_pairing.a3m"
        non_pairing_path.write_text("\n".join(non_pairing_lines) + "\n")

        # pairing.a3m = query sequence only
        pairing_path = chain_dir / "pairing.a3m"
        pairing_path.write_text(f">query\n{query_seq}\n")

        return {0: {"pairing": pairing_path, "non_pairing": non_pairing_path}}

    def _process_multimer(
        self, a3m_content: str, output_dir: Path
    ) -> dict[int, dict[str, Path]]:
        """Process multimer a3m — split into per-chain pairing and non_pairing."""
        lines = a3m_content.strip().split("\n")

        # Parse header to get chain lengths
        chain_lengths: list[int] = []
        chain_names: list[str] = []
        start_idx = 0

        if lines and lines[0].startswith("#"):
            header = lines[0]
            lengths_str, names_str = header[1:].split("\t")
            chain_lengths = [int(x) for x in lengths_str.split(",")]
            chain_names = [f"10{x + 1}" for x in range(len(names_str.split(",")))]
            start_idx = 1
        else:
            # No header — infer from sequences
            chain_lengths = [len(s) for s in self.sequences]
            chain_names = [f"10{x + 1}" for x in range(self.num_chains)]

        # Calculate sequence ranges for each chain
        seq_ranges: dict[str, tuple[int, int]] = {}
        for i, name in enumerate(chain_names):
            start = sum(chain_lengths[:i])
            end = sum(chain_lengths[: i + 1])
            seq_ranges[name] = (start, end)

        # Parse entries into pairing and non-pairing buckets
        pairing_a3ms: dict[str, list[str]] = {name: [] for name in chain_names}
        nonpairing_a3ms: dict[str, list[str]] = {name: [] for name in chain_names}

        current_query: str | None = None
        for line in lines[start_idx:]:
            if line.startswith(">"):
                name = line[1:]
                if name in chain_names:
                    # This is a per-chain (non-pairing) entry
                    current_query = chain_names[chain_names.index(name)]
                elif name == "\t".join(chain_names):
                    # This is a paired (cross-chain) entry
                    current_query = None

                if current_query:
                    nonpairing_a3ms[current_query].append(line)
                else:
                    for cn in chain_names:
                        pairing_a3ms[cn].append(line)
                continue

            if not line:
                continue

            if current_query:
                seq = self._extract_sequence(line, seq_ranges[current_query])
                nonpairing_a3ms[current_query].append(seq)
            else:
                for cn in chain_names:
                    seq = self._extract_sequence(line, seq_ranges[cn])
                    pairing_a3ms[cn].append(seq)

        # Write output files
        result: dict[int, dict[str, Path]] = {}

        for i, name in enumerate(chain_names):
            chain_dir = output_dir / str(i)
            chain_dir.mkdir(exist_ok=True)

            # Write non_pairing.a3m
            np_lines = nonpairing_a3ms[name]
            non_pairing_path = chain_dir / "non_pairing.a3m"
            if len(np_lines) >= 2:
                query_seq = np_lines[1]
                content = f">query\n{query_seq}\n"
                content += "\n".join(np_lines[2:])
                non_pairing_path.write_text(content + "\n")
            else:
                non_pairing_path.write_text(f">query\n{self.sequences[i]}\n")

            # Write pairing.a3m
            p_lines = pairing_a3ms[name]
            pairing_path = chain_dir / "pairing.a3m"
            if len(p_lines) >= 2:
                query_seq = p_lines[1]
                content = f">query\n{query_seq}\n"

                # Process remaining paired sequences with taxonomy-style headers
                sequences: dict[str, str] = {}
                current_name = ""
                for j, pline in enumerate(p_lines[2:]):
                    if pline.startswith(">"):
                        parts = pline[1:].split()
                        # Build UniRef-style header for Protenix pairing
                        current_name = f"UniRef100_{parts[i] if i < len(parts) else 'DUMMY'}_{j}"
                        sequences[current_name] = ""
                    elif current_name and "DUMMY" not in current_name:
                        sequences[current_name] = pline

                for seq_name, seq in sequences.items():
                    if seq:
                        content += f">{seq_name}\n{seq}\n"
                pairing_path.write_text(content)
            else:
                pairing_path.write_text(f">query\n{self.sequences[i]}\n")

            result[i] = {"pairing": pairing_path, "non_pairing": non_pairing_path}

        return result

    @staticmethod
    def _extract_sequence(line: str, range_tuple: tuple[int, int]) -> str:
        """Extract the subsequence for a specific chain from a concatenated MSA line.

        Handles insertion characters (lowercase) correctly: only uppercase
        letters and gaps (-) count toward positional indexing.
        """
        seq: list[str] = []
        no_insert_count = 0
        start, end = range_tuple

        for char in line:
            if char.isupper() or char == "-":
                no_insert_count += 1
            if start < no_insert_count <= end:
                seq.append(char)
            elif no_insert_count > end:
                break

        return "".join(seq)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fetch_and_convert_msas(
    input_json: list[dict[str, Any]],
    msa_dir: Path,
    server_url: str = DEFAULT_MSA_SERVER,
) -> list[dict[str, Any]]:
    """Fetch MSAs from ColabFold server and inject paths into Protenix input JSON.

    Args:
        input_json: Protenix input JSON (list of job dicts).
        msa_dir: Directory to write MSA files.
        server_url: ColabFold server URL.

    Returns:
        Modified input_json with pairedMsaPath and unpairedMsaPath
        injected into proteinChain entries.
    """
    for job in input_json:
        # Collect protein chain sequences and their indices
        protein_chains: list[str] = []
        protein_indices: list[int] = []
        for i, entity in enumerate(job.get("sequences", [])):
            if "proteinChain" in entity:
                chain = entity["proteinChain"]
                # Skip chains that already have MSA paths
                if chain.get("pairedMsaPath") or chain.get("unpairedMsaPath"):
                    continue
                protein_chains.append(chain["sequence"])
                protein_indices.append(i)

        if not protein_chains:
            print(f"[fetch_msa] Job '{job.get('name', '?')}': no protein chains need MSAs")
            continue

        print(
            f"[fetch_msa] Job '{job.get('name', '?')}': "
            f"fetching MSAs for {len(protein_chains)} protein chain(s)"
        )

        # Submit all protein sequences as one paired query
        ticket_id = submit_msa_query(server_url, protein_chains)
        poll_until_complete(server_url, ticket_id)
        tarball = download_result(server_url, ticket_id)

        # Extract a3m files from tarball
        a3m_files = extract_a3m_from_tarball(tarball)
        print(f"[fetch_msa] Extracted files: {list(a3m_files.keys())}")

        # Combine all a3m files into one (ColabFold produces multiple)
        # The primary file is typically the first .a3m found
        combined_a3m = ""
        for name in sorted(a3m_files.keys()):
            if name.endswith(".a3m"):
                if not combined_a3m:
                    combined_a3m = a3m_files[name]
                else:
                    # Append non-header lines from additional a3m files
                    extra_lines = a3m_files[name].strip().split("\n")
                    for line in extra_lines:
                        if not line.startswith("#") and not line.startswith(">query"):
                            combined_a3m += "\n" + line

        if not combined_a3m:
            print(f"[fetch_msa] WARNING: No a3m files found in tarball for job '{job.get('name')}'")
            continue

        # Process and convert to Protenix format
        processor = A3MProcessor(protein_chains)
        job_msa_dir = msa_dir / job.get("name", "prediction")
        chain_msas = processor.process(combined_a3m, job_msa_dir)

        # Inject MSA paths back into the input JSON
        for chain_idx, entity_idx in enumerate(protein_indices):
            msas = chain_msas.get(chain_idx, {})
            chain_data = job["sequences"][entity_idx]["proteinChain"]
            if "pairing" in msas:
                chain_data["pairedMsaPath"] = str(msas["pairing"])
            if "non_pairing" in msas:
                chain_data["unpairedMsaPath"] = str(msas["non_pairing"])

        print(
            f"[fetch_msa] Job '{job.get('name', '?')}': "
            f"wrote MSAs for {len(chain_msas)} chain(s) to {job_msa_dir}"
        )

    return input_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch MSAs from a ColabFold server and convert to Protenix format."
    )
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace directory")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_MSA_SERVER,
        help=f"ColabFold server URL (default: {DEFAULT_MSA_SERVER})",
    )
    args = parser.parse_args()

    config = json.loads((args.workspace / "config.json").read_text())
    input_json_path = Path(config["input_json_path"])
    input_json = json.loads(input_json_path.read_text())

    server_url = config.get("msa_server_url", args.server_url)
    msa_dir = args.workspace / "msa"

    print(f"[fetch_msa] Server: {server_url}")
    print(f"[fetch_msa] Input: {input_json_path}")
    print(f"[fetch_msa] MSA output: {msa_dir}")

    modified = fetch_and_convert_msas(input_json, msa_dir, server_url)

    # Overwrite the input JSON with MSA paths injected
    input_json_path.write_text(json.dumps(modified, indent=2))
    print("[fetch_msa] MSA fetch complete.")


if __name__ == "__main__":
    main()
