"""Matching by chromosome and position instead of rsID, in either genome build."""
import pandas as pd
import genesis_gwas as gg

ref = gg.load_reference("mini_chr22")
height = pd.read_csv(ref.example_path(), sep="\t")
rows = ref.lookup(height["SNP"].to_numpy(dtype=str))

# A file that identifies variants by GRCh38 coordinates only (no rsIDs).
by_pos = pd.DataFrame({
    "CHR": ["chr22"] * len(height),
    "POS": ref.pos_grch38[rows],
    "ID": [f"chr22:{p}:A:G" for p in ref.pos_grch38[rows]],
    "Z": height["Z"],
    "N": height["N"],
})
data = gg.prepare(by_pos, ref, snp="ID", qc=False)
print("matched on :", data.meta["matching"]["match_on"])
print("build used :", data.meta["matching"]["build"])
print("matches per build tried:", data.meta["matching"]["matched_by_position"])
print(f"markers    : {data.K:,} of {len(by_pos):,}")
