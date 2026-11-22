"""Print two PREFIX.estimates.tsv files side by side:  compare_estimates.py A.tsv B.tsv"""
import sys
import pandas as pd

a, b = (pd.read_csv(path, sep="\t") for path in sys.argv[1:3])
model = a["model"].iloc[0]
keep = ["pic", "p1", "sigmasq", "sigmasq1", "sigmasq2", "a", "n_ssnp", "heritability", "loglik", "n_iter"]
a, b = (t[(t["model"] == model) & t["quantity"].isin(keep)].set_index("quantity") for t in (a, b))
out = pd.DataFrame({"estimate_A": a["estimate"], "se_A": a["se"], "estimate_B": b["estimate"], "se_B": b["se"]})
out["B/A - 1"] = out["estimate_B"] / out["estimate_A"] - 1
print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n")
print(out.to_string(float_format=lambda v: f"{v:.6g}", na_rep=""))
