import genesis_gwas as gg

ref = gg.load_reference()
data = gg.prepare("height_giant2010.sumstats.tsv.gz", ref)

m2 = gg.fit(data, 2)                  # two-component model
m3 = gg.fit(data, 3, starting=m2)     # three-component model, started from the M2 estimate
print(m3.summary())

choice = gg.select_model([m2, m3])    # modified BIC
print()
print(choice.summary())

m2.to_json("results/py_height.M2.json")
m3.to_json("results/py_height.M3.json")
