import numpy as np
from pathlib import Path
from pyriemann.clustering import PotatoField

IN_DIR = Path("/home/alouis/scratch/dream_features_noica_1000hz_overlap")
KEYS = ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma", "cosp_beta"]
SUB_ID = "06"   # "groupe 6" — sujet qui a crashe avec le potato par-feature
STATE = "REM"   # stade atomique reel (pas de regroupement ici, on teste au niveau atomique)

X = []
for key in KEYS:
    f = IN_DIR / key / f"{key}_s{SUB_ID}_{STATE}.npz"
    d = np.load(f)
    X.append(d["data"])
    print(f"{key}: {d['data'].shape}")

n_epochs = [x.shape[0] for x in X]
assert len(set(n_epochs)) == 1, f"n_epochs differents avant filtrage : {dict(zip(KEYS, n_epochs))}"

pf = PotatoField(n_potatoes=len(KEYS), p_threshold=0.01, z_threshold=3, n_iter_max=300)
pf.fit(X)
mask = pf.predict(X)
proba = pf.predict_proba(X)

print(f"\nEpochs gardees : {mask.sum()}/{len(mask)} ({100*mask.mean():.1f}%)")
print(f"Distribution proba (p-value Fisher) : min={proba.min():.4f}, "
      f"mediane={np.median(proba):.4f}, max={proba.max():.4f}")
print(f"Epochs sous p_threshold=0.01 : {(proba < 0.01).sum()}")
