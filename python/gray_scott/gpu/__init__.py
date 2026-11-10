import os

# No FMA contraction by default, matching the CPU tracks (C++ -ffp-contract=off)
# and staying fixed when Triton 3.10 changes its default.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
