$ErrorActionPreference = "Stop"
foreach ($variant in @("ctrl_conv", "cfpg_periph_only", "cfpg_plus_csa", "cfpg_plus_entropy_no_grn", "cfpg")) {
    python train.py --config configs/ksdd2.yaml --variant $variant --seeds 42 1024
    if ($LASTEXITCODE -ne 0) { throw "Ablation failed" }
    python train.py --config configs/mtd.yaml --variant $variant --seeds 42
    if ($LASTEXITCODE -ne 0) { throw "Ablation failed" }
}
