$ErrorActionPreference = "Stop"
foreach ($dataset in @("ksdd2", "mtd", "neu_seg")) {
    foreach ($variant in @("baseline", "cfpg")) {
        python train.py --config "configs/$dataset.yaml" --variant $variant --seeds 42 2026 3407
        if ($LASTEXITCODE -ne 0) { throw "Training failed" }
        foreach ($seed in @(42, 2026, 3407)) {
            python eval.py --config "configs/$dataset.yaml" --variant $variant --weights "results/$dataset/seed$seed/$variant/best.pth" --output "results/eval/$dataset/seed$seed/$variant"
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
        }
    }
}
