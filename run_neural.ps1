# Create logs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "===== Running neural_models.py ====="
python src/neural_models.py `
    --root . `
    --texts texts_strict `
    --epochs 12 `
    --repeats 3 `
    --batch-size 8 2>&1 | Tee-Object -FilePath "logs/neural_models.txt"

if ($LASTEXITCODE -ne 0) {
    Write-Host "neural_models.py FAILED with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n===== Running transformer_malayalam.py ====="
python src/transformer_malayalam.py `
    --root . `
    --texts texts_strict `
    --model google/muril-base-cased `
    --epochs 3 `
    --repeats 1 `
    --batch-size 2 `
    --grad-accum 8 2>&1 | Tee-Object -FilePath "logs/transformer_malayalam.txt"

if ($LASTEXITCODE -ne 0) {
    Write-Host "transformer_malayalam.py FAILED with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n===== Running ablations_malayalam.py ====="
python src/ablations_malayalam.py `
    --root . `
    --texts texts_strict `
    --repeats 3 `
    --author-repeats 5 2>&1 | Tee-Object -FilePath "logs/ablations_malayalam.txt"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ablations_malayalam.py FAILED with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n===== ALL EXPERIMENTS COMPLETED ====="