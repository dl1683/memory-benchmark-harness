param(
    [string]$MapUBaseUrl = "http://127.0.0.1:8000",
    [string]$RunId = "",
    [string]$OutDir = "results",
    [string]$Solver = "gemini",
    [string]$SolverModel = "gemini-3.1-flash-lite",
    [string]$Judge = "gemini",
    [string]$JudgeModel = "gemini-3.1-flash-lite",
    [int]$Concurrency = 4,
    [int]$Offset = 0,
    [int]$Limit = 0,
    [int]$SolverMaxTokens = 2048,
    [int]$JudgeMaxTokens = 512,
    [switch]$SkipLeaderboard
)

$ErrorActionPreference = "Stop"

Remove-Item Env:MEMORYBENCH_ALLOW_ORACLE -ErrorAction SilentlyContinue
Remove-Item Env:MEMORYBENCH_ALLOW_BENCHMARK_CUES -ErrorAction SilentlyContinue
Remove-Item Env:MEMORYBENCH_ENABLE_INTENT_LAYER -ErrorAction SilentlyContinue

if (($Solver -eq "gemini" -or $Judge -eq "gemini") -and -not $env:GEMINI_API_KEY) {
    Write-Error "GEMINI_API_KEY must be set in the environment for Gemini solver or judge runs."
}

try {
    $health = Invoke-RestMethod -Uri "$MapUBaseUrl/health" -TimeoutSec 5
    Write-Output "MapU health: $($health.status)"
} catch {
    Write-Error "MapU API is not reachable at $MapUBaseUrl. Start the API before running AMA-Bench."
}

if (-not $RunId) {
    $RunId = "mapu_ama_official_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$report = Join-Path $OutDir "$RunId.json"
$submission = Join-Path $OutDir "$RunId.submission.jsonl"
$expectedEpisodes = [Math]::Max(0, 208 - $Offset)
if ($Limit -gt 0) {
    $expectedEpisodes = [Math]::Min($Limit, $expectedEpisodes)
}

$runArgs = @(
    "run",
    "memorybench",
    "run",
    "--benchmarks", "ama_bench",
    "--adapter", "mapu",
    "--mapu-base-url", $MapUBaseUrl,
    "--mapu-run-id", $RunId,
    "--solver", $Solver,
    "--solver-model", $SolverModel,
    "--solver-max-tokens", "$SolverMaxTokens",
    "--judge", $Judge,
    "--judge-model", $JudgeModel,
    "--judge-max-tokens", "$JudgeMaxTokens",
    "--offset", "$Offset",
    "--limit", "$Limit",
    "--max-turns-per-scenario", "0",
    "--concurrency", "$Concurrency",
    "--out", $report
)

if ($Solver -eq "gemini") {
    $runArgs += @("--solver-api-key-env", "GEMINI_API_KEY")
}
if ($Judge -eq "gemini") {
    $runArgs += @("--judge-api-key-env", "GEMINI_API_KEY")
}

& uv @runArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$submissionArgs = @(
    "run",
    "memorybench",
    "ama-submission",
    "--report", $report,
    "--out", $submission,
    "--expected-episodes", "$expectedEpisodes",
    "--expected-questions-per-episode", "12"
)

$submissionOutput = & uv @submissionArgs
Write-Output $submissionOutput
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$submissionResult = $submissionOutput | Out-String | ConvertFrom-Json
$score = [double]$submissionResult.official_macro_accuracy

if (-not $SkipLeaderboard) {
    & uv run memorybench ama-leaderboard --kind all --compare-score $score --top 5
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Output "Official-shape report: $report"
Write-Output "Official-shape submission: $submission"
