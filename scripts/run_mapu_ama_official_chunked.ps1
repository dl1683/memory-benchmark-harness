param(
    [string]$MapUBaseUrl = "http://127.0.0.1:8000",
    [string]$RunId = "",
    [string]$OutDir = "results",
    [string]$Solver = "gemini",
    [string]$SolverModel = "gemini-3.1-flash-lite",
    [string]$Judge = "gemini",
    [string]$JudgeModel = "gemini-3.1-flash-lite",
    [int]$Concurrency = 4,
    [int]$ChunkWorkers = 1,
    [int]$ChunkSize = 8,
    [int]$StartOffset = 0,
    [int]$MaxChunks = 0,
    [int]$SolverMaxTokens = 2048,
    [int]$JudgeMaxTokens = 512,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

if ($ChunkSize -le 0) {
    Write-Error "ChunkSize must be positive."
}

if ($ChunkWorkers -le 0) {
    Write-Error "ChunkWorkers must be positive."
}

if (-not $RunId) {
    $RunId = "mapu_ama_official_chunked_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$chunks = @()
$chunkIndex = 0

for ($offset = $StartOffset; $offset -lt 208; $offset += $ChunkSize) {
    if ($MaxChunks -gt 0 -and $chunkIndex -ge $MaxChunks) {
        break
    }

    $limit = [Math]::Min($ChunkSize, 208 - $offset)
    $chunkRunId = "${RunId}_offset${offset}_limit${limit}"
    $report = Join-Path $OutDir "$chunkRunId.json"
    $submission = Join-Path $OutDir "$chunkRunId.submission.jsonl"

    $chunks += [pscustomobject]@{
        Offset = $offset
        Limit = $limit
        RunId = $chunkRunId
        Report = $report
        Submission = $submission
    }

    $chunkIndex += 1
}

if ($chunks.Count -eq 0) {
    Write-Error "No chunks were selected."
}

$repoRoot = (Get-Location).Path
$pendingChunks = [System.Collections.Generic.Queue[object]]::new()
foreach ($chunk in $chunks) {
    if ($Resume -and (Test-Path -LiteralPath $chunk.Report) -and (Test-Path -LiteralPath $chunk.Submission)) {
        Write-Output "Skipping existing chunk offset=$($chunk.Offset) limit=$($chunk.Limit) report=$($chunk.Report)"
    } else {
        $pendingChunks.Enqueue($chunk)
    }
}

$runningJobs = @()
$failed = $false

while ($pendingChunks.Count -gt 0 -or $runningJobs.Count -gt 0) {
    while (-not $failed -and $pendingChunks.Count -gt 0 -and $runningJobs.Count -lt $ChunkWorkers) {
        $chunk = $pendingChunks.Dequeue()
        Write-Output "Starting chunk offset=$($chunk.Offset) limit=$($chunk.Limit) worker=$($runningJobs.Count + 1)/$ChunkWorkers"

        $job = Start-Job -ArgumentList @(
            $repoRoot,
            $MapUBaseUrl,
            $chunk.RunId,
            $OutDir,
            $Solver,
            $SolverModel,
            $Judge,
            $JudgeModel,
            $Concurrency,
            $chunk.Offset,
            $chunk.Limit,
            $SolverMaxTokens,
            $JudgeMaxTokens
        ) -ScriptBlock {
            param(
                $RepoRoot,
                $MapUBaseUrl,
                $RunId,
                $OutDir,
                $Solver,
                $SolverModel,
                $Judge,
                $JudgeModel,
                $Concurrency,
                $Offset,
                $Limit,
                $SolverMaxTokens,
                $JudgeMaxTokens
            )

            Set-Location -LiteralPath $RepoRoot
            powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_mapu_ama_official.ps1 `
                -MapUBaseUrl $MapUBaseUrl `
                -RunId $RunId `
                -OutDir $OutDir `
                -Solver $Solver `
                -SolverModel $SolverModel `
                -Judge $Judge `
                -JudgeModel $JudgeModel `
                -Concurrency $Concurrency `
                -Offset $Offset `
                -Limit $Limit `
                -SolverMaxTokens $SolverMaxTokens `
                -JudgeMaxTokens $JudgeMaxTokens `
                -SkipLeaderboard

            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }

        $job | Add-Member -NotePropertyName ChunkOffset -NotePropertyValue $chunk.Offset
        $job | Add-Member -NotePropertyName ChunkLimit -NotePropertyValue $chunk.Limit
        $runningJobs += $job
    }

    if ($runningJobs.Count -eq 0) {
        break
    }

    $completedJob = Wait-Job -Job $runningJobs -Any
    $jobOutput = Receive-Job -Job $completedJob
    if ($jobOutput) {
        Write-Output $jobOutput
    }

    if ($completedJob.State -ne "Completed") {
        Write-Error "Chunk offset=$($completedJob.ChunkOffset) limit=$($completedJob.ChunkLimit) failed with state=$($completedJob.State)."
        $failed = $true
    }

    Remove-Job -Job $completedJob -Force
    $runningJobs = @($runningJobs | Where-Object { $_.Id -ne $completedJob.Id })
}

if ($failed) {
    foreach ($job in $runningJobs) {
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
    exit 1
}

$reports = @($chunks | ForEach-Object { $_.Report })
$expectedEpisodes = ($chunks | Measure-Object -Property Limit -Sum).Sum

if ($reports.Count -eq 0) {
    Write-Error "No chunk reports were produced or selected."
}

$mergedSubmission = Join-Path $OutDir "$RunId.merged.submission.jsonl"
$mergeArgs = @(
    "run",
    "memorybench",
    "ama-submission",
    "--report"
) + $reports + @(
    "--out", $mergedSubmission,
    "--expected-episodes", "$expectedEpisodes",
    "--expected-questions-per-episode", "12"
)

$mergeOutput = & uv @mergeArgs
Write-Output $mergeOutput
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$mergeResult = $mergeOutput | Out-String | ConvertFrom-Json
$score = [double]$mergeResult.official_macro_accuracy

& uv run memorybench ama-leaderboard --kind all --compare-score $score --top 5
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Merged official-shape submission: $mergedSubmission"
Write-Output "Merged report count: $($reports.Count)"
