param(
    [ValidateSet('balanced', 'moderate', 'wide')]
    [string]$Preset = 'balanced',
    [int]$TimeoutSeconds = 120,
    [string]$ProfileName = '',
    [string]$PythonExe = '.\.venv\Scripts\python.exe',
    [string]$StepPath = '.\src\ade3x3\steps\ade3x3_step84_metaheuristic_rank19_search.py'
)

$ErrorActionPreference = 'Stop'

function Get-LogicalCoreCount {
    $cpu = Get-CimInstance Win32_ComputerSystem
    if ($null -ne $cpu.NumberOfLogicalProcessors -and $cpu.NumberOfLogicalProcessors -gt 0) {
        return [int]$cpu.NumberOfLogicalProcessors
    }
    return 1
}

function New-Step84Environment {
    param([string]$PresetName, [int]$BudgetSeconds)

    $envMap = @{
        'STEP84_GENERATIONS' = '1000000'
        'STEP84_WORKERS' = '24'
        'STEP84_TIMEOUT_SECONDS' = [string]$BudgetSeconds
        'STEP84_CHECKPOINT_INTERVAL' = '100'
        'STEP84_ALS_SWEEPS' = '1'
        'STEP84_SIGNATURE333_SURVIVOR_FRACTION' = '0.15'
        'STEP84_EXECUTOR_KIND' = 'thread'
        'STEP84_OFFSPRING_MULTIPLIER' = '1'
        'STEP84_BEST_EXPORT_INCLUDE_DENSE' = '0'
        'STEP84_SHADOW_METRICS_INTERVAL' = '20'
    }

    switch ($PresetName) {
        'balanced' {
            $envMap['STEP84_ISLAND_POPULATIONS'] = '32,24,16,12,8,8,6,6'
        }
        'moderate' {
            $envMap['STEP84_ISLAND_MIN_EXP'] = '2'
            $envMap['STEP84_ISLAND_MAX_EXP'] = '8'
            $envMap['STEP84_ISLAND_COPIES'] = '2'
        }
        'wide' {
            $envMap['STEP84_ISLAND_MIN_EXP'] = '2'
            $envMap['STEP84_ISLAND_MAX_EXP'] = '9'
            $envMap['STEP84_ISLAND_COPIES'] = '2'
        }
        default {
            throw "Unknown preset: $PresetName"
        }
    }

    return $envMap
}

if ([string]::IsNullOrWhiteSpace($ProfileName)) {
    $ProfileName = $Preset
}

$workspaceRoot = (Get-Location).Path
$exportsDir = Join-Path $workspaceRoot 'outputs\exports\step84_resource_profiles'
New-Item -ItemType Directory -Force -Path $exportsDir | Out-Null

$summaryPath = Join-Path $workspaceRoot 'outputs\exports\step84_summary.json'
$profileCsvPath = Join-Path $exportsDir ("step84_profile_{0}.csv" -f $ProfileName)
$profileJsonPath = Join-Path $exportsDir ("step84_profile_{0}.json" -f $ProfileName)
$stdoutPath = Join-Path $exportsDir ("step84_profile_{0}_stdout.log" -f $ProfileName)
$stderrPath = Join-Path $exportsDir ("step84_profile_{0}_stderr.log" -f $ProfileName)

$envMap = New-Step84Environment -PresetName $Preset -BudgetSeconds $TimeoutSeconds
$logicalCores = Get-LogicalCoreCount

$previousSummaryTimestamp = if (Test-Path $summaryPath) { (Get-Item $summaryPath).LastWriteTimeUtc } else { $null }

$originalEnv = @{}
foreach ($entry in $envMap.GetEnumerator()) {
    $originalEnv[$entry.Key] = [System.Environment]::GetEnvironmentVariable($entry.Key, 'Process')
    [System.Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, 'Process')
}

try {
    $process = Start-Process -FilePath $PythonExe -ArgumentList $StepPath -WorkingDirectory $workspaceRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
} finally {
    foreach ($entry in $envMap.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($entry.Key, $originalEnv[$entry.Key], 'Process')
    }
}

$samples = New-Object System.Collections.Generic.List[object]

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$lastCpuSeconds = $null
$lastSampleTime = $null

while (-not $process.HasExited) {
    Start-Sleep -Milliseconds 1000
    $process.Refresh()
    if ($process.HasExited) { break }

    $now = Get-Date
    $cpuSeconds = $process.TotalProcessorTime.TotalSeconds
    $cpuPercent = $null
    if ($null -ne $lastCpuSeconds -and $null -ne $lastSampleTime) {
        $elapsedSample = ($now - $lastSampleTime).TotalSeconds
        if ($elapsedSample -gt 0) {
            $cpuPercent = (($cpuSeconds - $lastCpuSeconds) / ($elapsedSample * $logicalCores)) * 100.0
        }
    }

    $samples.Add([pscustomobject]@{
        second = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        cpu_percent = if ($null -eq $cpuPercent) { $null } else { [math]::Round($cpuPercent, 3) }
        working_set_mb = [math]::Round($process.WorkingSet64 / 1MB, 3)
        private_memory_mb = [math]::Round($process.PrivateMemorySize64 / 1MB, 3)
        threads = $process.Threads.Count
        handles = $process.HandleCount
    }) | Out-Null

    $lastCpuSeconds = $cpuSeconds
    $lastSampleTime = $now
}

$process.WaitForExit()
$sw.Stop()

$samples | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $profileCsvPath

$newSummaryAvailable = $false
if (Test-Path $summaryPath) {
    $summaryTimestamp = (Get-Item $summaryPath).LastWriteTimeUtc
    if ($null -eq $previousSummaryTimestamp -or $summaryTimestamp -gt $previousSummaryTimestamp) {
        $newSummaryAvailable = $true
    }
}

$stepSummary = $null
if ($newSummaryAvailable) {
    $stepSummary = Get-Content $summaryPath -Raw | ConvertFrom-Json
}

$peakWorkingSet = ($samples | Measure-Object -Property working_set_mb -Maximum).Maximum
$peakPrivateMemory = ($samples | Measure-Object -Property private_memory_mb -Maximum).Maximum
$meanCpu = ($samples | Where-Object { $null -ne $_.cpu_percent } | Measure-Object -Property cpu_percent -Average).Average
$stdoutLines = if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Select-Object -Last 20 } else { @() }
$stderrLines = if (Test-Path $stderrPath) { Get-Content $stderrPath | Select-Object -Last 20 } else { @() }

$profile = [ordered]@{
    profile_name = $ProfileName
    preset = $Preset
    timeout_seconds = $TimeoutSeconds
    python_exe = $PythonExe
    step_path = $StepPath
    logical_cores = $logicalCores
    env = $envMap
    process_exit_code = $process.ExitCode
    elapsed_wall_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
    sample_count = $samples.Count
    peak_working_set_mb = if ($null -eq $peakWorkingSet) { $null } else { [math]::Round($peakWorkingSet, 3) }
    peak_private_memory_mb = if ($null -eq $peakPrivateMemory) { $null } else { [math]::Round($peakPrivateMemory, 3) }
    mean_cpu_percent = if ($null -eq $meanCpu) { $null } else { [math]::Round($meanCpu, 3) }
    stdout_log_path = $stdoutPath
    stderr_log_path = $stderrPath
    stdout_tail = $stdoutLines
    stderr_tail = $stderrLines
    summary_path = if ($newSummaryAvailable) { $summaryPath } else { $null }
    step84_summary = $stepSummary
}

$profile | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $profileJsonPath

Write-Host ("Profile written: {0}" -f $profileJsonPath)
Write-Host ("Samples written: {0}" -f $profileCsvPath)
Write-Host ("Peak working set MB: {0}" -f $profile.peak_working_set_mb)
Write-Host ("Peak private memory MB: {0}" -f $profile.peak_private_memory_mb)
Write-Host ("Mean CPU %: {0}" -f $profile.mean_cpu_percent)
if ($null -ne $stepSummary) {
    Write-Host ("Step84 evaluations: {0}" -f $stepSummary.total_evaluations)
    Write-Host ("Step84 best fitness: {0}" -f $stepSummary.best_fitness_ever)
}