# =====================================================================
#  run_diagnostics.ps1
#
#  Runs the remaining low-baselines sequence across both corpora.
#  Assumes you have ALREADY run, in sayahna-essays:
#      find_boilerplate_v2.py --strip --strip-named
#      low_baselines_v3.py --texts corpus/texts
#
#  Usage, from anywhere in a VS Code PowerShell terminal:
#      .\run_diagnostics.ps1
#      .\run_diagnostics.ps1 -SkipDone        # skip steps whose output exists
#      .\run_diagnostics.ps1 -WhatIf          # print the plan, run nothing
#
#  If PowerShell refuses to run the file:
#      powershell -ExecutionPolicy Bypass -File .\run_diagnostics.ps1
# =====================================================================

[CmdletBinding()]
param(
    [string]$Root    = "C:\Users\CSE\Documents\Authorship-Attribution-MTech",
    [string]$Src     = "C:\Users\CSE\Documents\Authorship-Attribution-MTech\src",
    [string]$Python  = "python",
    [switch]$SkipDone,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# Malayalam in the console. Without this, printing the feature lists can throw
# UnicodeEncodeError on a cp1252 Windows console.
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Essays   = Join-Path $Root "sayahna-essays"
$Fiction  = Join-Path $Root "sayahna-fiction"
$LowBase  = Join-Path $Src  "low_baselines_v3.py"
$Boiler   = Join-Path $Src  "find_boilerplate_v2.py"
$LogDir   = Join-Path $Root "logs"
$Stamp    = Get-Date -Format "yyyyMMdd-HHmmss"

# ---------------------------------------------------------------- checks
foreach ($p in @($Essays, $Fiction, $LowBase, $Boiler)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# ---------------------------------------------------------------- plan
# Each step: a label, the folder to run in, the script, its arguments, and
# the file it is expected to produce (used by -SkipDone).
$Steps = @(
    @{ N="essays  / clean";  Dir=$Essays;  Script=$LowBase; Args=@("--texts","corpus/texts_clean");
       Out="corpus\prelim\low-baselines-clean.txt" }

    @{ N="essays  / strict"; Dir=$Essays;  Script=$LowBase; Args=@("--texts","corpus/texts_strict");
       Out="corpus\prelim\low-baselines-strict.txt" }

    @{ N="fiction / strip";  Dir=$Fiction; Script=$Boiler;  Args=@("--strip","--strip-named");
       Out="corpus\texts_strict" }

    @{ N="fiction / raw";    Dir=$Fiction; Script=$LowBase; Args=@("--texts","corpus/texts");
       Out="corpus\prelim\low-baselines.txt" }

    @{ N="fiction / clean";  Dir=$Fiction; Script=$LowBase; Args=@("--texts","corpus/texts_clean");
       Out="corpus\prelim\low-baselines-clean.txt" }

    @{ N="fiction / strict"; Dir=$Fiction; Script=$LowBase; Args=@("--texts","corpus/texts_strict");
       Out="corpus\prelim\low-baselines-strict.txt" }
)

Write-Host ""
Write-Host ("=" * 70)
Write-Host " DIAGNOSTIC SEQUENCE - $($Steps.Count) steps"
Write-Host ("=" * 70)
Write-Host " root   : $Root"
Write-Host " logs   : $LogDir"
Write-Host ""

if ($WhatIf) {
    $i = 0
    foreach ($s in $Steps) {
        $i++
        Write-Host ("  {0}. [{1}]  {2} {3}" -f $i, $s.N,
                    (Split-Path $s.Script -Leaf), ($s.Args -join " "))
        Write-Host ("      in {0}" -f $s.Dir)
    }
    Write-Host "`n  -WhatIf: nothing was run."
    return
}

# ---------------------------------------------------------------- run
$Results = @()
$i = 0
$TotalStart = Get-Date

foreach ($s in $Steps) {
    $i++
    $header = "[{0}/{1}]  {2}" -f $i, $Steps.Count, $s.N
    Write-Host ""
    Write-Host ("-" * 70) -ForegroundColor DarkGray
    Write-Host $header -ForegroundColor Cyan
    Write-Host ("   {0} {1}" -f (Split-Path $s.Script -Leaf), ($s.Args -join " ")) -ForegroundColor DarkGray

    $expected = Join-Path $s.Dir $s.Out
    if ($SkipDone -and (Test-Path $expected)) {
        Write-Host "   already present, skipping (-SkipDone)" -ForegroundColor Yellow
        $Results += [pscustomobject]@{ Step=$s.N; Status="skipped"; Seconds=0 }
        continue
    }

    $logName = "{0}_{1}.log" -f $Stamp, ($s.N -replace '[^\w]+','_')
    $logPath = Join-Path $LogDir $logName
    $start   = Get-Date

    Push-Location $s.Dir
    try {
        & $Python $s.Script @($s.Args) 2>&1 | Tee-Object -FilePath $logPath
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    $secs = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)

    if ($code -ne 0) {
        Write-Host ("   FAILED (exit {0}) after {1}s" -f $code, $secs) -ForegroundColor Red
        Write-Host ("   log: {0}" -f $logPath) -ForegroundColor Red
        $Results += [pscustomobject]@{ Step=$s.N; Status="FAILED"; Seconds=$secs }
        Write-Host ""
        Write-Host "Stopping. Fix the error above, then re-run with -SkipDone" -ForegroundColor Red
        break
    }

    if (Test-Path $expected) {
        Write-Host ("   done in {0}s  ->  {1}" -f $secs, $s.Out) -ForegroundColor Green
        $Results += [pscustomobject]@{ Step=$s.N; Status="ok"; Seconds=$secs }
    } else {
        Write-Host ("   finished in {0}s but {1} was not created" -f $secs, $s.Out) -ForegroundColor Yellow
        $Results += [pscustomobject]@{ Step=$s.N; Status="no output"; Seconds=$secs }
    }
}

# ---------------------------------------------------------------- summary
$total = [math]::Round(((Get-Date) - $TotalStart).TotalMinutes, 1)
Write-Host ""
Write-Host ("=" * 70)
Write-Host " SUMMARY   (total $total min)"
Write-Host ("=" * 70)
$Results | Format-Table -AutoSize

Write-Host "Output files:"
foreach ($d in @($Essays, $Fiction)) {
    $p = Join-Path $d "corpus\prelim"
    if (Test-Path $p) {
        Write-Host ("  {0}" -f $p)
        Get-ChildItem $p -Filter "low-baselines*.txt" |
            ForEach-Object { Write-Host ("     {0}  ({1:N0} bytes)" -f $_.Name, $_.Length) }
    }
}

Write-Host ""
Write-Host "Next: build the comparison table with"
Write-Host ("  python {0} --root {1}" -f (Join-Path $Src "collect_results.py"), $Root)
Write-Host ""
