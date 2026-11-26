param(
    [Parameter(Mandatory=$true)][string]$OutputLog,
    [Parameter(Mandatory=$true)][string]$ErrorLog,
    [Parameter(Mandatory=$true)][int]$RuntimeId
)
$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'Victory RTX - Live Runtime'
Write-Host 'VICTORY RTX | Live runtime output' -ForegroundColor Green
Write-Host "Output: $OutputLog"
Write-Host "Errors: $ErrorLog"
Write-Host 'Watching shader preparation, pipeline creation and game execution...'
Write-Host ''
function Open-Log([string]$Path) {
    $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
    return (New-Object System.IO.StreamReader -ArgumentList $stream)
}
$out = $null
$err = $null
try {
    $out = Open-Log $OutputLog
    $err = Open-Log $ErrorLog
    $runtime = Get-Process -Id $RuntimeId -ErrorAction SilentlyContinue
    while ($true) {
        $running = ($null -ne $runtime -and -not $runtime.HasExited)
        $normal = $out.ReadToEnd()
        $diagnostics = $err.ReadToEnd()
        if ($normal.Length -gt 0) { Write-Host -NoNewline $normal }
        if ($diagnostics.Length -gt 0) { Write-Host -NoNewline $diagnostics }
        if (-not $running) { break }
        Start-Sleep -Milliseconds 250
    }
    Write-Host ''
    Write-Host 'Victory has exited. See the launcher for its exit code.' -ForegroundColor Yellow
    Write-Host 'The logs have been retained. You can close this terminal.'
} catch {
    Write-Host "Live viewer error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Victory can continue running. Check the saved logs directly.'
} finally {
    if ($null -ne $out) { $out.Dispose() }
    if ($null -ne $err) { $err.Dispose() }
}
