# Launch an experimental NVIDIA profile without changing the Radeon baseline.
# Install into an isolated directory first:
#   .\Setup.cmd -InstallDir "$env:LOCALAPPDATA\Victory RTX"
# Then run:
#   .\Start-RTX.ps1
#   .\Start-RTX.ps1 -Mode StaticBVH
[CmdletBinding()]
param(
    [ValidateSet('Baseline', 'StaticBVH')]
    [string] $Mode = 'Baseline',
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'Victory RTX')
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $InstallDir 'victory-play.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw 'Install Victory RTX first using Setup.cmd -InstallDir with the same directory.'
}
$rtFlag = if ($Mode -eq 'StaticBVH') { '--rt-static-bvh' } else { '--no-rt' }
$previousFsr = [Environment]::GetEnvironmentVariable('VICTORY_FSR_4K', 'Process')
$previousAmbient = [Environment]::GetEnvironmentVariable('VICTORY_AMBIENT_FALLBACK', 'Process')
try {
    [Environment]::SetEnvironmentVariable('VICTORY_FSR_4K', '0', 'Process')
    [Environment]::SetEnvironmentVariable('VICTORY_AMBIENT_FALLBACK', '1', 'Process')
    Write-Host "Victory RTX experimental profile: $Mode ($rtFlag). FSR disabled."
    Write-Host "Runtime: $launcher"
    Write-Host "Logs: $(Join-Path $InstallDir 'logs')"
    Push-Location -LiteralPath $InstallDir
    try {
        & $launcher $rtFlag --console series-x
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    [Environment]::SetEnvironmentVariable('VICTORY_FSR_4K', $previousFsr, 'Process')
    [Environment]::SetEnvironmentVariable('VICTORY_AMBIENT_FALLBACK', $previousAmbient, 'Process')
}
exit $code
