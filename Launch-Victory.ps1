$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$root = $PSScriptRoot
$stateDir = Join-Path $env:LOCALAPPDATA 'VictoryRTX'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$settings = Join-Path $stateDir 'launcher.json'
$form = New-Object Windows.Forms.Form
$form.Text = 'Victory RTX'
$form.ClientSize = New-Object Drawing.Size(640,270)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
function Add-Label($text,$y) {
    $c = New-Object Windows.Forms.Label
    $c.Text = $text; $c.SetBounds(20,$y,590,22)
    $form.Controls.Add($c)
}
function Add-Field($y) {
    $c = New-Object Windows.Forms.TextBox
    $c.SetBounds(20,$y,490,25)
    $form.Controls.Add($c)
    return $c
}
function Add-Button($text,$x,$y,$w) {
    $c = New-Object Windows.Forms.Button
    $c.Text = $text; $c.SetBounds($x,$y,$w,30)
    $form.Controls.Add($c)
    return $c
}
Add-Label 'Game folder' 15
$folder = Add-Field 38
$browseFolder = Add-Button 'Browse...' 520 35 100
Add-Label 'Game executable' 78
$exe = Add-Field 101
$browseExe = Add-Button 'Browse...' 520 98 100
$launch = Add-Button 'Launch game' 20 155 600
$status = New-Object Windows.Forms.Label
$status.SetBounds(20,198,600,55)
$status.Text = 'Select your game files. Shaders and pipelines prepare automatically.'
$form.Controls.Add($status)
if (Test-Path -LiteralPath $settings) {
    try {
        $saved = Get-Content -LiteralPath $settings -Raw | ConvertFrom-Json
        $folder.Text = $saved.folder
        $exe.Text = $saved.exe
    } catch { $status.Text = 'Saved settings could not be loaded.' }
}
$browseFolder.Add_Click({
    $dialog = New-Object Windows.Forms.FolderBrowserDialog
    try {
        if (Test-Path -LiteralPath $folder.Text -PathType Container) { $dialog.SelectedPath = $folder.Text }
        if ($dialog.ShowDialog($form) -eq 'OK') { $folder.Text = $dialog.SelectedPath }
    } finally { $dialog.Dispose() }
})
$browseExe.Add_Click({
    $dialog = New-Object Windows.Forms.OpenFileDialog
    try {
        $dialog.Filter = 'Game executable (*.exe)|*.exe'
        if (Test-Path -LiteralPath $folder.Text -PathType Container) { $dialog.InitialDirectory = $folder.Text }
        if ($dialog.ShowDialog($form) -eq 'OK') { $exe.Text = $dialog.FileName }
    } finally { $dialog.Dispose() }
})
$launch.Add_Click({
    try {
        if (!(Test-Path -LiteralPath $folder.Text -PathType Container)) { throw 'Select a valid game folder.' }
        if (!(Test-Path -LiteralPath $exe.Text -PathType Leaf) -or [IO.Path]::GetExtension($exe.Text) -ne '.exe') { throw 'Select a valid game executable.' }
        $game = (Get-Item -LiteralPath $folder.Text).FullName
        $title = (Get-Item -LiteralPath $exe.Text).FullName
        $prefix = $game.TrimEnd([char]92) + [char]92
        if (!$title.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'The executable must be inside the selected game folder.' }
        $runtime = Join-Path $root 'target\x86_64-pc-windows-msvc\release\victory.exe'
        if (!(Test-Path -LiteralPath $runtime)) { $runtime = Join-Path $root 'victory.exe' }
        if (!(Test-Path -LiteralPath $runtime)) { throw 'Victory runtime not found. Build the Windows release first or place victory.exe beside the launcher.' }
        @{folder=$game;exe=$title} | ConvertTo-Json | Set-Content -LiteralPath $settings -Encoding UTF8
        $env:VICTORY_GAME_DIR = $game
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $log = Join-Path $stateDir "victory-$stamp.log"
        $err = Join-Path $stateDir "victory-$stamp.err.log"
        $script:child = Start-Process -FilePath $runtime -ArgumentList @('run',('"' + $title + '"'),'--execute') -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
        $launch.Enabled = $false
        $status.Text = "Victory running. Preparing shaders and pipelines. Logs: $stateDir"
        $timer.Start()
    } catch { [Windows.Forms.MessageBox]::Show($_.Exception.Message,'Victory RTX') | Out-Null }
})
$timer = New-Object Windows.Forms.Timer
$timer.Interval = 1000
$timer.Add_Tick({
    if ($script:child.HasExited) {
        $timer.Stop()
        $launch.Enabled = $true
        $status.Text = "Victory exited ($($script:child.ExitCode)). Logs: $stateDir"
        $script:child.Dispose()
    }
})
try { [void]$form.ShowDialog() } finally { $timer.Dispose(); $form.Dispose() }
