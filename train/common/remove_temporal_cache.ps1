param([Parameter(Mandatory=$true)][string]$Root, [Parameter(Mandatory=$true)][string]$Target)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
$targetPath = [System.IO.Path]::GetFullPath($Target)
if (-not $targetPath.StartsWith($rootPath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Cache target is outside its reviewed owner: $targetPath"
}
$cursor = $targetPath
while ($true) {
    if (Test-Path -LiteralPath $cursor) {
        $entry = Get-Item -LiteralPath $cursor -Force
        if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing a reparse-point cache path: $cursor"
        }
    }
    if ($cursor -eq $rootPath) { break }
    $cursor = Split-Path -Parent $cursor
    if (-not $cursor) { throw 'Could not resolve the cache owner.' }
}
if (Test-Path -LiteralPath $targetPath) {
    $links = @(Get-ChildItem -LiteralPath $targetPath -Recurse -Force | Where-Object {
        ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
    })
    if ($links.Count -gt 0) { throw "Cache contains reparse points: $targetPath" }
    Remove-Item -LiteralPath $targetPath -Recurse -Force -ErrorAction Stop
}
if (Test-Path -LiteralPath $targetPath) { throw "Cache removal failed: $targetPath" }
