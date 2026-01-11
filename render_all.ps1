$scriptDir = $PSScriptRoot
New-Item -ItemType Directory -Force output
Get-ChildItem scenes -Filter *.txt | ForEach-Object {
    $scene = $_.FullName
    $out = "output/$($_.BaseName)_our_result.png"
    Start-Job -ScriptBlock { param($s, $o, $d) Set-Location $d; python ray_tracer.py $s $o --height 500 --width 500 } -ArgumentList $scene, $out, $scriptDir
}
Get-Job | Wait-Job
Get-Job | Receive-Job