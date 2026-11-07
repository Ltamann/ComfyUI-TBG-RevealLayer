[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$nodeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not (Test-Path -LiteralPath (Join-Path $nodeRoot 'model_downloads.py'))) {
    throw 'Run this script from the TBG RevealLayer scripts folder.'
}
if ((Get-Item -LiteralPath $nodeRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'The node root must not be a link or junction.'
}

$preserved = @{
    'pytorch_lora_weights.safetensors' = 'f5069b954306f882d384c059fbc07b931d2d4d5c4d6d4d16e8d888f8c512bc7c'
    'layer_pe.pt' = 'c3144b547bf8dad8246ee0070c4ea37b7d12cf5aa54f0aacae45ba3cc12cf6f0'
    'Refiner.pt' = '752aff6cb62a44e35098b7e21eabf93744eb204e4ebb6d9615e3d1daffe40d3c'
    'xvae\transparent_decoder_ckpt.pth' = '44653f514096dedf906354310d21ae9a62c812d79ac69f8dc4e6d7b8575ee8c3'
}
foreach ($entry in $preserved.GetEnumerator()) {
    $modelFile = Join-Path (Join-Path $nodeRoot 'models\reveallayer') $entry.Key
    if (-not (Test-Path -LiteralPath $modelFile)) {
        throw "Preserved model is missing; run the node first: $modelFile"
    }
    if ((Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash -ne $entry.Value) {
        throw "Preserved model failed verification: $modelFile"
    }
}

$obsolete = @(
    'RevealLayer', 'worker_deps', '.reveallayer-venv', 'tmp',
    '.pytest_cache', '__pycache__', 'tests\__pycache__', 'scripts\__pycache__',
    'reveallayer_worker.log', 'vit_b_32-d86f8d99.pth',
    'assets\upstream\reveallayer-logo.png'
)
foreach ($relative in $obsolete) {
    $target = [IO.Path]::GetFullPath((Join-Path $nodeRoot $relative))
    if (-not $target.StartsWith($nodeRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escaped the node folder: $target"
    }
    if (-not (Test-Path -LiteralPath $target)) { continue }
    $item = Get-Item -LiteralPath $target -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Refusing to remove a link or junction: $target"
    }
    if ($item.PSIsContainer) {
        $links = @(Get-ChildItem -LiteralPath $target -Force -Recurse -Attributes ReparsePoint)
        if ($links.Count) { throw "Refusing a cleanup tree containing links: $target" }
    }
    if ($PSCmdlet.ShouldProcess($target, 'Remove obsolete RevealLayer remnant')) {
        if ($item.PSIsContainer) {
            Remove-Item -LiteralPath $target -Force -Recurse
        } else {
            Remove-Item -LiteralPath $target -Force
        }
    }
}
