<#

.EXAMPLE
    ./build_vs2026.ps1
#>
Param(
    [Parameter(Mandatory = $false)]
    [String]
    $conda_env
)

if($conda_env)
{
    Write-Output "[CppBuild] Activating conda env: $conda_env"
    conda activate $conda_env
}

Write-Output "[CppBuild] Installed tools"
Write-Output "[CppBuild] $(cmake --version)"
Write-Output "[CppBuild] $(conan --version)"
Write-Output "[CppBuild] $(python --version)"
Write-Output "[CppBuild] $(python -c 'import sys; print(sys.prefix)')"

$buildJobs = 4

./generate_vs_solution.ps1 -VsVersion vs2026
$rc = $LASTEXITCODE

Write-Output $Global:buildPath
$buildPath = $Global:buildPath


if ($rc -ne 0)
{
    Write-Output "[CppBuild] !!! Build system generation failed, aborting."
    exit $rc
}


$configurations = "Debug","Release"


foreach($config in $configurations)
{
    Write-Output "[CppBuild] Build and test [$config]"

    cmake --build "$buildPath" -j "$buildJobs" --config $config
    if (! $?) {exit $LASTEXITCODE}
    cmake --build "$buildPath" -j "$buildJobs" --target RUN_TESTS --config $config
    if (! $?) {exit $LASTEXITCODE}
}
