<#
.PARAMETER EnableKudanSLAM
    If set to true, will build with KudanSLAM support. 
    Disabled by default.

.EXAMPLE
    ./build_vs2019.ps1 -enablekudanslam true
#>
Param(
    [Parameter(Mandatory = $false)]
    [String]
    $conda_env,
    [Parameter(Mandatory = $false)]
    [String]
    $EnableKudanSLAM
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

./generate_vs_solution.ps1 -VsVersion vs2019 -enablekudanslam $EnableKudanSLAM
$rc = $LASTEXITCODE

Write-Output $Global:buildPath
$buildPath = $Global:buildPath


if ($rc -ne 0)
{
    Write-Output "[CppBuild] !!! Build system generation failed, aborting."
    exit $rc
}


# KudanSLAM has only release libraries
if ("" -eq $EnableKudanSLAM)
{
    $configurations = "Debug","Release"
}
else {
    $configurations = "Release"
}

foreach($config in $configurations)
{
    Write-Output "[CppBuild] Build and test [$config]"

    cmake --build "$buildPath" -j "$buildJobs" --config $config
    if (! $?) {exit $LASTEXITCODE}
    cmake --build "$buildPath" -j "$buildJobs" --target RUN_TESTS --config $config
    if (! $?) {exit $LASTEXITCODE}
}
