<#

.SYNOPSIS
    Generate a Visual Solution with CMake.

.DESCRIPTION
    Generate a Visual Solution with CMake.
    The name of the build root is derived from the Visual Studio version and the
    python interpreter version.

.PARAMETER VsVersion
    VsVersion can be one of: vs2022, vs2019
    If no argument is given, vs2022 will be used by default.

.EXAMPLE
    ./generate_vs_solution.ps1 vs2022
#>
Param(
    [Parameter(Mandatory = $false)]
    [String]
    $VsVersion
)


if ("" -eq $VsVersion)
{
    $VsVersion = "vs2022"
}

$generators = @{
    vs2026 = "Visual Studio 18 2026";
    vs2022 = "Visual Studio 17 2022";
    vs2019 = "Visual Studio 16 2019";
}



$sourceDir = "$PSScriptRoot/../.."
$buildRoot = "$sourceDir/.build"

$pythonVersion = (python -c "from distutils import sysconfig; print(sysconfig.get_config_var('VERSION'))").trim()

$buildName = "_build_${VsVersion}_py${pythonVersion}"
$buildPath = "$buildRoot/$buildName"


if(!(test-path $buildPath))
{
    mkdir $buildPath
}

Push-Location .
$Global:buildPath = $buildPath
Set-Location $buildPath

Write-Output "[CppBuild] Generating visual studio solution for $VsVersion"
$generator = $generators[$VsVersion]

cmake $sourceDir -G $generator -A x64


$rc = $LASTEXITCODE

Pop-Location

exit $rc
