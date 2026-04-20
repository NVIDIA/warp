<#
Copyright 2019-2026 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
#>

[CmdletBinding()]
param()

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$PM_PACKMAN_VERSION = "8.2.2"
$PM_PYTHON_VERSION = "3.12.13-nv1-windows-x86_64"
$BOOTSTRAP_ROOT = $PSScriptRoot
$PM_INSTALL_PATH = [System.IO.Path]::GetFullPath((Join-Path $BOOTSTRAP_ROOT ".."))
$SENTINEL_FILE_NAME = ".packman.rdy"
$LOCK_WAIT_LOG_INTERVAL_MS = 5000

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

[System.Net.ServicePointManager]::SecurityProtocol = `
    [System.Net.ServicePointManager]::SecurityProtocol -bor `
    [System.Net.SecurityProtocolType]::Tls12

function Write-Log {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Write-Failure {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Get-DefaultPackagesRoot {
    $drive = [System.IO.Path]::GetPathRoot($PM_INSTALL_PATH)
    if ([string]::IsNullOrEmpty($drive)) {
        throw "Unable to resolve install drive from '$PM_INSTALL_PATH'."
    }

    return [System.IO.Path]::Combine($drive, "packman-repo")
}

function Set-UserEnvironmentVariable {
    param(
        [string]$Name,
        [string]$Value
    )

    Write-Log "Setting user environment variable $Name to $Value"
    $null = & setx $Name $Value
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to persist user environment variable $Name."
    }
}

function Write-VisualStudioWarning {
    if ($env:PM_DISABLE_VS_WARNING) {
        return
    }

    if (-not $env:VSLANG) {
        return
    }

    Write-Log "The above is a once-per-computer operation. Unfortunately VisualStudio cannot pick up environment change"
    Write-Log "unless *VisualStudio is RELAUNCHED*."
    Write-Log "If you are launching VisualStudio from command line or command line utility make sure"
    Write-Log "you have a fresh launch environment (relaunch the command line or utility)."
    Write-Log "If you are using 'linkPath' and referring to packages via local folder links you can safely ignore this warning."
    Write-Log "You can disable this warning by setting the environment variable PM_DISABLE_VS_WARNING."
    Write-Log ""
}

function Ensure-DirectoryExists {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Resolve-NormalizedPath {
    param([string]$Path)

    return [System.IO.Path]::GetFullPath($Path)
}

function Remove-DirectoryIfPresent {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Get-SentinelPath {
    param([string]$DestinationDir)

    return [System.IO.Path]::Combine($DestinationDir, $SENTINEL_FILE_NAME)
}

function Test-Ready {
    param([string]$DestinationDir)

    return Test-Path -LiteralPath (Get-SentinelPath -DestinationDir $DestinationDir)
}

function Get-PathHash {
    param([string]$Path)

    $normalizedPath = [System.IO.Path]::GetFullPath($Path).ToLowerInvariant()
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($normalizedPath)
        $hash = $sha256.ComputeHash($bytes)
    }
    finally {
        $sha256.Dispose()
    }

    return ([System.BitConverter]::ToString($hash).Replace("-", "")).ToLowerInvariant()
}

function Get-LockFilePath {
    param(
        [string]$PackagesRoot,
        [string]$DestinationDir
    )

    $lockRoot = Join-Path $PackagesRoot ".locks"
    Ensure-DirectoryExists -Path $lockRoot
    return Join-Path $lockRoot ((Get-PathHash -Path $DestinationDir) + ".txt")
}

function Write-LockMetadata {
    param(
        [System.IO.FileStream]$LockStream,
        [string]$DestinationDir
    )

    $metadata = @(
        "target=" + [System.IO.Path]::GetFullPath($DestinationDir)
        "pid=" + $PID
        "machine=" + $env:COMPUTERNAME
        "acquired_utc=" + [DateTime]::UtcNow.ToString("o")
    ) -join [Environment]::NewLine

    $LockStream.SetLength(0)
    $LockStream.Position = 0
    $writer = New-Object System.IO.StreamWriter($LockStream, [System.Text.Encoding]::UTF8, 1024, $true)
    try {
        $writer.WriteLine($metadata)
        $writer.Flush()
        $LockStream.Flush()
    }
    finally {
        $writer.Dispose()
    }
}

function Invoke-WithDestinationFileLock {
    param(
        [string]$PackagesRoot,
        [string]$DestinationDir,
        [string]$Label,
        [scriptblock]$Body
    )

    $lockPath = Get-LockFilePath -PackagesRoot $PackagesRoot -DestinationDir $DestinationDir
    $lockStream = $null

    try {
        while ($null -eq $lockStream) {
            try {
                $lockStream = [System.IO.File]::Open(
                    $lockPath,
                    [System.IO.FileMode]::OpenOrCreate,
                    [System.IO.FileAccess]::ReadWrite,
                    [System.IO.FileShare]::None
                )
                Write-LockMetadata -LockStream $lockStream -DestinationDir $DestinationDir
            }
            catch [System.IO.IOException] {
                if ($null -ne $lockStream) {
                    $lockStream.Dispose()
                    $lockStream = $null
                }

                Write-Log "Waiting for $Label install lock at $DestinationDir ..."
                Start-Sleep -Milliseconds $LOCK_WAIT_LOG_INTERVAL_MS
            }
        }

        & $Body
    }
    finally {
        if ($null -ne $lockStream) {
            $lockStream.Dispose()
        }
    }
}

function Invoke-DownloadBytes {
    param(
        [string]$PackageName,
        [string]$DestinationLabel
    )

    $triesLeft = 4
    $delaySeconds = 2
    $sourceUrl = "https://bootstrap.packman.nvidia.com/$PackageName"

    while ($triesLeft -gt 0) {
        $triesLeft -= 1
        try {
            Write-Log "Fetching $PackageName for $DestinationLabel ..."
            $webClient = New-Object System.Net.WebClient
            try {
                return $webClient.DownloadData($sourceUrl)
            }
            finally {
                $webClient.Dispose()
            }
        }
        catch {
            Write-Log "Error downloading $sourceUrl"
            Write-Log $_.Exception.ToString()
            if ($triesLeft -le 0) {
                break
            }

            Write-Log "Retrying in $delaySeconds seconds ..."
            Start-Sleep -Seconds $delaySeconds
            $delaySeconds *= $delaySeconds
        }
    }

    throw "Failed to download $PackageName."
}

function Expand-ZipBytes {
    param(
        [byte[]]$ZipBytes,
        [string]$DestinationDir
    )

    $destinationRoot = [System.IO.Path]::GetFullPath($DestinationDir)
    $destinationRootWithSeparator = $destinationRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    $stream = New-Object System.IO.MemoryStream(,$ZipBytes)
    try {
        $archive = New-Object System.IO.Compression.ZipArchive(
            $stream,
            [System.IO.Compression.ZipArchiveMode]::Read,
            $false
        )
        try {
            foreach ($entry in $archive.Entries) {
                if ([string]::IsNullOrEmpty($entry.FullName)) {
                    continue
                }

                $entryPath = $entry.FullName.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
                $targetPath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($destinationRoot, $entryPath))
                if (($targetPath -ne $destinationRoot) -and (-not $targetPath.StartsWith($destinationRootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase))) {
                    throw "Archive entry '$($entry.FullName)' escapes destination '$destinationRoot'."
                }

                if ($entry.FullName.EndsWith("/")) {
                    Ensure-DirectoryExists -Path $targetPath
                    continue
                }

                $targetDir = Split-Path -Parent $targetPath
                if (-not [string]::IsNullOrEmpty($targetDir)) {
                    Ensure-DirectoryExists -Path $targetDir
                }

                $entryStream = $entry.Open()
                try {
                    $fileStream = [System.IO.File]::Open($targetPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
                    try {
                        $entryStream.CopyTo($fileStream)
                    }
                    finally {
                        $fileStream.Dispose()
                    }
                }
                finally {
                    $entryStream.Dispose()
                }
            }
        }
        finally {
            $archive.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Write-Sentinel {
    param(
        [string]$DestinationDir,
        [string]$PackageName
    )

    $sentinelPath = Get-SentinelPath -DestinationDir $DestinationDir
    $content = @(
        "package=$PackageName"
        "ready_utc=$([DateTime]::UtcNow.ToString("o"))"
    )
    [System.IO.File]::WriteAllLines($sentinelPath, $content)
}

function Ensure-ZipPackageInstall {
    param(
        [string]$PackagesRoot,
        [string]$DestinationDir,
        [string]$PackageName,
        [string]$Label
    )

    if (Test-Ready -DestinationDir $DestinationDir) {
        return
    }

    Invoke-WithDestinationFileLock -PackagesRoot $PackagesRoot -DestinationDir $DestinationDir -Label $Label -Body {
        if (Test-Ready -DestinationDir $DestinationDir) {
            return
        }

        Remove-DirectoryIfPresent -Path $DestinationDir
        Ensure-DirectoryExists -Path $DestinationDir

        $zipBytes = Invoke-DownloadBytes -PackageName $PackageName -DestinationLabel $Label
        Write-Log "Unpacking $Label ..."
        Expand-ZipBytes -ZipBytes $zipBytes -DestinationDir $DestinationDir
        Write-Sentinel -DestinationDir $DestinationDir -PackageName $PackageName
    }
}

function Write-Export {
    param(
        [string]$Name,
        [string]$Value
    )

    [Console]::Out.WriteLine($Name + "=" + $Value)
}

try {
    $pmPackagesRootWasProvided = -not [string]::IsNullOrEmpty($env:PM_PACKAGES_ROOT)
    if ($pmPackagesRootWasProvided) {
        $pmPackagesRoot = Resolve-NormalizedPath -Path $env:PM_PACKAGES_ROOT
    }
    else {
        $pmPackagesRoot = Resolve-NormalizedPath -Path (Get-DefaultPackagesRoot)
        Set-UserEnvironmentVariable -Name "PM_PACKAGES_ROOT" -Value $pmPackagesRoot
        Write-VisualStudioWarning
    }

    if (-not (Test-Path -LiteralPath $pmPackagesRoot)) {
        Write-Log "Creating packman packages cache at $pmPackagesRoot"
        Ensure-DirectoryExists -Path $pmPackagesRoot
    }

    if ($env:PM_PYTHON_EXT) {
        $pmPython = Resolve-NormalizedPath -Path $env:PM_PYTHON_EXT
    }
    else {
        $pmPythonBaseDir = Join-Path $pmPackagesRoot "python"
        $pmPythonDir = Join-Path $pmPythonBaseDir $PM_PYTHON_VERSION
        $pmPython = Resolve-NormalizedPath -Path (Join-Path $pmPythonDir "python.exe")
        Ensure-DirectoryExists -Path $pmPythonBaseDir
        Ensure-ZipPackageInstall -PackagesRoot $pmPackagesRoot -DestinationDir $pmPythonDir -PackageName ("python@" + $PM_PYTHON_VERSION + ".zip") -Label "Python interpreter"
    }

    if ($env:PM_MODULE_DIR_EXT) {
        $pmModuleDir = Resolve-NormalizedPath -Path $env:PM_MODULE_DIR_EXT
    }
    else {
        $pmModuleDir = Resolve-NormalizedPath -Path (Join-Path (Join-Path $pmPackagesRoot "packman-common") $PM_PACKMAN_VERSION)
        Ensure-ZipPackageInstall -PackagesRoot $pmPackagesRoot -DestinationDir $pmModuleDir -PackageName ("packman-common@" + $PM_PACKMAN_VERSION + ".zip") -Label "packman"
    }

    $pmModule = Resolve-NormalizedPath -Path (Join-Path $pmModuleDir "run.py")

    Write-Export -Name "PM_INSTALL_PATH" -Value $PM_INSTALL_PATH
    Write-Export -Name "PM_PACKAGES_ROOT" -Value $pmPackagesRoot
    Write-Export -Name "PM_PYTHON" -Value $pmPython
    Write-Export -Name "PM_MODULE_DIR" -Value $pmModuleDir
    Write-Export -Name "PM_MODULE" -Value $pmModule
    exit 0
}
catch {
    Write-Failure "!!! Failure while configuring local machine :( !!!"
    Write-Failure $_.Exception.Message
    exit 1
}

# SIG # Begin signature block
# MIIogAYJKoZIhvcNAQcCoIIocTCCKG0CAQExDzANBglghkgBZQMEAgEFADB5Bgor
# BgEEAYI3AgEEoGswaTA0BgorBgEEAYI3AgEeMCYCAwEAAAQQH8w7YFlLCE63JNLG
# KX7zUQIBAAIBAAIBAAIBAAIBADAxMA0GCWCGSAFlAwQCAQUABCCspjcJWcy0hS1C
# vQvipunEdK4JwP8LZ7usgUuWEB/4QKCCDbUwggawMIIEmKADAgECAhAIrUCyYNKc
# TJ9ezam9k67ZMA0GCSqGSIb3DQEBDAUAMGIxCzAJBgNVBAYTAlVTMRUwEwYDVQQK
# EwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5jb20xITAfBgNV
# BAMTGERpZ2lDZXJ0IFRydXN0ZWQgUm9vdCBHNDAeFw0yMTA0MjkwMDAwMDBaFw0z
# NjA0MjgyMzU5NTlaMGkxCzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwg
# SW5jLjFBMD8GA1UEAxM4RGlnaUNlcnQgVHJ1c3RlZCBHNCBDb2RlIFNpZ25pbmcg
# UlNBNDA5NiBTSEEzODQgMjAyMSBDQTEwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAw
# ggIKAoICAQDVtC9C0CiteLdd1TlZG7GIQvUzjOs9gZdwxbvEhSYwn6SOaNhc9es0
# JAfhS0/TeEP0F9ce2vnS1WcaUk8OoVf8iJnBkcyBAz5NcCRks43iCH00fUyAVxJr
# Q5qZ8sU7H/Lvy0daE6ZMswEgJfMQ04uy+wjwiuCdCcBlp/qYgEk1hz1RGeiQIXhF
# LqGfLOEYwhrMxe6TSXBCMo/7xuoc82VokaJNTIIRSFJo3hC9FFdd6BgTZcV/sk+F
# LEikVoQ11vkunKoAFdE3/hoGlMJ8yOobMubKwvSnowMOdKWvObarYBLj6Na59zHh
# 3K3kGKDYwSNHR7OhD26jq22YBoMbt2pnLdK9RBqSEIGPsDsJ18ebMlrC/2pgVItJ
# wZPt4bRc4G/rJvmM1bL5OBDm6s6R9b7T+2+TYTRcvJNFKIM2KmYoX7BzzosmJQay
# g9Rc9hUZTO1i4F4z8ujo7AqnsAMrkbI2eb73rQgedaZlzLvjSFDzd5Ea/ttQokbI
# YViY9XwCFjyDKK05huzUtw1T0PhH5nUwjewwk3YUpltLXXRhTT8SkXbev1jLchAp
# QfDVxW0mdmgRQRNYmtwmKwH0iU1Z23jPgUo+QEdfyYFQc4UQIyFZYIpkVMHMIRro
# OBl8ZhzNeDhFMJlP/2NPTLuqDQhTQXxYPUez+rbsjDIJAsxsPAxWEQIDAQABo4IB
# WTCCAVUwEgYDVR0TAQH/BAgwBgEB/wIBADAdBgNVHQ4EFgQUaDfg67Y7+F8Rhvv+
# YXsIiGX0TkIwHwYDVR0jBBgwFoAU7NfjgtJxXWRM3y5nP+e6mK4cD08wDgYDVR0P
# AQH/BAQDAgGGMBMGA1UdJQQMMAoGCCsGAQUFBwMDMHcGCCsGAQUFBwEBBGswaTAk
# BggrBgEFBQcwAYYYaHR0cDovL29jc3AuZGlnaWNlcnQuY29tMEEGCCsGAQUFBzAC
# hjVodHRwOi8vY2FjZXJ0cy5kaWdpY2VydC5jb20vRGlnaUNlcnRUcnVzdGVkUm9v
# dEc0LmNydDBDBgNVHR8EPDA6MDigNqA0hjJodHRwOi8vY3JsMy5kaWdpY2VydC5j
# b20vRGlnaUNlcnRUcnVzdGVkUm9vdEc0LmNybDAcBgNVHSAEFTATMAcGBWeBDAED
# MAgGBmeBDAEEATANBgkqhkiG9w0BAQwFAAOCAgEAOiNEPY0Idu6PvDqZ01bgAhql
# +Eg08yy25nRm95RysQDKr2wwJxMSnpBEn0v9nqN8JtU3vDpdSG2V1T9J9Ce7FoFF
# UP2cvbaF4HZ+N3HLIvdaqpDP9ZNq4+sg0dVQeYiaiorBtr2hSBh+3NiAGhEZGM1h
# mYFW9snjdufE5BtfQ/g+lP92OT2e1JnPSt0o618moZVYSNUa/tcnP/2Q0XaG3Ryw
# YFzzDaju4ImhvTnhOE7abrs2nfvlIVNaw8rpavGiPttDuDPITzgUkpn13c5Ubdld
# AhQfQDN8A+KVssIhdXNSy0bYxDQcoqVLjc1vdjcshT8azibpGL6QB7BDf5WIIIJw
# 8MzK7/0pNVwfiThV9zeKiwmhywvpMRr/LhlcOXHhvpynCgbWJme3kuZOX956rEnP
# LqR0kq3bPKSchh/jwVYbKyP/j7XqiHtwa+aguv06P0WmxOgWkVKLQcBIhEuWTatE
# QOON8BUozu3xGFYHKi8QxAwIZDwzj64ojDzLj4gLDb879M4ee47vtevLt/B3E+bn
# KD+sEq6lLyJsQfmCXBVmzGwOysWGw/YmMwwHS6DTBwJqakAwSEs0qFEgu60bhQji
# WQ1tygVQK+pKHJ6l/aCnHwZ05/LWUpD9r4VIIflXO7ScA+2GRfS0YW6/aOImYIbq
# yK+p/pQd52MbOoZWeE4wggb9MIIE5aADAgECAhAJrnXkyTwXU23cf6lU5rhZMA0G
# CSqGSIb3DQEBCwUAMGkxCzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwg
# SW5jLjFBMD8GA1UEAxM4RGlnaUNlcnQgVHJ1c3RlZCBHNCBDb2RlIFNpZ25pbmcg
# UlNBNDA5NiBTSEEzODQgMjAyMSBDQTEwHhcNMjUwNzMwMDAwMDAwWhcNMjgwNzI5
# MjM1OTU5WjCBhDELMAkGA1UEBhMCVVMxEzARBgNVBAgTCkNhbGlmb3JuaWExFDAS
# BgNVBAcTC1NhbnRhIENsYXJhMRswGQYDVQQKExJOVklESUEgQ29ycG9yYXRpb24x
# EDAOBgNVBAsTBzIwMDhCOUYxGzAZBgNVBAMTEk5WSURJQSBDb3Jwb3JhdGlvbjCC
# AaIwDQYJKoZIhvcNAQEBBQADggGPADCCAYoCggGBAKa2dMEsbSxk5eQlpZLnH3gh
# gkCcxlFDR/+as8oUqOztXFPrpGWMZkyOu06MNiPycUbVMh6pb8nJjr0ULpqPwd+7
# lPzZ1RBDiUizSvLW7QkcVrtcpaTzLV5N0aezqj1lXwFWU1MVGan0tXmcrSAaAFJG
# F6ChQbmh5ltEeJd+9ZQenKaH+eYZIPCS4Fk9sQefuuH060+oN4aME1iGyPL/l/cL
# wXnfGFkfy0TdmZDO+IHERhcjqrAFKfYgsDlDsUnOc6smhyRv1RFgsx2W73Mztt7U
# sZ81qORdZDTKztGykoQIC/YYo03iI7BfotTNY9c+81iN8qIsHBhdpRrerEKA/rm7
# dF8HrGjg7Nn+p75fRrIKZ7v86dOxPJm7s6HjDL/Ww37XwK+yK1XH+o+376bx0mFE
# OhmGczyn8YMUwz8frLHDb+Hi0Z0qLerYaU4Io1hxk6QCciCNToGwSzj+G+Cy1TH4
# DTtgl4A+GRDFMG9dY745HfRVlxdMlYpIMMfoO1kGIQIDAQABo4ICAzCCAf8wHwYD
# VR0jBBgwFoAUaDfg67Y7+F8Rhvv+YXsIiGX0TkIwHQYDVR0OBBYEFM8QW3t1WsAl
# sCyglf8KFZDQkFOgMD4GA1UdIAQ3MDUwMwYGZ4EMAQQBMCkwJwYIKwYBBQUHAgEW
# G2h0dHA6Ly93d3cuZGlnaWNlcnQuY29tL0NQUzAOBgNVHQ8BAf8EBAMCB4AwEwYD
# VR0lBAwwCgYIKwYBBQUHAwMwgbUGA1UdHwSBrTCBqjBToFGgT4ZNaHR0cDovL2Ny
# bDMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0VHJ1c3RlZEc0Q29kZVNpZ25pbmdSU0E0
# MDk2U0hBMzg0MjAyMUNBMS5jcmwwU6BRoE+GTWh0dHA6Ly9jcmw0LmRpZ2ljZXJ0
# LmNvbS9EaWdpQ2VydFRydXN0ZWRHNENvZGVTaWduaW5nUlNBNDA5NlNIQTM4NDIw
# MjFDQTEuY3JsMIGUBggrBgEFBQcBAQSBhzCBhDAkBggrBgEFBQcwAYYYaHR0cDov
# L29jc3AuZGlnaWNlcnQuY29tMFwGCCsGAQUFBzAChlBodHRwOi8vY2FjZXJ0cy5k
# aWdpY2VydC5jb20vRGlnaUNlcnRUcnVzdGVkRzRDb2RlU2lnbmluZ1JTQTQwOTZT
# SEEzODQyMDIxQ0ExLmNydDAJBgNVHRMEAjAAMA0GCSqGSIb3DQEBCwUAA4ICAQAb
# hocVBslPLkweNoXnzDyHjgUHVsdaBSxnKjHDTdOzXpo/a6VkK1VXK1fIhWYy4CcZ
# /wfyeb80+99KnfWWQzgL7nIElm4SkJRIMK8dODX3my4CQR6oSEsOimM1QUr8Gfio
# H98oQe8fhIQUOjnQsWiqbPzukx7ehCYnm2Xbu6nnSuvdzwFjiPykA91IFkVyP4Ex
# kf7JzWrFko4nceMXwtfGLB0jH1L+fmFUlAzXKNVIV/2GlXElSCMGHlUDy18D2hk+
# nij2DT1Gp+PmSSBCmQVIr+6HJMVXdz3jCtE3nQGKfAT+M8dvvEIq//0+cpkZdJxQ
# ctnHK5qukn8InimLh7fK8B+gu08wPHAhdEAc2eNM5Mmw67iCKN2/9hGvoVQlrtYm
# Ta1YglDOm4G0uSwtGIoa8O4S2uZES75HpAXRtc/hzSCOdeHR52wCMCx91OXrJ1kz
# NxIWIJZCE3NlzwKFObOZKONRiHPpC8oEToY/cxhTIn3Gg/70emY8JMvJoVfyokbu
# NfteFha3Z/5wIoxkcW971JaP2Z2TO1StjGcKEZvUF2OW7PQ7Toh55AowUx2UuSZQ
# MioaOVUrhqxcYWSBobA1Y3mwvuRfAgMqrfQ+eIwWYX+t7qg9fNjTPtxlQViE1H3+
# cRW9p8uV5rqJXhNzXmKYH8k4AZoUUOMnY//9ZE8TtjGCGiEwghodAgEBMH0waTEL
# MAkGA1UEBhMCVVMxFzAVBgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMUEwPwYDVQQDEzhE
# aWdpQ2VydCBUcnVzdGVkIEc0IENvZGUgU2lnbmluZyBSU0E0MDk2IFNIQTM4NCAy
# MDIxIENBMQIQCa515Mk8F1Nt3H+pVOa4WTANBglghkgBZQMEAgEFAKB8MBAGCisG
# AQQBgjcCAQwxAjAAMBkGCSqGSIb3DQEJAzEMBgorBgEEAYI3AgEEMBwGCisGAQQB
# gjcCAQsxDjAMBgorBgEEAYI3AgEVMC8GCSqGSIb3DQEJBDEiBCDq+IHl4M39mDu2
# 6avqildC6BcrE8zbxG/8k+j+uit/7TANBgkqhkiG9w0BAQEFAASCAYANHZPVDDKi
# 0+cKcfKhQEv95BdbPqYzWw43gV2eqFdb1C/WeEl90/7L+Ta1PcRj25FQByohFeL1
# ufQrOvwVDX3kWgKUs9lzpmcBELDCDYKwkmXzMAar6Av2pNuIfwYoKf4Ng07Boqal
# JprgOlfLIPbrvG9F3t+W53bV39Ojq181sP08p9niB7Et9amWvCGqE78snS7RdbgP
# 7tdQokl0iOGvoVXdTHbNloIn471niPMwW5ross9y9sCn2xUnH1ycMvr6pIYSw18F
# xGJs6Sg1sljYsMDT2lCKkCmyc/wE3b+/NbUSKHF+UzhcIiusb52q0mxyRyvKjWr5
# UtWVHatycmOBBzueBQRF8gASdPEW4/flGXycQYI+hvyTUOzoYnvPq0SYxmus8toP
# I/gCOCoEdaoR9+l9VnFKIC8p4mzQmhX8wzoUvkualbfd+K7TRceykZt82RdNlD2O
# vHqZAEDs4YN4w6RiYp4BWn7xf4fN4kKYOJW97TMBJcKf6LjZF7TO3NChghd3MIIX
# cwYKKwYBBAGCNwMDATGCF2MwghdfBgkqhkiG9w0BBwKgghdQMIIXTAIBAzEPMA0G
# CWCGSAFlAwQCAQUAMHgGCyqGSIb3DQEJEAEEoGkEZzBlAgEBBglghkgBhv1sBwEw
# MTANBglghkgBZQMEAgEFAAQgleACcOygsjV6YF0LRu8WPzHJGaSEBQH9pGZhtVIz
# LbwCEQCFDXzedyKjEJRYoLwUel14GA8yMDI2MDQxMjE0MTQxNVqgghM6MIIG7TCC
# BNWgAwIBAgIQCoDvGEuN8QWC0cR2p5V0aDANBgkqhkiG9w0BAQsFADBpMQswCQYD
# VQQGEwJVUzEXMBUGA1UEChMORGlnaUNlcnQsIEluYy4xQTA/BgNVBAMTOERpZ2lD
# ZXJ0IFRydXN0ZWQgRzQgVGltZVN0YW1waW5nIFJTQTQwOTYgU0hBMjU2IDIwMjUg
# Q0ExMB4XDTI1MDYwNDAwMDAwMFoXDTM2MDkwMzIzNTk1OVowYzELMAkGA1UEBhMC
# VVMxFzAVBgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMTswOQYDVQQDEzJEaWdpQ2VydCBT
# SEEyNTYgUlNBNDA5NiBUaW1lc3RhbXAgUmVzcG9uZGVyIDIwMjUgMTCCAiIwDQYJ
# KoZIhvcNAQEBBQADggIPADCCAgoCggIBANBGrC0Sxp7Q6q5gVrMrV7pvUf+GcAoB
# 38o3zBlCMGMyqJnfFNZx+wvA69HFTBdwbHwBSOeLpvPnZ8ZN+vo8dE2/pPvOx/Vj
# 8TchTySA2R4QKpVD7dvNZh6wW2R6kSu9RJt/4QhguSssp3qome7MrxVyfQO9sMx6
# ZAWjFDYOzDi8SOhPUWlLnh00Cll8pjrUcCV3K3E0zz09ldQ//nBZZREr4h/GI6Dx
# b2UoyrN0ijtUDVHRXdmncOOMA3CoB/iUSROUINDT98oksouTMYFOnHoRh6+86Ltc
# 5zjPKHW5KqCvpSduSwhwUmotuQhcg9tw2YD3w6ySSSu+3qU8DD+nigNJFmt6LAHv
# H3KSuNLoZLc1Hf2JNMVL4Q1OpbybpMe46YceNA0LfNsnqcnpJeItK/DhKbPxTTuG
# oX7wJNdoRORVbPR1VVnDuSeHVZlc4seAO+6d2sC26/PQPdP51ho1zBp+xUIZkpSF
# A8vWdoUoHLWnqWU3dCCyFG1roSrgHjSHlq8xymLnjCbSLZ49kPmk8iyyizNDIXj/
# /cOgrY7rlRyTlaCCfw7aSUROwnu7zER6EaJ+AliL7ojTdS5PWPsWeupWs7NpChUk
# 555K096V1hE0yZIXe+giAwW00aHzrDchIc2bQhpp0IoKRR7YufAkprxMiXAJQ1XC
# mnCfgPf8+3mnAgMBAAGjggGVMIIBkTAMBgNVHRMBAf8EAjAAMB0GA1UdDgQWBBTk
# O/zyMe39/dfzkXFjGVBDz2GM6DAfBgNVHSMEGDAWgBTvb1NK6eQGfHrK4pBW9i/U
# SezLTjAOBgNVHQ8BAf8EBAMCB4AwFgYDVR0lAQH/BAwwCgYIKwYBBQUHAwgwgZUG
# CCsGAQUFBwEBBIGIMIGFMCQGCCsGAQUFBzABhhhodHRwOi8vb2NzcC5kaWdpY2Vy
# dC5jb20wXQYIKwYBBQUHMAKGUWh0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0LmNvbS9E
# aWdpQ2VydFRydXN0ZWRHNFRpbWVTdGFtcGluZ1JTQTQwOTZTSEEyNTYyMDI1Q0Ex
# LmNydDBfBgNVHR8EWDBWMFSgUqBQhk5odHRwOi8vY3JsMy5kaWdpY2VydC5jb20v
# RGlnaUNlcnRUcnVzdGVkRzRUaW1lU3RhbXBpbmdSU0E0MDk2U0hBMjU2MjAyNUNB
# MS5jcmwwIAYDVR0gBBkwFzAIBgZngQwBBAIwCwYJYIZIAYb9bAcBMA0GCSqGSIb3
# DQEBCwUAA4ICAQBlKq3xHCcEua5gQezRCESeY0ByIfjk9iJP2zWLpQq1b4URGnwW
# BdEZD9gBq9fNaNmFj6Eh8/YmRDfxT7C0k8FUFqNh+tshgb4O6Lgjg8K8elC4+oWC
# qnU/ML9lFfim8/9yJmZSe2F8AQ/UdKFOtj7YMTmqPO9mzskgiC3QYIUP2S3HQvHG
# 1FDu+WUqW4daIqToXFE/JQ/EABgfZXLWU0ziTN6R3ygQBHMUBaB5bdrPbF6MRYs0
# 3h4obEMnxYOX8VBRKe1uNnzQVTeLni2nHkX/QqvXnNb+YkDFkxUGtMTaiLR9wjxU
# xu2hECZpqyU1d0IbX6Wq8/gVutDojBIFeRlqAcuEVT0cKsb+zJNEsuEB7O7/cuvT
# QasnM9AWcIQfVjnzrvwiCZ85EE8LUkqRhoS3Y50OHgaY7T/lwd6UArb+BOVAkg2o
# Ovol/DJgddJ35XTxfUlQ+8Hggt8l2Yv7roancJIFcbojBcxlRcGG0LIhp6GvReQG
# gMgYxQbV1S3CrWqZzBt1R9xJgKf47CdxVRd/ndUlQ05oxYy2zRWVFjF7mcr4C34M
# j3ocCVccAvlKV9jEnstrniLvUxxVZE/rptb7IRE2lskKPIJgbaP5t2nGj/ULLi49
# xTcBZU8atufk+EMF/cWuiC7POGT75qaL6vdCvHlshtjdNXOCIUjsarfNZzCCBrQw
# ggScoAMCAQICEA3HrFcF/yGZLkBDIgw6SYYwDQYJKoZIhvcNAQELBQAwYjELMAkG
# A1UEBhMCVVMxFTATBgNVBAoTDERpZ2lDZXJ0IEluYzEZMBcGA1UECxMQd3d3LmRp
# Z2ljZXJ0LmNvbTEhMB8GA1UEAxMYRGlnaUNlcnQgVHJ1c3RlZCBSb290IEc0MB4X
# DTI1MDUwNzAwMDAwMFoXDTM4MDExNDIzNTk1OVowaTELMAkGA1UEBhMCVVMxFzAV
# BgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMUEwPwYDVQQDEzhEaWdpQ2VydCBUcnVzdGVk
# IEc0IFRpbWVTdGFtcGluZyBSU0E0MDk2IFNIQTI1NiAyMDI1IENBMTCCAiIwDQYJ
# KoZIhvcNAQEBBQADggIPADCCAgoCggIBALR4MdMKmEFyvjxGwBysddujRmh0tFEX
# nU2tjQ2UtZmWgyxU7UNqEY81FzJsQqr5G7A6c+Gh/qm8Xi4aPCOo2N8S9SLrC6Kb
# ltqn7SWCWgzbNfiR+2fkHUiljNOqnIVD/gG3SYDEAd4dg2dDGpeZGKe+42DFUF0m
# R/vtLa4+gKPsYfwEu7EEbkC9+0F2w4QJLVSTEG8yAR2CQWIM1iI5PHg62IVwxKSp
# O0XaF9DPfNBKS7Zazch8NF5vp7eaZ2CVNxpqumzTCNSOxm+SAWSuIr21Qomb+zzQ
# WKhxKTVVgtmUPAW35xUUFREmDrMxSNlr/NsJyUXzdtFUUt4aS4CEeIY8y9IaaGBp
# PNXKFifinT7zL2gdFpBP9qh8SdLnEut/GcalNeJQ55IuwnKCgs+nrpuQNfVmUB5K
# lCX3ZA4x5HHKS+rqBvKWxdCyQEEGcbLe1b8Aw4wJkhU1JrPsFfxW1gaou30yZ46t
# 4Y9F20HHfIY4/6vHespYMQmUiote8ladjS/nJ0+k6MvqzfpzPDOy5y6gqztiT96F
# v/9bH7mQyogxG9QEPHrPV6/7umw052AkyiLA6tQbZl1KhBtTasySkuJDpsZGKdls
# jg4u70EwgWbVRSX1Wd4+zoFpp4Ra+MlKM2baoD6x0VR4RjSpWM8o5a6D8bpfm4CL
# KczsG7ZrIGNTAgMBAAGjggFdMIIBWTASBgNVHRMBAf8ECDAGAQH/AgEAMB0GA1Ud
# DgQWBBTvb1NK6eQGfHrK4pBW9i/USezLTjAfBgNVHSMEGDAWgBTs1+OC0nFdZEzf
# Lmc/57qYrhwPTzAOBgNVHQ8BAf8EBAMCAYYwEwYDVR0lBAwwCgYIKwYBBQUHAwgw
# dwYIKwYBBQUHAQEEazBpMCQGCCsGAQUFBzABhhhodHRwOi8vb2NzcC5kaWdpY2Vy
# dC5jb20wQQYIKwYBBQUHMAKGNWh0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0LmNvbS9E
# aWdpQ2VydFRydXN0ZWRSb290RzQuY3J0MEMGA1UdHwQ8MDowOKA2oDSGMmh0dHA6
# Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydFRydXN0ZWRSb290RzQuY3JsMCAG
# A1UdIAQZMBcwCAYGZ4EMAQQCMAsGCWCGSAGG/WwHATANBgkqhkiG9w0BAQsFAAOC
# AgEAF877FoAc/gc9EXZxML2+C8i1NKZ/zdCHxYgaMH9Pw5tcBnPw6O6FTGNpoV2V
# 4wzSUGvI9NAzaoQk97frPBtIj+ZLzdp+yXdhOP4hCFATuNT+ReOPK0mCefSG+tXq
# GpYZ3essBS3q8nL2UwM+NMvEuBd/2vmdYxDCvwzJv2sRUoKEfJ+nN57mQfQXwcAE
# GCvRR2qKtntujB71WPYAgwPyWLKu6RnaID/B0ba2H3LUiwDRAXx1Neq9ydOal95C
# HfmTnM4I+ZI2rVQfjXQA1WSjjf4J2a7jLzWGNqNX+DF0SQzHU0pTi4dBwp9nEC8E
# AqoxW6q17r0z0noDjs6+BFo+z7bKSBwZXTRNivYuve3L2oiKNqetRHdqfMTCW/Nm
# KLJ9M+MtucVGyOxiDf06VXxyKkOirv6o02OoXN4bFzK0vlNMsvhlqgF2puE6Fndl
# ENSmE+9JGYxOGLS/D284NHNboDGcmWXfwXRy4kbu4QFhOm0xJuF2EZAOk5eCkhSx
# ZON3rGlHqhpB/8MluDezooIs8CVnrpHMiD2wL40mm53+/j7tFaxYKIqL0Q4ssd8x
# HZnIn/7GELH3IdvG2XlM9q7WP/UwgOkw/HQtyRN62JK4S1C8uw3PdBunvAZapsiI
# 5YKdvlarEvf8EA+8hcpSM9LHJmyrxaFtoza2zNaQ9k+5t1wwggWNMIIEdaADAgEC
# AhAOmxiO+dAt5+/bUOIIQBhaMA0GCSqGSIb3DQEBDAUAMGUxCzAJBgNVBAYTAlVT
# MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j
# b20xJDAiBgNVBAMTG0RpZ2lDZXJ0IEFzc3VyZWQgSUQgUm9vdCBDQTAeFw0yMjA4
# MDEwMDAwMDBaFw0zMTExMDkyMzU5NTlaMGIxCzAJBgNVBAYTAlVTMRUwEwYDVQQK
# EwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5jb20xITAfBgNV
# BAMTGERpZ2lDZXJ0IFRydXN0ZWQgUm9vdCBHNDCCAiIwDQYJKoZIhvcNAQEBBQAD
# ggIPADCCAgoCggIBAL/mkHNo3rvkXUo8MCIwaTPswqclLskhPfKK2FnC4SmnPVir
# dprNrnsbhA3EMB/zG6Q4FutWxpdtHauyefLKEdLkX9YFPFIPUh/GnhWlfr6fqVcW
# WVVyr2iTcMKyunWZanMylNEQRBAu34LzB4TmdDttceItDBvuINXJIB1jKS3O7F5O
# yJP4IWGbNOsFxl7sWxq868nPzaw0QF+xembud8hIqGZXV59UWI4MK7dPpzDZVu7K
# e13jrclPXuU15zHL2pNe3I6PgNq2kZhAkHnDeMe2scS1ahg4AxCN2NQ3pC4FfYj1
# gj4QkXCrVYJBMtfbBHMqbpEBfCFM1LyuGwN1XXhm2ToxRJozQL8I11pJpMLmqaBn
# 3aQnvKFPObURWBf3JFxGj2T3wWmIdph2PVldQnaHiZdpekjw4KISG2aadMreSx7n
# DmOu5tTvkpI6nj3cAORFJYm2mkQZK37AlLTSYW3rM9nF30sEAMx9HJXDj/chsrIR
# t7t/8tWMcCxBYKqxYxhElRp2Yn72gLD76GSmM9GJB+G9t+ZDpBi4pncB4Q+UDCEd
# slQpJYls5Q5SUUd0viastkF13nqsX40/ybzTQRESW+UQUOsxxcpyFiIJ33xMdT9j
# 7CFfxCBRa2+xq4aLT8LWRV+dIPyhHsXAj6KxfgommfXkaS+YHS312amyHeUbAgMB
# AAGjggE6MIIBNjAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQWBBTs1+OC0nFdZEzf
# Lmc/57qYrhwPTzAfBgNVHSMEGDAWgBRF66Kv9JLLgjEtUYunpyGd823IDzAOBgNV
# HQ8BAf8EBAMCAYYweQYIKwYBBQUHAQEEbTBrMCQGCCsGAQUFBzABhhhodHRwOi8v
# b2NzcC5kaWdpY2VydC5jb20wQwYIKwYBBQUHMAKGN2h0dHA6Ly9jYWNlcnRzLmRp
# Z2ljZXJ0LmNvbS9EaWdpQ2VydEFzc3VyZWRJRFJvb3RDQS5jcnQwRQYDVR0fBD4w
# PDA6oDigNoY0aHR0cDovL2NybDMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0QXNzdXJl
# ZElEUm9vdENBLmNybDARBgNVHSAECjAIMAYGBFUdIAAwDQYJKoZIhvcNAQEMBQAD
# ggEBAHCgv0NcVec4X6CjdBs9thbX979XB72arKGHLOyFXqkauyL4hxppVCLtpIh3
# bb0aFPQTSnovLbc47/T/gLn4offyct4kvFIDyE7QKt76LVbP+fT3rDB6mouyXtTP
# 0UNEm0Mh65ZyoUi0mcudT6cGAxN3J0TU53/oWajwvy8LpunyNDzs9wPHh6jSTEAZ
# NUZqaVSwuKFWjuyk1T3osdz9HNj0d1pcVIxv76FQPfx2CWiEn2/K2yCNNWAcAgPL
# ILCsWKAOQGPFmCLBsln1VWvPJ6tsds5vIy30fnFqI2si/xK4VC0nftg62fC2h5b9
# W9FcrBjDTZ9ztwGpn1eqXijiuZQxggN8MIIDeAIBATB9MGkxCzAJBgNVBAYTAlVT
# MRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8GA1UEAxM4RGlnaUNlcnQgVHJ1
# c3RlZCBHNCBUaW1lU3RhbXBpbmcgUlNBNDA5NiBTSEEyNTYgMjAyNSBDQTECEAqA
# 7xhLjfEFgtHEdqeVdGgwDQYJYIZIAWUDBAIBBQCggdEwGgYJKoZIhvcNAQkDMQ0G
# CyqGSIb3DQEJEAEEMBwGCSqGSIb3DQEJBTEPFw0yNjA0MTIxNDE0MTVaMCsGCyqG
# SIb3DQEJEAIMMRwwGjAYMBYEFN1iMKyGCi0wa9o4sWh5UjAH+0F+MC8GCSqGSIb3
# DQEJBDEiBCBEDE1bM0JeVsgkt8e0EpioS1nmYUH5bHns2K30sk7s/DA3BgsqhkiG
# 9w0BCRACLzEoMCYwJDAiBCBKoD+iLNdchMVck4+CjmdrnK7Ksz/jbSaaozTxRhEK
# MzANBgkqhkiG9w0BAQEFAASCAgAtk4IxB00uVktLWdB9jlGBfwLT7wjozVTIqeZc
# VzghOXe/9DiyFY9FzPS9pL44G0oJ7fyhVP+3luevuCecL6PzCfk9xdeoXP1hkwhn
# 5EuySAA8DYtr8zC29LwzeAu6yBgaBtAhu29IGaHtxAIsFSJSVvW6EpdRMuoHq6yC
# 5FSe+p9A77cDTg/U6g9bL4dpgLuiv0r7Z4YnAr7d3c2m/Tkl6XSOfiLxMirUcM8b
# SD92R7GPgzUKe36+Gn8+Y2Ysdk9MyIm2nbFSW5w83uBaCw/IHnwRu5V0bsa5kuop
# d05LzY2aaEX/YnucA8d1ZwmsJmMhfxa81SRoswFcJs2H26zixIlb3V9Ynvka+98X
# MPjTrhXNy0Hozj2ythROkzgWFe2MKDxJc+kSdA4LzwrnsyKDHPQ/5rjZLYXAul/+
# /nTZ6e+rlqLBLXMhQzm1ZaRemA4gkvWWYw6+To6Zf7AsPoHCPo/rjXNNb5Cge2WB
# qQPGQ7yCa6Qk/nISiT2XlAwmCmSBA0tRsQwvKBXc7ooET13hq8acIZdc+w5e4OOZ
# 1cpR+6pL7Hec3jxcYnIjDQ0HJCfvLv/6scyAJaTW9GqR+4S0UDDj2eLylqx4Q8Q+
# gWVr2iingfXdUwlXnfvffz34IzEO6TLrjHQDHcdXYhGW12JdVHnVxNwTsu33+c88
# 1u8V5w==
# SIG # End signature block
