import jetbrains.buildServer.configs.kotlin.v2019_2.*
import jetbrains.buildServer.configs.kotlin.v2019_2.buildFeatures.freeDiskSpace
import jetbrains.buildServer.configs.kotlin.v2019_2.buildSteps.script
import jetbrains.buildServer.configs.kotlin.v2019_2.triggers.finishBuildTrigger
import jetbrains.buildServer.configs.kotlin.v2019_2.triggers.vcs
import jetbrains.buildServer.configs.kotlin.v2019_2.vcs.GitVcsRoot

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2021.1"

project {

    vcsRoot(GitlabMasterWarp)

    params {
        param("env.OV_BRANCH_NAME", "no need")
    }

    subProject(Master)
}

object GitlabMasterWarp : GitVcsRoot({
    name = "gitlab-master-warp"
    url = "ssh://git@gitlab-master.nvidia.com:12051/omniverse/warp.git"
    branch = "refs/heads/tc"
    branchSpec = """
        +:refs/heads/WARForTeamCityNotFillingParamsUnlessThereIsABranchSpecDefined
        +:refs/(merge-requests*)/head
        +:refs/heads/(*)
    """.trimIndent()
    checkoutPolicy = GitVcsRoot.AgentCheckoutPolicy.USE_MIRRORS
    authMethod = uploadedKey {
        userName = "git"
        uploadedKey = "TC Omniverse RW key"
    }
    param("secure:password", "")
})


object Master : Project({
    name = "master"

    buildType(Master_PublishExtensions)
    buildType(Master_BuildAndValidation)

    params {
        param("omni.branchname", "master")
    }
    subProjectsOrder = arrayListOf(RelativeId("Master_Building"), RelativeId("Master_Testing"), RelativeId("Master_Publishing"))

    subProject(Master_Testing)
    subProject(Master_Building)
    subProject(Master_Publishing)
})

object Master_BuildAndValidation : BuildType({
    name = "build and validation"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(GitlabMasterWarp)

        showDependenciesChanges = true
    }

    triggers {
        vcs {
            branchFilter = """
                +:<default>
                +:merge-requests*
            """.trimIndent()
        }
    }

    dependencies {
        snapshot(Master_Building_BuildLinuxX8664) {
            onDependencyFailure = FailureAction.CANCEL
        }
        snapshot(Master_Building_BuildWindowsX8664) {
            onDependencyFailure = FailureAction.CANCEL
        }
        snapshot(Master_Building_GenerateBuildNumber) {
            onDependencyFailure = FailureAction.CANCEL
        }
        snapshot(Master_Testing_TestLinuxX8664) {
        }
        snapshot(Master_Testing_TestWindowsX8664) {
        }
    }
})

object Master_PublishExtensions : BuildType({
    name = "publish extensions"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        showDependenciesChanges = true
    }

    triggers {
        finishBuildTrigger {
            buildType = "${Master_BuildAndValidation.id}"
            successfulOnly = true
        }
    }

    dependencies {
        snapshot(Master_Publishing_PublishExtensionsLinuxX8664) {
            onDependencyFailure = FailureAction.CANCEL
            onDependencyCancel = FailureAction.CANCEL
        }
        snapshot(Master_Publishing_PublishExtensionsWindowsX8664) {
            onDependencyFailure = FailureAction.CANCEL
            onDependencyCancel = FailureAction.CANCEL
        }
    }
})


object Master_Building : Project({
    name = "building"

    buildType(Master_Building_BuildLinuxX8664)
    buildType(Master_Building_BuildLinuxAarch64)
    buildType(Master_Building_GenerateBuildNumber)
    buildType(Master_Building_BuildWindowsX8664)
})

object Master_Building_BuildLinuxAarch64 : BuildType({
    name = "build (linux-aarch64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    params {
        param("env.BUILD_NUMBER", "%build.number%")
    }

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
    }

    steps {
        script {
            scriptContent = "./tools/ci/building/build-linux-x86_64/step.sh"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    failureConditions {
        executionTimeoutMin = 30
    }

    dependencies {
        snapshot(Master_Building_GenerateBuildNumber) {
            onDependencyFailure = FailureAction.CANCEL
            onDependencyCancel = FailureAction.CANCEL
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Linux")
        doesNotExist("system.feature.nvidia.gpu.name")
    }
})

object Master_Building_BuildLinuxX8664 : BuildType({
    name = "build (linux-x86_64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    params {
        param("env.BUILD_NUMBER", "%build.number%")
    }

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
    }

    steps {
        script {
            scriptContent = "./tools/ci/building/build-linux-x86_64/step.sh"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    failureConditions {
        executionTimeoutMin = 30
    }

    dependencies {
        snapshot(Master_Building_GenerateBuildNumber) {
            onDependencyFailure = FailureAction.CANCEL
            onDependencyCancel = FailureAction.CANCEL
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Linux")
        doesNotExist("system.feature.nvidia.gpu.name")
    }
})

object Master_Building_BuildWindowsX8664 : BuildType({
    name = "build (windows-x86_64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    params {
        param("env.BUILD_NUMBER", "%build.number%")
    }

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
    }

    steps {
        script {
            scriptContent = "tools/ci/building/build-windows-x86_64/step.bat"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    failureConditions {
        executionTimeoutMin = 30
    }

    dependencies {
        snapshot(Master_Building_GenerateBuildNumber) {
            onDependencyFailure = FailureAction.FAIL_TO_START
            onDependencyCancel = FailureAction.FAIL_TO_START
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows")
        doesNotExist("system.feature.nvidia.gpu.name")
    }
})

object Master_Building_GenerateBuildNumber : BuildType({
    name = "generate build number"

    params {
        param("env.OV_BRANCH_NAME", "master")
    }

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
    }

    steps {
        script {
            scriptContent = """call tools\ci\building\generate-build-number\step.bat"""
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows")
        doesNotExist("system.feature.nvidia.gpu.count")
    }
})


object Master_Publishing : Project({
    name = "publishing"

    buildType(Master_Publishing_PublishExtensionsLinuxX8664)
    buildType(Master_Publish)
    buildType(Master_Publishing_PublishExtensionsWindowsX8664)
    buildType(Master_Publishing_PublishDocs)
})

object Master_Publish : BuildType({
    name = "publish"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    vcs {
        root(GitlabMasterWarp)
    }

    steps {
        script {
            scriptContent = "tools/ci/publish/step.bat"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    dependencies {
        dependency(Master_Building_BuildLinuxX8664) {
            snapshot {
                onDependencyFailure = FailureAction.CANCEL
                onDependencyCancel = FailureAction.CANCEL
            }

            artifacts {
                artifactRules = """
                    *.7z=>warp
                """.trimIndent()
            }
        }
        dependency(Master_Building_BuildWindowsX8664) {
            snapshot {
                onDependencyFailure = FailureAction.CANCEL
                onDependencyCancel = FailureAction.CANCEL
            }

            artifacts {
                artifactRules = """
                    *.7z=>warp
                """.trimIndent()
            }
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows")
    }
})

object Master_Publishing_PublishExtensionsLinuxX8664 : BuildType({
    name = "publish extensions (linux-x86_64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
        showDependenciesChanges = true
    }

    steps {
        script {
            scriptContent = "tools/ci/publishing/publish-extensions-linux-x86_64/step.sh"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    features {
        freeDiskSpace {
            failBuild = false
        }
    }

    dependencies {
        dependency(Master_Building_BuildLinuxX8664) {
            snapshot {
                onDependencyFailure = FailureAction.CANCEL
                onDependencyCancel = FailureAction.CANCEL
            }

            artifacts {
                artifactRules = """
                    *.7z => _build/packages
                """.trimIndent()
            }
        }
    }

    requirements {
        doesNotExist("system.feature.nvidia.gpu.driver.major")
        contains("teamcity.agent.jvm.os.name", "Linux")
    }
})

object Master_Publishing_PublishExtensionsWindowsX8664 : BuildType({
    name = "publish extensions (windows-x86_64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
        showDependenciesChanges = true
    }

    steps {
        script {
            scriptContent = """tools\ci\publishing\publish-extensions-windows-x86_64\step.bat"""
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    features {
        freeDiskSpace {
            failBuild = false
        }
    }

    dependencies {
        dependency(Master_Building_BuildWindowsX8664) {
            snapshot {
                onDependencyFailure = FailureAction.CANCEL
                onDependencyCancel = FailureAction.CANCEL
            }

            artifacts {
                cleanDestination = true
                artifactRules = """
                    *.7z => _build/packages
                """.trimIndent()
            }
        }
    }

    requirements {
        exists("system.feature.windows.version")
        doesNotExist("system.feature.nvidia.gpu.driver.major")
    }
})

object Master_Publishing_PublishDocs : BuildType({
    name = "publish docs"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
        showDependenciesChanges = true
    }

    triggers {
        finishBuildTrigger {
            buildType = "${Master_BuildAndValidation.id}"
            successfulOnly = true
        }
    }

    steps {
        script {
            scriptContent = """tools\ci\publishing\publish-docs\step.bat"""
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    features {
        freeDiskSpace {
            failBuild = false
        }
    }

    dependencies {
        dependency(Master_Building_BuildWindowsX8664) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
                onDependencyCancel = FailureAction.FAIL_TO_START
            }

            artifacts {
                artifactRules = """
                    docs*.7z!** => .
                """.trimIndent()
            }
        }
    }

    requirements {
        doesNotExist("system.feature.nvidia.gpu.driver.major")
        exists("system.feature.windows.version")
    }

    params {
        param("env.AWS_ACCESS_KEY_ID", "%omniverse-docs.AWS_ACCESS_KEY_ID%")
        param("env.AWS_SECRET_ACCESS_KEY", "%omniverse-docs.AWS_SECRET_ACCESS_KEY%")
    }
})

object Master_Testing : Project({
    name = "testing"

    buildType(Master_Testing_TestLinuxX8664)
    buildType(Master_Testing_TestWindowsX8664)
})

object Master_Testing_TestLinuxX8664 : BuildType({
    name = "test (linux-x86_64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
    }

    steps {
        script {
            scriptContent = "./tools/ci/testing/test-linux-x86_64/step.sh"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    dependencies {
        dependency(Master_Building_BuildLinuxX8664) {
            snapshot {
                onDependencyFailure = FailureAction.CANCEL
                onDependencyCancel = FailureAction.CANCEL
            }

            artifacts {
                cleanDestination = true
                artifactRules = "*.7z => _build/packages"
            }
        }
    }

    requirements {
        exists("system.feature.nvidia.gpu.driver.major")
        contains("teamcity.agent.jvm.os.name", "Linux")
    }
})

object Master_Testing_TestWindowsX8664 : BuildType({
    name = "test (windows-x86_64)"

    buildNumberPattern = "${Master_Building_GenerateBuildNumber.depParamRefs.buildNumber}"

    vcs {
        root(GitlabMasterWarp)

        cleanCheckout = true
    }

    steps {
        script {
            scriptContent = """call tools\ci\testing\test-windows-x86_64\step.bat"""
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    dependencies {
        dependency(Master_Building_BuildWindowsX8664) {
            snapshot {
                onDependencyFailure = FailureAction.CANCEL
                onDependencyCancel = FailureAction.CANCEL
            }

            artifacts {
                cleanDestination = true
                artifactRules = "*.7z => _build/packages"
            }
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows")
        exists("system.feature.nvidia.gpu.driver.major")
    }
})
