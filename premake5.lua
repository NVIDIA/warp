-- Shared build scripts from repo_build package
repo_build = require("omni/repo/build")

-- Repo root
root = repo_build.get_abs_path(".")

-- Insert kit template premake configuration, it creates solution, finds extensions.. Look inside for more details.
dofile("_repo/deps/repo_kit_tools/kit-template/premake5.lua")

-- Extra folder linking and file copy setup:
repo_build.prebuild_link {
    -- Link python app sources in target dir for easier edit
    { "source/pythonapps/target", bin_dir.."/pythonapps" },
}
repo_build.prebuild_copy {
    -- Copy python app running scripts in target dir
    {"source/pythonapps/runscripts/$config/*$shell_ext", bin_dir}
}

-- Application example. Only runs Kit with a config, doesn't build anything. Helper for debugging.
define_app("omni.app.new_exts_demo_mini")
define_app("omni.app.precache_exts_demo")
