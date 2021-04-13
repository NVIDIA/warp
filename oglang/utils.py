import os
import timeit
import cProfile

# timer utils

class ScopedTimer:

    indent = -1

    enabled = True

    def __init__(self, name, active=True, detailed=False, dict=None):
        self.name = name
        self.active = active and self.enabled
        self.detailed = detailed
        self.dict = dict

        if self.dict is not None:
            if name not in self.dict:
                self.dict[name] = []

    def __enter__(self):

        if (self.active):
            self.start = timeit.default_timer()
            ScopedTimer.indent += 1

            if (self.detailed):
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()


    def __exit__(self, exc_type, exc_value, traceback):

        if (self.detailed):
            self.cp.disable()
            self.cp.print_stats(sort='tottime')

        if (self.active):
            self.elapsed = (timeit.default_timer() - self.start) * 1000.0

            if self.dict is not None:
                self.dict[self.name].append(self.elapsed)

            indent = ""
            for i in range(ScopedTimer.indent):
                indent += "\t"

            print("{}{} took {:.2f} ms".format(indent, self.name, self.elapsed))

            ScopedTimer.indent -= 1

        


# runs vcvars and copies back the build environment
def set_build_env():

    def find_vcvars_path():
        import glob
        for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
            paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Auxiliary\Build\vcvars64.bat" % edition), reverse=True)
            if paths:
                return paths[0]

    if os.name == 'nt':

        vcvars_path = find_vcvars_path()

        # merge vcvars with our env
        s = '"{}" && set'.format(vcvars_path)
        output = os.popen(s).read()
        for line in output.splitlines():
            pair = line.split("=", 1)
            if (len(pair) >= 2):
                os.environ[pair[0]] = pair[1]



# See PyTorch for reference on how to find nvcc.exe more robustly, https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension
def find_cuda():
    
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return cuda_home
    
