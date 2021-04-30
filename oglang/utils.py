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

        

