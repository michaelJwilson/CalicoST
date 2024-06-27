import io
import cProfile
import pstats

class ProfileContext:
    def __init__(self, line_threshold=10):
        self.threshold = line_threshold

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        return self

    def __exit__(self, *args):
        self.profiler.disable()

        if True:
            ss = io.StringIO()

            ps = pstats.Stats(self.profiler, stream=ss).sort_stats("cumulative")
            ps.print_stats(self.threshold)

            profile = ss.getvalue()

            # Only print if there's something to print
            if profile.strip():
                print(profile)
