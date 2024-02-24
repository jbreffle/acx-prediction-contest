"""Profile the streamlit app to find bottlenecks."""

# Imports
import os
import sys
from datetime import datetime
import cProfile
import pstats
import pyprojroot
import importlib

import Home

streamlit_pages_dir = pyprojroot.here("streamlit/pages")
streamlit_logs_dir = pyprojroot.here("streamlit/logs")
date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def profile_all_pages():
    for file in streamlit_pages_dir.iterdir():
        if file.is_file() and file.name.endswith(".py") and file.name != "__init__.py":
            module_name = file.stem  # This will get the file name without '.py'
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            profiler = cProfile.Profile()
            profiler.enable()
            if hasattr(module, "main"):
                module.main()
            else:
                print(f"No main() found in {module_name}")

            profiler.disable()
            # Ensure the logs directory exists
            streamlit_logs_dir.mkdir(parents=True, exist_ok=True)
            output_file = (
                streamlit_logs_dir / f"profile_{date_time_str}_{module_name}.txt"
            )
            with open(output_file, "w", encoding="utf-8") as stream:
                stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
                stats.print_stats()


def profile_home_page():
    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    Home.main()
    profiler.disable()
    # Writing the stats to a file, with datetime in the filename
    output_file = streamlit_logs_dir / f"profile_{date_time_str}_0_Home.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as stream:
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats()


if __name__ == "__main__":
    profile_home_page()
    profile_all_pages()
