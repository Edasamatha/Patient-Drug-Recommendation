from pathlib import Path
import runpy


# Streamlit reruns this file often; execute the real app script each rerun.
runpy.run_path(str(Path(__file__).resolve().parent / "streamlit_app" / "app.py"), run_name="__main__")
