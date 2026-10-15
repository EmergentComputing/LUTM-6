"""Execute human workflows without saving generated outputs into the notebooks."""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def validate():
    root = Path(__file__).resolve().parents[1]
    for name in ("01_simulate.ipynb", "02_train_eval.ipynb"):
        notebook = nbformat.read(root / name, as_version=4)
        nbformat.validate(notebook)
        # Test this exact conda interpreter, without installing a kernelspec.
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        try:
            NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f"{name}: all {sum(c.cell_type == 'code' for c in notebook.cells)} code cells executed", flush=True)


if __name__ == "__main__":
    validate()
