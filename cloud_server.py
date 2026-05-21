from __future__ import annotations

import os
import shutil
from pathlib import Path

from ia_apprend.web_app import run_web_server


APP_ROOT = Path(__file__).resolve().parent
DATA_FILES = (
    "qa_seed.json",
    "qa_model.json",
    "science.json",
    "celine dion.json",
    "histoire.json",
    "histoire deux.json",
    "PHYTON.json",
    "WEB.json",
    "mega_qa_pack.json",
    "lucie_extra_questions.json",
    "lucie_extra_questions_2.json",
    "label_studio_training.json",
    "label_studio_ner_clean.json",
    "label_studio_relations_clean.json",
)


def prepare_cloud_data() -> Path:
    data_root = Path(os.getenv("LUCIE_DATA_DIR", str(APP_ROOT))).resolve()
    memory_dir = data_root / "ia_apprend"
    memory_dir.mkdir(parents=True, exist_ok=True)

    bundled_memory = APP_ROOT / "ia_apprend" / "memory.json"
    target_memory = memory_dir / "memory.json"
    if bundled_memory.exists() and not target_memory.exists():
        shutil.copy2(bundled_memory, target_memory)

    for name in DATA_FILES:
        src = APP_ROOT / name
        dst = data_root / name
        if src.exists():
            shutil.copy2(src, dst)

    return target_memory


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    memory_path = prepare_cloud_data()
    run_web_server(host=host, port=port, memory_path=memory_path)


if __name__ == "__main__":
    main()
