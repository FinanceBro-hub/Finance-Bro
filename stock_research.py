# -*- coding: utf-8 -*-
"""
Finance Bro — update the Stock Research price-chart selector.

Place this file in the same folder as stock_research.py and run it once:

    python update_stock_research_all_tickers.py

What it changes:
- Removes the previous five-stock default limit.
- Makes every valid ticker selected initially in "Stocks in Price Chart".
- Keeps the multiselect editable, so the user can remove individual tickers.
- Creates a backup before changing the original file.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


TARGET_FILENAME = "stock_research.py"
BACKUP_FILENAME = "stock_research_before_all_tickers.py"


def main() -> None:
    project_folder = Path(__file__).resolve().parent
    target_path = project_folder / TARGET_FILENAME
    backup_path = project_folder / BACKUP_FILENAME

    if not target_path.exists():
        raise FileNotFoundError(
            f"{TARGET_FILENAME} was not found in:\n{project_folder}\n\n"
            "Move this update file into the same folder as stock_research.py "
            "and run it again."
        )

    source = target_path.read_text(encoding="utf-8")

    if '"Stocks in Price Chart"' not in source:
        raise RuntimeError(
            'The selector "Stocks in Price Chart" was not found. '
            "No changes were made."
        )

    already_updated_pattern = re.compile(
        r'"Stocks in Price Chart"\s*,'
        r'(?:(?!st\.multiselect).)*?'
        r'default\s*=\s*valid_tickers\s*,',
        flags=re.DOTALL,
    )

    if already_updated_pattern.search(source):
        print(
            "No change was necessary: all valid tickers are already selected "
            "by default."
        )
        return

    # Remove the old helper variable that limited the initial selection
    # to the first five valid tickers.
    limit_pattern = re.compile(
        r'\n(?P<indent>[ \t]+)default_chart_tickers\s*=\s*valid_tickers\[\s*'
        r':\s*min\(\s*5\s*,\s*len\(\s*valid_tickers\s*\)\s*\)\s*'
        r'\]\s*\n',
        flags=re.MULTILINE,
    )

    updated_source, removed_count = limit_pattern.subn("\n", source, count=1)

    # Change only the default belonging to the Stocks in Price Chart multiselect.
    selector_pattern = re.compile(
        r'(?P<prefix>'
        r'chart_tickers\s*=\s*st\.multiselect\(\s*'
        r'"Stocks in Price Chart"\s*,'
        r'(?:(?!\n[ \t]*\)).)*?'
        r'default\s*=\s*)'
        r'default_chart_tickers'
        r'(?P<suffix>\s*,)',
        flags=re.DOTALL,
    )

    updated_source, changed_count = selector_pattern.subn(
        r"\g<prefix>valid_tickers\g<suffix>",
        updated_source,
        count=1,
    )

    if changed_count != 1:
        raise RuntimeError(
            "The old five-ticker default could not be replaced safely. "
            "No changes were written."
        )

    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)

    target_path.write_text(updated_source, encoding="utf-8")

    print("Update completed successfully.")
    print(f"Updated file: {target_path}")
    print(f"Backup file:  {backup_path}")
    print(
        'Every valid ticker is now selected initially in '
        '"Stocks in Price Chart".'
    )

    if removed_count == 0:
        print(
            "Note: the old helper variable was not present, but the selector "
            "default was updated successfully."
        )


if __name__ == "__main__":
    main()

