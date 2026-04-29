"""First-launch consent + DPDP Act 2023 notice.

State is stored in `~/.sahaayak/consent.json` (per-user, local). The dialog
appears once; subsequent launches read the file and skip it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sahaayak.utils.i18n import i18n
from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

CONSENT_PATH = Path.home() / ".sahaayak" / "consent.json"


def has_accepted() -> bool:
    """Return ``True`` if the user has previously accepted."""
    if not CONSENT_PATH.exists():
        return False
    try:
        data = json.loads(CONSENT_PATH.read_text(encoding="utf-8"))
        return bool(data.get("accepted"))
    except (json.JSONDecodeError, OSError):
        return False


def record_acceptance() -> Path:
    """Persist the user's acceptance. Returns the file path."""
    CONSENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSENT_PATH.write_text(
        json.dumps({"accepted": True, "ts": round(time.time(), 3)}, indent=2),
        encoding="utf-8",
    )
    logger.info("Consent recorded at %s", CONSENT_PATH)
    return CONSENT_PATH


def show_consent_dialog() -> bool:
    """Show the consent dialog. Returns ``True`` if accepted.

    If PyQt6 is unavailable we fall back to a stdin prompt — useful for
    headless testing and Linux servers.
    """
    if has_accepted():
        return True
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: PLC0415
    except ImportError:
        return _stdin_consent()

    app = QApplication.instance() or QApplication([])
    box = QMessageBox()
    box.setWindowTitle(i18n("consent.title"))
    box.setText(i18n("consent.dpdp_notice"))
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.button(QMessageBox.StandardButton.Yes).setText(i18n("consent.accept"))
    box.button(QMessageBox.StandardButton.No).setText(i18n("consent.decline"))
    accepted = box.exec() == QMessageBox.StandardButton.Yes
    _ = app
    if accepted:
        record_acceptance()
    return accepted


def _stdin_consent() -> bool:
    print()
    print("=" * 72)
    print(i18n("consent.title"))
    print("-" * 72)
    print(i18n("consent.dpdp_notice"))
    print("=" * 72)
    try:
        reply = input("Accept? [y/N]: ").strip().lower()
    except EOFError:
        return False
    if reply == "y":
        record_acceptance()
        return True
    return False
