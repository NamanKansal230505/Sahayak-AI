"""Tiny in-process translation table.

Spec: every UI string must be wrapped from day 1. Keys are the English
canonical form. Missing keys silently fall back to the key itself, so
shipping a UI string before its Hindi translation does not crash.
"""

from __future__ import annotations

from threading import RLock
from typing import Final

from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

_AVAILABLE: Final[tuple[str, ...]] = ("en", "hi")

# English keys → translations. Add new strings here as the UI grows.
_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "consent.title": "Welcome to SahaayakAI",
        "consent.dpdp_notice": (
            "SahaayakAI runs entirely on your device. No webcam frames, audio, "
            "or text ever leave this laptop. Under India's DPDP Act 2023 you "
            "have the right to withdraw consent at any time by closing the app."
        ),
        "consent.accept": "I understand and accept",
        "consent.decline": "Decline and exit",
        "rest.title": "Eye-rest reminder",
        "rest.message": "Look at something 20 feet away for 20 seconds.",
        "rest.snooze_1h": "Snooze 1 hour",
        "rest.dismiss": "Done",
        "calibration.intro": "Look at each red dot until it turns green.",
        "calibration.complete": "Calibration complete.",
        "calibration.failed": "Calibration failed. Please retry.",
        "copilot.disclaimer": "AI-generated, please review.",
        "copilot.tldr": "TL;DR",
        "copilot.action_items": "Action items",
        "copilot.suggested_replies": "Suggested replies",
        "copilot.send_blink_confirm": "Blink long to send. Look away to cancel.",
        "kill_switch.engaged": "Eye control disabled. Press F12 to re-enable.",
        "low_confidence.frozen": "Cursor paused — gaze confidence too low.",
        "focus.nudge": "Switching tasks a lot. Want to take a breath?",
        "device.banner": "Inference device: {device}",
    },
    "hi": {
        "consent.title": "SahaayakAI में आपका स्वागत है",
        "consent.dpdp_notice": (
            "SahaayakAI पूरी तरह आपके डिवाइस पर चलता है। कोई भी वेबकैम फ्रेम, "
            "ऑडियो या टेक्स्ट इस लैपटॉप से बाहर नहीं जाता। DPDP अधिनियम 2023 के "
            "अंतर्गत आप कभी भी ऐप बंद करके सहमति वापस ले सकते हैं।"
        ),
        "consent.accept": "मैं समझता/समझती हूँ और स्वीकार करता/करती हूँ",
        "consent.decline": "अस्वीकार करें और बाहर निकलें",
        "rest.title": "आँखों को आराम दें",
        "rest.message": "20 फीट दूर किसी चीज़ को 20 सेकंड तक देखें।",
        "rest.snooze_1h": "1 घंटे के लिए टालें",
        "rest.dismiss": "हो गया",
        "calibration.intro": "हर लाल बिंदु को तब तक देखें जब तक वह हरा न हो जाए।",
        "calibration.complete": "कैलिब्रेशन पूरा हुआ।",
        "calibration.failed": "कैलिब्रेशन विफल। कृपया पुनः प्रयास करें।",
        "copilot.disclaimer": "AI द्वारा निर्मित, कृपया समीक्षा करें।",
        "copilot.tldr": "संक्षेप में",
        "copilot.action_items": "करने योग्य कार्य",
        "copilot.suggested_replies": "सुझाए गए उत्तर",
        "copilot.send_blink_confirm": "भेजने के लिए लंबा पलक झपकाएँ। रद्द करने के लिए दूर देखें।",
        "kill_switch.engaged": "आँख नियंत्रण बंद। पुनः सक्षम करने के लिए F12 दबाएँ।",
        "low_confidence.frozen": "कर्सर रुका — दृष्टि की सटीकता कम है।",
        "focus.nudge": "बहुत बार कार्य बदल रहे हैं। थोड़ा रुकें?",
        "device.banner": "इन्फरेंस डिवाइस: {device}",
    },
}

_lock = RLock()
_current_language: str = "en"


def set_language(code: str) -> None:
    """Set the active UI language (``en`` or ``hi``)."""
    global _current_language  # noqa: PLW0603
    if code not in _AVAILABLE:
        logger.warning("Unknown language %r; falling back to en.", code)
        code = "en"
    with _lock:
        _current_language = code


def get_language() -> str:
    """Return the currently active language code."""
    with _lock:
        return _current_language


def i18n(key: str, **fmt: object) -> str:
    """Look up a translation by English key. Falls back to the key itself."""
    table = _TRANSLATIONS.get(get_language(), _TRANSLATIONS["en"])
    template = table.get(key) or _TRANSLATIONS["en"].get(key, key)
    if fmt:
        try:
            return template.format(**fmt)
        except (KeyError, IndexError):
            return template
    return template
