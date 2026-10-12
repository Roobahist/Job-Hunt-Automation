from job_hunt.integrations.telegram import TelegramNotifier

CAPTION = """Software Engineer
🏢 Example

Status: ⏳ Processing
✓ Normalize job: 1.2s
✓ Persist to Baserow: 0.8s
▶ Qualification: started 2:41:37 PM
Retry in: 300s"""


def test_debug_caption_keeps_timing_details() -> None:
    notifier = TelegramNotifier("token", debug=True)
    assert notifier._visible_caption(CAPTION) == CAPTION


def test_non_debug_caption_keeps_only_general_status() -> None:
    notifier = TelegramNotifier("token", debug=False)
    assert (
        notifier._visible_caption(CAPTION)
        == """Software Engineer
🏢 Example

Status: ⏳ Processing"""
    )


def test_non_debug_failed_caption_keeps_error_but_not_timings() -> None:
    caption = CAPTION + "\nError: LaTeX compilation failed"
    notifier = TelegramNotifier("token", debug=False)
    assert (
        notifier._visible_caption(caption)
        == """Software Engineer
🏢 Example

Status: ⏳ Processing
Error: LaTeX compilation failed"""
    )
