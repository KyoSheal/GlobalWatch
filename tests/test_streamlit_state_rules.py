import os
import re

from risk_profile_ui_utils import resolve_widget_profile_default


def test_widget_default_helper_rules():
    assert (
        resolve_widget_profile_default(
            source_profile="high",
            current_widget_value="",
            previous_source_profile="mid",
        )
        == "high"
    )
    assert (
        resolve_widget_profile_default(
            source_profile="high",
            current_widget_value="ultra",
            previous_source_profile="mid",
        )
        == "ultra"
    )
    assert (
        resolve_widget_profile_default(
            source_profile="low",
            current_widget_value="mid",
            previous_source_profile="mid",
        )
        == "low"
    )


def test_ui_does_not_assign_legacy_widget_key_and_init_happens_before_selectbox():
    target_path = os.path.abspath("GlobalWatch_V2.py")
    with open(target_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Legacy problematic key should be gone.
    assert "st.session_state[\"risk_profile_selected\"]" not in code
    assert "st.session_state['risk_profile_selected']" not in code

    # If widget key assignment exists, it must appear before selectbox creation.
    assign_pattern = re.compile(r"st\.session_state\[_risk_profile_widget_key\]\s*=")
    selectbox_idx = code.find('key=_risk_profile_widget_key')
    assign_positions = [m.start() for m in assign_pattern.finditer(code)]
    assert selectbox_idx > 0
    assert assign_positions
    assert all(pos < selectbox_idx for pos in assign_positions)

