"""Tests the step 'Write Report'"""

#   Test module import
from toolbox.steps.custom import write_report
import pytest

#   Other imports
from datetime import datetime, timezone
from unittest.mock import patch


def test_flatten_qc_dict():
    qc_dict = {
        "TEMP_QC": {
            "range_test": {
                "stats": {"min": 0, "max": 30},
                "flag_counts": {
                    0: 0,
                    1: 2212921,
                    2: 0,
                    3: 2500,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                },
            }
        },
        "CNDC_QC": {},  # should be skipped entirely
    }

    result = write_report.flatten_qc_dict(qc_dict)

    expected = [
        ["TEMP_QC", "range_test", 1, "2,212,921"],
        ["TEMP_QC", "range_test", 3, "2,500"],
    ]

    assert result == expected
    assert "CNDC_QC" not in [item for sublist in expected for item in sublist]


def test_current_info():
    #   Better test - uses mock to emulate OS with context managers
    #   import getpass becomes import write_report.getpass within mock
    fixed_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with (
        patch.object(write_report, "datetime") as mock_datetime,
        patch.object(write_report.getpass, "getuser", return_value="aaron-mau"),
        patch.object(write_report, "version", return_value="0.1.dev318+gdaacfb0d8"),
        patch.object(write_report.platform, "python_version", return_value="3.14.2"),
        patch.object(write_report.platform, "system", return_value="Linux"),
        patch.object(write_report.platform, "release", return_value="6.17.0-8-generic"),
    ):
        mock_datetime.now.return_value = fixed_now
        result = write_report.current_info()

    expected = {
        "timestamp_utc": fixed_now.isoformat(),
        "user": "aaron-mau",
        "toolbox_version": "0.1.dev318+gdaacfb0d8",
        "python_version": "3.14.2",
        "system": "Linux: 6.17.0-8-generic",
    }

    assert result == expected
