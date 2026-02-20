"""Tests the step 'Write Report'"""

#   Test module import
from toolbox.steps.custom import write_report
import pytest

#   Other imports
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock  #   Patch for OS, MagicMock for .rst stream object
import xarray as xr
import json
import numpy as np


def test_add_log(tmp_path):
    log_content = """\
2026-02-17 12:56:08 - INFO - toolbox.pipeline - Logging to file: /toolbox/examples/data/OG1/testing/processing.log
2026-02-17 12:56:08 - INFO - toolbox.pipeline - Assembling steps to run from config.
2026-02-17 12:56:08 - INFO - toolbox.pipeline - Step 'Load OG1' added successfully!
2026-02-17 12:56:18 - WARNING - toolbox.pipeline.step.Apply QC - [Apply QC] PROFILE_NUMBER_QC is all 0 after running all QC steps. Check intended QC variables and test requirements.
2026-02-17 12:56:23 - WARNING - toolbox.pipeline.step.Write Data Report - [Write Data Report] Lines below this will not be captured in the run report. See logfile if other steps follow this one.
"""

    logfile = tmp_path / "processing.log"
    logfile.write_text(log_content)

    doc = MagicMock()

    write_report.add_log(str(logfile), doc)
    doc.h2.assert_called_once_with("Logfile of run")

    #   Pull the data out of the output table
    kwargs = doc.table_list.call_args.kwargs
    rows = kwargs["data"]

    assert rows[0] == (
        "12:56:08",  #  Shouldn't have a date on it
        "INFO",
        "pipeline",  #  Prefix should be removed
        "Logging to file: /toolbox/examples/data/OG1/testing/processing.log",
    )

    # Check WARNING row with deeper path
    assert rows[3][0] == "12:56:18"
    assert rows[3][1] == "WARNING"
    assert rows[3][2] == "pipeline.step.Apply QC"

    # Ensure padding applied for formatting difficulties
    assert len(rows) >= 28

    doc.newline.assert_called()


@pytest.fixture
def qc_dataset():
    ds = xr.Dataset()

    attrs = {
        "range_test_flag_cts": json.dumps({"1": 10, "4": 2}),
        "range_test_stats": json.dumps({"mean": 1.2}),
        "range_test_params": json.dumps({"threshold": [-2.5, 40]}),
        "gross_range_test_flag_cts": json.dumps({"1": 8}),
        # intentionally omit stats to test default {}
        "gross_range_test_params": json.dumps({"fail": [0, 20]}),
    }

    ds["TEMP_QC"] = xr.DataArray(np.zeros(5), attrs=attrs)

    #   Non-QC variable for reference
    ds["TEMP"] = xr.DataArray(np.zeros(5))

    return ds


def test_build_qc_dict(qc_dataset):
    result = write_report.build_qc_dict(qc_dataset)

    assert "TEMP_QC" in result
    assert "range_test" in result["TEMP_QC"]
    assert "gross_range_test" in result["TEMP_QC"]

    range_test = result["TEMP_QC"]["range_test"]

    assert range_test["params"] == {"threshold": [-2.5, 40]}
    assert range_test["flag_counts"] == {"1": 10, "4": 2}
    assert range_test["stats"] == {"mean": 1.2}


def test_flatten_qc_dict(qc_dataset):
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


def test_qc_section(qc_dataset):
    doc = (
        MagicMock()
    )  #   Fake object that records what is done to it (representing the file)

    write_report.qc_section(doc, qc_dataset)

    #   Header called once, table and newline called however many times
    doc.h2.assert_called_once_with("Quality Control Summary")
    assert doc.table.called
    assert doc.newline.called

    headers, rows = doc.table.call_args[0]

    assert headers == [
        "QC Variable",
        "Test",
        "Flag",
        "Count",
    ]
    assert len(rows) > 0


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
