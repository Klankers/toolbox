import os
import datetime as _dt
import numpy as np
import xarray as xr

# --- small helpers reused below ---


def _ensure_dirs(path):
    os.makedirs(path, exist_ok=True)
    return path


def _timestamped_dir(prefix="exports", base=None):
    root = base or os.getcwd()
    stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return _ensure_dirs(os.path.join(root, f"{prefix}_{stamp}"))


def _alias_map_for(manager, source_name):
    # alignment_map: {std_var: {glider_name: alias_name}}
    # returns {alias_name: std_var} for this source so renaming makes std vars exist
    return {
        alias: std
        for std, mapping in manager.alignment_map.items()
        if (alias := mapping.get(source_name))
    }


def _fit_from_r2_ds(r2_ds, target_name, ancillary_name, var):
    """
    Fit y = slope*x + intercept with:
      x = median_{var}_{ancillary_name} (all PAIR_INDEX × DEPTH_bin)
      y = median_{var}_TARGET_{target_name}
    Returns slope, intercept, r2, n.  NaNs → no fit.
    """
    x_name = f"median_{var}_{ancillary_name}"
    y_name = f"median_{var}_TARGET_{target_name}"
    if x_name not in r2_ds or y_name not in r2_ds:
        return np.nan, np.nan, np.nan, 0

    x = r2_ds[x_name].values.ravel()
    y = r2_ds[y_name].values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.allclose(x, x[0], equal_nan=False):
        return np.nan, np.nan, np.nan, int(x.size)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = float(model.score(x.reshape(-1, 1), y))
    return float(model.coef_[0]), float(model.intercept_), r2, int(x.size)


def _append_history(attrs: dict, line: str) -> dict:
    """Return a copy of attrs with 'history' appended with a new line."""
    out = dict(attrs) if attrs else {}
    ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    prev = out.get("history", "").rstrip()
    add = f"[{ts}] {line}"
    out["history"] = prev + ("\n" if prev else "") + add
    return out


def _copy_global_attrs(
    dst_ds: xr.Dataset, src_ds: xr.Dataset, history_line: str = None
):
    """
    Copy dataset-level attrs from src to dst (non-destructive), optionally append a history line.
    dst_ds.attrs <- src_ds.attrs ∪ dst_ds.attrs (dst wins on key conflicts).
    """
    merged = dict(src_ds.attrs or {})
    merged.update(dst_ds.attrs or {})
    if history_line:
        merged = _append_history(merged, history_line)
    dst_ds.attrs = merged


def _copy_coord_attrs(dst_ds: xr.Dataset, src_ds: xr.Dataset, coord_names=None):
    """
    Copy coordinate attrs from src to dst where names overlap.
    """
    names = coord_names or list(dst_ds.coords)
    for name in names:
        if (name in dst_ds.coords) and (name in src_ds.coords):
            dst_ds.coords[name].attrs = dict(src_ds.coords[name].attrs)


def _copy_var_attrs(
    dst_da: xr.DataArray, src_da: xr.DataArray, extra: dict | None = None
):
    """
    Copy variable attrs from src to dst, optionally adding extras (dst wins on conflicts).
    """
    attrs = dict(src_da.attrs or {})
    attrs.update(dst_da.attrs or {})
    if extra:
        attrs.update(extra)
    dst_da.attrs = attrs


def _source_name_for_standard_var(
    pmanager, pipeline_name: str, std_var: str
) -> str | None:
    """
    Given a standard variable name (e.g. 'CNDC') and a pipeline name, return the
    actual variable name in that pipeline's raw dataset using alignment_map.
    Returns None if not found.
    """
    amap = pmanager.alignment_map.get(std_var, {})
    alias = amap.get(pipeline_name)
    return (
        alias
        if alias
        else (std_var if std_var in pmanager._contexts[pipeline_name]["data"] else None)
    )


def _enrich_processed_attrs(
    pmanager,
    pipeline_name: str,
    ds_processed: xr.Dataset,
    std_vars: list[str],
    bin_dim="DEPTH_bin",
):
    """
    - Copy global attrs from the raw pipeline dataset.
    - Copy coordinate attrs where possible.
    - For each median_{var}, copy attrs from the raw var and add cell_methods.
    """
    raw_ds = pmanager._contexts[pipeline_name]["data"]

    _copy_global_attrs(
        ds_processed,
        raw_ds,
        history_line=f"Processed medians computed (per {bin_dim}) for {pipeline_name}.",
    )
    _copy_coord_attrs(ds_processed, raw_ds)

    for var in std_vars:
        med = f"median_{var}"
        if med not in ds_processed:
            continue
        # find the source variable name in raw dataset via alignment map
        src_name = _source_name_for_standard_var(pmanager, pipeline_name, var)
        if (src_name is not None) and (src_name in raw_ds):
            _copy_var_attrs(
                ds_processed[med],
                raw_ds[src_name],
                extra={
                    "cell_methods": f"{bin_dim}: median (computed from N_MEASUREMENTS)",
                    "comment": f"Median over {bin_dim} by profile.",
                    "derived_from": src_name,
                    "pipeline": pipeline_name,
                },
            )
    return ds_processed


def _enrich_combined_attrs(
    pmanager, target: str, ds_combined: xr.Dataset, std_vars: list[str]
):
    """
    - Copy global attrs from target raw dataset.
    - Copy coordinate attrs from target raw where names overlap.
    - For each median_{var} (target side), copy attrs from target raw var.
    - For each {var}_ALIGNED_TO_{target}, copy attrs from the first ancillary raw that has that var
      and add alignment provenance.
    """
    tgt_raw = pmanager._contexts[target]["data"]
    _copy_global_attrs(
        ds_combined,
        tgt_raw,
        history_line=f"Combined target dataset assembled for {target}.",
    )
    _copy_coord_attrs(ds_combined, tgt_raw)

    # median_* on target side
    for var in std_vars:
        name = f"median_{var}"
        if name in ds_combined:
            src_name = _source_name_for_standard_var(pmanager, target, var)
            if (src_name is not None) and (src_name in tgt_raw):
                _copy_var_attrs(
                    ds_combined[name],
                    tgt_raw[src_name],
                    extra={
                        "cell_methods": "DEPTH_bin: median (computed from N_MEASUREMENTS)",
                        "derived_from": src_name,
                        "pipeline": target,
                    },
                )

    # aligned vars: {VAR}_ALIGNED_TO_{target}
    suffix = f"_ALIGNED_TO_{target}"
    for vname in list(ds_combined.data_vars):
        if not vname.endswith(suffix):
            continue
        base = vname[: -len(suffix)]
        # find any ancillary whose raw dataset contains base (consider alias mapping)
        found_src = None
        for anc in pmanager._contexts.keys():
            if anc == target:
                continue
            anc_raw = pmanager._contexts[anc]["data"]
            src_name = _source_name_for_standard_var(pmanager, anc, base)
            if (src_name is not None) and (src_name in anc_raw):
                _copy_var_attrs(
                    ds_combined[vname],
                    anc_raw[src_name],
                    extra={
                        "comment": f"{base} aligned to {target} via linear fit y = a*x + b.",
                        "alignment_target": target,
                        "alignment_source": anc,
                        "derived_from": src_name,
                    },
                )
                found_src = anc
                break
        if not found_src:
            # fall back to at least mark the variable with minimal provenance
            ds_combined[vname].attrs.update(
                {
                    "comment": f"{base} aligned to {target} via linear fit y = a*x + b.",
                    "alignment_target": target,
                }
            )
    return ds_combined
