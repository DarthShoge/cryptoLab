import { useState } from "react";
import type { components } from "./api.generated";
import { useResource } from "./api";
import type { Dataset } from "./labApi";
import {
  changeClass,
  changeMode,
  classes,
  generalUniverse,
  type MarketUniverse,
} from "./marketConfig";
import { Status } from "./Status";

export function MarketUniverseBuilder({
  dataset,
  value,
  onChange,
  lockReselection = false,
}: {
  dataset?: Dataset;
  value: MarketUniverse;
  onChange: (value: MarketUniverse) => void;
  lockReselection?: boolean;
}) {
  const [page, setPage] = useState(1),
    [search, setSearch] = useState(""),
    [notice, setNotice] = useState("");
  const query = new URLSearchParams({
    page: String(page),
    page_size: "50",
    search,
  });
  const catalogue = useResource<components["schemas"]["Page_InstrumentRow_"]>(
    dataset
      ? `/api/lab/datasets/${encodeURIComponent(dataset.id)}/instruments?${query}`
      : null,
  );
  const reset = (next: MarketUniverse) => {
    onChange(next);
    setPage(1);
    setNotice(
      "Explicit selections and custom budgets are cleared when changing classes or selection mode.",
    );
  };
  return (
    <fieldset>
      <legend>Copied markets</legend>
      <p className="muted">
        Choose the market universe before ranking traders. BTC benchmark data
        remains independent.
      </p>
      <div className="check-row">
        <label>
          <input
            type="checkbox"
            checked={value.general}
            disabled={!dataset?.liquidity_available}
            onChange={(e) =>
              reset(
                e.target.checked
                  ? generalUniverse(value)
                  : { ...value, general: false, classes: [] },
              )
            }
          />
          General
        </label>
        {classes.map(([id, title]) => (
          <label
            key={id}
            title={
              !dataset?.supported_classes?.includes(id)
                ? "Unavailable in this local dataset"
                : undefined
            }
          >
            <input
              type="checkbox"
              checked={value.classes.includes(id)}
              disabled={!dataset?.supported_classes?.includes(id)}
              onChange={(e) => reset(changeClass(value, id, e.target.checked))}
            />
            {title}
          </label>
        ))}
      </div>
      {!dataset?.liquidity_available && (
        <small>
          General / liquidity selection requires a dataset with market-wide
          daily USD volume. Unavailable classes are disabled.
        </small>
      )}
      <div className="form-grid">
        <label>
          Market selection mode
          <select
            aria-label="Market selection mode"
            value={value.mode}
            disabled={value.general}
            onChange={(e) =>
              reset(changeMode(value, e.target.value as MarketUniverse["mode"]))
            }
          >
            <option value="explicit">Select instruments</option>
            <option value="liquidity" disabled={!dataset?.liquidity_available}>
              Top N by market volume
            </option>
          </select>
        </label>
        <label>
          Market reselection
          <select
            aria-label="Market reselection"
            disabled={lockReselection}
            value={value.reselection}
            onChange={(e) =>
              onChange({
                ...value,
                reselection: e.target.value as MarketUniverse["reselection"],
              })
            }
          >
            {["daily", "weekly", "monthly"].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
      </div>
      {value.mode === "liquidity" ? (
        <>
          <div className="form-grid">
            <label>
              Top N markets
              <input
                aria-label="Top N markets"
                type="number"
                min="1"
                max="25"
                step="1"
                value={value.top_n}
                onChange={(e) =>
                  onChange({ ...value, top_n: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Market volume lookback (days)
              <input
                aria-label="Market volume lookback (days)"
                type="number"
                min="1"
                step="1"
                value={value.lookback_days}
                onChange={(e) =>
                  onChange({ ...value, lookback_days: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Minimum window volume (USD)
              <input
                aria-label="Minimum window volume (USD)"
                type="number"
                min="0"
                value={value.min_volume_usd}
                onChange={(e) =>
                  onChange({ ...value, min_volume_usd: Number(e.target.value) })
                }
              />
            </label>
          </div>
          <small>
            Rank the combined pool by market-wide traded USD notional, with a
            fixed one-day publication lag. Equal budgets across selected
            markets; missing trader cohorts keep their allocation in cash.
          </small>
        </>
      ) : (
        <>
          <label>
            Find instruments
            <input
              aria-label="Find instruments"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
            />
          </label>
          <Status {...catalogue} />
          <div className="check-row">
            {catalogue.data?.rows
              .filter((row) =>
                value.classes.includes(
                  row.asset_class as (typeof value.classes)[number],
                ),
              )
              .map((row) => (
                <label key={`${row.instrument_id}:${row.effective_from}`}>
                  <input
                    type="checkbox"
                    checked={value.instrument_ids.includes(row.instrument_id)}
                    disabled={!row.supported}
                    onChange={(e) => {
                      const ids = e.target.checked
                        ? [
                            ...new Set([
                              ...value.instrument_ids,
                              row.instrument_id,
                            ]),
                          ].sort()
                        : value.instrument_ids.filter(
                            (id) => id !== row.instrument_id,
                          );
                      onChange({
                        ...value,
                        instrument_ids: ids,
                        allocation: "equal",
                        weights: null,
                      });
                      setNotice(
                        "Instrument selection changed: asset budgets reset to equal allocation.",
                      );
                    }}
                  />
                  {row.instrument_id} · {row.display_name}
                  {!row.supported ? " (unsupported contract)" : ""}
                </label>
              ))}
          </div>
          <div className="pagination">
            <button disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
              Previous instruments
            </button>
            <span>Catalogue page {page}</span>
            <button
              disabled={page * 50 >= (catalogue.data?.total ?? 0)}
              onClick={() => setPage((p) => p + 1)}
            >
              More instruments
            </button>
          </div>
          <p>
            Selected:{" "}
            {value.instrument_ids.join(", ") ||
              "None — select at least one instrument"}
          </p>
          <label>
            Asset allocation
            <select
              aria-label="Asset allocation"
              value={value.allocation}
              onChange={(e) =>
                onChange({
                  ...value,
                  allocation: e.target.value as "equal" | "custom",
                  weights:
                    e.target.value === "custom"
                      ? Object.fromEntries(
                          value.instrument_ids.map((id) => [
                            id,
                            1 / value.instrument_ids.length,
                          ]),
                        )
                      : null,
                })
              }
            >
              <option value="equal">Equal across selected markets</option>
              <option value="custom">Custom budgets</option>
            </select>
          </label>
          {value.allocation === "custom" && (
            <div className="form-grid">
              {value.instrument_ids.map((id) => (
                <label key={id}>
                  {id} asset budget
                  <input
                    type="number"
                    aria-label={`${id} asset budget`}
                    value={value.weights?.[id] ?? 0}
                    step="any"
                    onChange={(e) =>
                      onChange({
                        ...value,
                        weights: {
                          ...value.weights,
                          [id]: Number(e.target.value),
                        },
                      })
                    }
                  />
                </label>
              ))}
            </div>
          )}
          <small>
            Only continuous USD-linear contracts with unit multiplier are
            supported. Catalogue presence is not a claim that real non-crypto
            data is qualified.
          </small>
        </>
      )}
      {notice && (
        <p role="status" className="muted">
          {notice}
        </p>
      )}
    </fieldset>
  );
}
