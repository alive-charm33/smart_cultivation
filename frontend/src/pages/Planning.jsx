import { useEffect, useState } from "react";
import { api, diseaseName, rupees } from "../api";
import { Alert, Card, Field, PageHeader } from "../components/ui.jsx";

export default function Planning({ farm, farmId }) {
  return (
    <>
      <PageHeader
        title="Tomato Planning"
        subtitle="Step 4: cultivation cost estimate and a step-by-step farming roadmap dated from your transplanting date."
      />
      <CostEstimate farm={farm} />
      <Roadmap farm={farm} farmId={farmId} />
    </>
  );
}

function CostEstimate({ farm }) {
  const [area, setArea] = useState(farm?.area_acres || 1);
  const [overrides, setOverrides] = useState({});
  const [yieldT, setYieldT] = useState(20);
  const [price, setPrice] = useState(10);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => setArea(farm?.area_acres || 1), [farm]);
  useEffect(() => {
    const t = setTimeout(() => {
      api.post("/api/planning/cost", {
        area_acres: Number(area) || 1, overrides,
        yield_t_per_acre: Number(yieldT) || null, price_per_kg: Number(price) || null,
      }).then((r) => { setRes(r); setErr(null); }).catch((e) => setErr(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [area, overrides, yieldT, price]);

  const edit = (key, field) => (e) =>
    setOverrides({ ...overrides, [key]: { ...(overrides[key] || {}), [field]: Number(e.target.value) } });

  return (
    <Card title="Cultivation cost estimate" icon="💰" right={<button className="ghost" onClick={() => setOverrides({})}>Reset rates</button>}>
      {err && <Alert kind="error">{err}</Alert>}
      <div className="row gap">
        <Field label="Area (acres)"><input type="number" step="0.1" value={area} onChange={(e) => setArea(e.target.value)} /></Field>
        <Field label="Expected yield (t/acre)"><input type="number" value={yieldT} onChange={(e) => setYieldT(e.target.value)} /></Field>
        <Field label="Expected price (₹/kg)"><input type="number" value={price} onChange={(e) => setPrice(e.target.value)} /></Field>
      </div>
      {res && (
        <>
          <div className="stats">
            <div><small>Total cost</small><strong>{rupees(res.total_cost)}</strong></div>
            <div><small>Cost / acre</small><strong>{rupees(res.cost_per_acre)}</strong></div>
            <div><small>Expected revenue</small><strong>{rupees(res.expected_revenue)}</strong></div>
            <div className={res.expected_profit >= 0 ? "good" : "bad"}><small>Expected profit</small><strong>{rupees(res.expected_profit)}</strong></div>
            <div><small>Break-even price</small><strong>₹{res.breakeven_price_per_kg}/kg</strong></div>
          </div>
          <table className="table">
            <thead><tr><th>Item</th><th>Qty / acre</th><th>Unit</th><th>Rate (₹)</th><th className="num">Amount</th></tr></thead>
            <tbody>
              {res.items.map((it) => (
                <tr key={it.key}>
                  <td>{it.label}</td>
                  <td><input className="mini" type="number" value={overrides[it.key]?.quantity ?? it.quantity} onChange={edit(it.key, "quantity")} /></td>
                  <td className="muted">{it.unit}</td>
                  <td><input className="mini" type="number" value={overrides[it.key]?.rate ?? it.rate} onChange={edit(it.key, "rate")} /></td>
                  <td className="num">{rupees(it.amount)}</td>
                </tr>
              ))}
              <tr><td colSpan="4">Contingency ({res.contingency_pct}%)</td><td className="num">{rupees(res.contingency)}</td></tr>
              <tr className="total"><td colSpan="4">Total for {res.area_acres} acre(s)</td><td className="num">{rupees(res.total_cost)}</td></tr>
            </tbody>
          </table>
          <p className="muted small">{res.note}</p>
        </>
      )}
    </Card>
  );
}

function Roadmap({ farm, farmId }) {
  const [date, setDate] = useState(farm?.transplant_date || "");
  const [res, setRes] = useState(null);
  useEffect(() => setDate(farm?.transplant_date || ""), [farm]);
  useEffect(() => {
    const qs = new URLSearchParams();
    if (farmId) qs.set("farm_id", farmId);
    if (date) qs.set("transplant_date", date);
    api.get("/api/planning/roadmap?" + qs).then(setRes).catch(() => {});
  }, [farmId, date]);
  const today = new Date().toISOString().slice(0, 10);

  return (
    <Card title="Farming roadmap" icon="🗺️">
      <div className="row gap">
        <Field label="Transplanting date" hint="Change it to plan a future season.">
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </Field>
      </div>
      <ol className="roadmap">
        {res?.steps.map((s) => {
          const now = s.start && s.start <= today && today <= s.end;
          const past = s.end && s.end < today;
          return (
            <li key={s.stage_id} className={(now ? "now " : "") + (past ? "past" : "")}>
              <div className="when">
                {s.start ? <>{s.start}<br />→ {s.end}</> : s.days || ""}
                {now && <span className="pill ok">Now</span>}
              </div>
              <div>
                <h4>{s.stage_name} {s.days && <small className="muted">({s.days})</small>}</h4>
                <ul>{s.tasks.map((t, i) => <li key={i}>{t}</li>)}</ul>
                {s.watch_for?.length > 0 && (
                  <div className="small muted">Watch for: {s.watch_for.map(diseaseName).join(", ")}</div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}
