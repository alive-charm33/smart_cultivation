import { useEffect, useState } from "react";
import { api } from "../api";
import { Alert, Bar, Card, Field, PageHeader, Spinner } from "../components/ui.jsx";

const EMPTY_FARM = { name: "", farmer_name: "", village: "", lat: "", lon: "", area_acres: 1, variety: "", transplant_date: "" };
const SOIL_FIELDS = [
  ["ph", "Soil pH", 6.4, 0.1],
  ["nitrogen_ppm", "Nitrogen (total, mg/kg)", 1500, 10],
  ["phosphorus_ppm", "Phosphorus (mg/kg)", 15, 1],
  ["potassium_ppm", "Potassium (mg/kg)", 180, 1],
  ["organic_carbon_pct", "Organic carbon (%)", 2.2, 0.1],
  ["rainfall_mm", "Season rainfall (mm)", 500, 10],
  ["temperature_c", "Avg temperature (°C)", 24, 0.5],
  ["humidity_pct", "Humidity (%)", 65, 1],
];

export default function FarmSoil({ farm, farmId, reloadFarms, setFarmId }) {
  const [form, setForm] = useState(EMPTY_FARM);
  const [editing, setEditing] = useState(!farm);
  const [msg, setMsg] = useState(null);

  useEffect(() => {
    if (farm) {
      setForm({ ...EMPTY_FARM, ...Object.fromEntries(Object.entries(farm).map(([k, v]) => [k, v ?? ""])) });
      setEditing(false);
    } else {
      setForm(EMPTY_FARM);
      setEditing(true);
    }
  }, [farm]);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const useGPS = () => {
    if (!navigator.geolocation) return setMsg({ kind: "error", text: "GPS not available in this browser." });
    navigator.geolocation.getCurrentPosition(
      (p) => setForm((f) => ({ ...f, lat: p.coords.latitude.toFixed(5), lon: p.coords.longitude.toFixed(5) })),
      (e) => setMsg({ kind: "error", text: "Could not read location: " + e.message })
    );
  };

  const save = async (asNew) => {
    const body = {
      ...form,
      lat: form.lat === "" ? null : Number(form.lat),
      lon: form.lon === "" ? null : Number(form.lon),
      area_acres: Number(form.area_acres) || 1,
      transplant_date: form.transplant_date || null,
    };
    try {
      const saved = asNew || !farm ? await api.post("/api/farms", body) : await api.put(`/api/farms/${farm.id}`, body);
      await reloadFarms();
      setFarmId(saved.id);
      setEditing(false);
      setMsg({ kind: "success", text: "Farm saved." });
    } catch (e) {
      setMsg({ kind: "error", text: e.message });
    }
  };

  return (
    <>
      <PageHeader
        title="Farm & Soil Suitability"
        subtitle="Step 1-3: enter the farm location, fetch soil + climate data, and check tomato suitability with XGBoost + SHAP."
      />
      {msg && <Alert kind={msg.kind}>{msg.text}</Alert>}
      <div className="grid two">
        <Card
          title={farm && !editing ? farm.name : "Farm details"}
          icon="🏡"
          right={
            farm && !editing ? (
              <div className="row gap">
                <button className="ghost" onClick={() => setEditing(true)}>Edit</button>
                <button className="ghost" onClick={() => { setFarmId(null); setForm(EMPTY_FARM); setEditing(true); }}>New farm</button>
              </div>
            ) : null
          }
        >
          {farm && !editing ? (
            <dl className="facts">
              <dt>Farmer</dt><dd>{farm.farmer_name || "-"}</dd>
              <dt>Village</dt><dd>{farm.village || "-"}</dd>
              <dt>Location</dt><dd>{farm.lat != null ? `${farm.lat}, ${farm.lon}` : "-"}</dd>
              <dt>Area</dt><dd>{farm.area_acres} acre(s)</dd>
              <dt>Variety</dt><dd>{farm.variety || "-"}</dd>
              <dt>Transplanted</dt>
              <dd>{farm.transplant_date ? `${farm.transplant_date} (${farm.days_after_transplant} days ago)` : "not yet"}</dd>
            </dl>
          ) : (
            <div className="form">
              <Field label="Farm name *"><input value={form.name} onChange={set("name")} placeholder="e.g. Patil farm - plot 2" /></Field>
              <div className="row gap">
                <Field label="Farmer name"><input value={form.farmer_name} onChange={set("farmer_name")} /></Field>
                <Field label="Village"><input value={form.village} onChange={set("village")} /></Field>
              </div>
              <div className="row gap end">
                <Field label="Latitude"><input value={form.lat} onChange={set("lat")} placeholder="16.85" /></Field>
                <Field label="Longitude"><input value={form.lon} onChange={set("lon")} placeholder="74.57" /></Field>
                <button className="ghost" type="button" onClick={useGPS}>📍 Use GPS</button>
              </div>
              <div className="row gap">
                <Field label="Area (acres)"><input type="number" step="0.1" value={form.area_acres} onChange={set("area_acres")} /></Field>
                <Field label="Variety / hybrid"><input value={form.variety} onChange={set("variety")} /></Field>
              </div>
              <Field label="Transplanting date" hint="Used to estimate the growth stage and to date the roadmap.">
                <input type="date" value={form.transplant_date} onChange={set("transplant_date")} />
              </Field>
              <div className="row gap">
                <button disabled={!form.name} onClick={() => save(false)}>{farm ? "Save changes" : "Create farm"}</button>
                {farm && <button className="ghost" onClick={() => setEditing(false)}>Cancel</button>}
              </div>
            </div>
          )}
        </Card>
        <Suitability farm={farm} farmId={farmId} />
      </div>
    </>
  );
}

function Suitability({ farm, farmId }) {
  const [mode, setMode] = useState("location");
  const [soil, setSoil] = useState(Object.fromEntries(SOIL_FIELDS.map(([k, , d]) => [k, d])));
  const [pk, setPk] = useState({ phosphorus_ppm: "", potassium_ppm: "" });
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      let r;
      if (mode === "location") {
        if (!farm || farm.lat == null) throw new Error("Save the farm with latitude/longitude first (or use manual soil values).");
        r = await api.post("/api/suitability/by-location", {
          lat: farm.lat, lon: farm.lon, farm_id: farmId,
          phosphorus_ppm: pk.phosphorus_ppm === "" ? null : Number(pk.phosphorus_ppm),
          potassium_ppm: pk.potassium_ppm === "" ? null : Number(pk.potassium_ppm),
        });
      } else {
        r = await api.post("/api/suitability/by-soil", {
          ...Object.fromEntries(Object.entries(soil).map(([k, v]) => [k, Number(v)])),
          farm_id: farmId,
        });
      }
      setRes(r);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const maxShap = res ? Math.max(...res.shap.map((s) => Math.abs(s.impact)), 0.0001) : 1;

  return (
    <Card title="Tomato suitability" icon="🧪">
      <div className="tabs small">
        <button className={mode === "location" ? "active" : ""} onClick={() => setMode("location")}>By GPS location</button>
        <button className={mode === "manual" ? "active" : ""} onClick={() => setMode("manual")}>Manual soil test values</button>
      </div>
      {mode === "location" ? (
        <>
          <p className="muted">
            pH, organic carbon and nitrogen come from ISRIC SoilGrids; temperature, humidity and rainfall from NASA POWER.
            Phosphorus and potassium have no free API - enter them from your Soil Health Card if you have it.
          </p>
          <div className="row gap">
            <Field label="Phosphorus (mg/kg)"><input value={pk.phosphorus_ppm} onChange={(e) => setPk({ ...pk, phosphorus_ppm: e.target.value })} placeholder="optional" /></Field>
            <Field label="Potassium (mg/kg)"><input value={pk.potassium_ppm} onChange={(e) => setPk({ ...pk, potassium_ppm: e.target.value })} placeholder="optional" /></Field>
          </div>
        </>
      ) : (
        <div className="grid fields">
          {SOIL_FIELDS.map(([k, label, , step]) => (
            <Field key={k} label={label}>
              <input type="number" step={step} value={soil[k]} onChange={(e) => setSoil({ ...soil, [k]: e.target.value })} />
            </Field>
          ))}
        </div>
      )}
      <button onClick={run} disabled={busy}>Check suitability</button>
      {busy && <Spinner label="Fetching soil & climate data and running the model..." />}
      {err && <Alert kind="error">{err}</Alert>}
      {res && (
        <div className="result">
          <div className={"verdict " + (res.suitable ? "good" : "bad")}>
            {res.suitable ? "✅ Suitable for tomato" : "⚠️ Not suitable as-is"}
            <span>{(res.confidence * 100).toFixed(0)}% confidence</span>
          </div>
          <h4>Why? (SHAP feature impact)</h4>
          <p className="muted">Green bars pushed the prediction towards "suitable", red bars against it.</p>
          {res.shap.map((s) => (
            <Bar key={s.feature} label={`${s.label} = ${s.value}`} value={s.impact} max={maxShap} signed
              suffix={(s.impact > 0 ? "+" : "") + s.impact.toFixed(2)} />
          ))}
          <h4>Soil fertility status: {res.fertility.overall} ({res.fertility.optimal_count}/{res.fertility.total} optimal)</h4>
          <table className="table">
            <thead><tr><th>Parameter</th><th>Value</th><th>Ideal</th><th>Status</th><th>Source</th></tr></thead>
            <tbody>
              {res.fertility.parameters.map((p) => (
                <tr key={p.feature}>
                  <td>{p.label}</td>
                  <td>{p.value}</td>
                  <td>{p.ideal[0]}-{p.ideal[1]}</td>
                  <td><span className={"pill " + (p.band === "optimal" ? "ok" : p.band.includes("very") ? "bad" : "warn")}>{p.band}</span></td>
                  <td className="muted small">{res.data_sources?.[p.feature] || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {res.fertility.parameters.filter((p) => p.advice).length > 0 && (
            <>
              <h4>What to improve</h4>
              <ul>{res.fertility.parameters.filter((p) => p.advice).map((p) => <li key={p.feature}>{p.advice}</li>)}</ul>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
