import { useEffect, useState } from "react";
import { api, diseaseName, media, rupees, stageName } from "../api";
import { Card, Empty, PageHeader, Spinner, StageTrack } from "../components/ui.jsx";
import { Timeline } from "./GrowthMonitor.jsx";

export default function Dashboard({ farm, farmId, goto }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!farmId) return setD(null);
    setErr(null);
    api.get(`/api/dashboard/${farmId}`).then(setD).catch((e) => setErr(e.message));
  }, [farmId]);

  if (!farmId)
    return (
      <>
        <PageHeader title="Farmer Dashboard" subtitle="Everything about your tomato crop in one place." />
        <Card>
          <Empty>
            No farm yet. <button onClick={() => goto("farm")}>Create your farm</button>
          </Empty>
        </Card>
      </>
    );
  if (err) return <Card><Empty>{err}</Empty></Card>;
  if (!d) return <Spinner label="Loading dashboard..." />;

  const s = d.suitability;
  const g = d.latest_growth;
  const scan = d.recent_scans[0];
  return (
    <>
      <PageHeader
        title={`${d.farm.name}`}
        subtitle={[d.farm.farmer_name, d.farm.village, `${d.farm.area_acres} acre(s)`,
          d.farm.transplant_date ? `transplanted ${d.farm.transplant_date} · day ${d.farm.days_after_transplant}` : "not transplanted yet"]
          .filter(Boolean).join(" · ")}
      />
      <div className="grid four">
        <div className="kpi" onClick={() => goto("farm")}>
          <small>Suitability</small>
          <strong className={s ? (s.suitable ? "good" : "bad") : ""}>{s ? (s.suitable ? "Suitable" : "Not suitable") : "Not checked"}</strong>
          <span>{s ? `${Math.round(s.probability * 100)}% P(suitable) · soil ${s.fertility?.overall}` : "Run the soil check"}</span>
        </div>
        <div className="kpi" onClick={() => goto("growth")}>
          <small>Growth stage</small>
          <strong>{g ? stageName(g.stage_id) : d.expected_stages.length ? stageName(d.expected_stages[0]) : "-"}</strong>
          <span>
            {g
              ? (g.progress?.stage_source === "days"
                  ? "estimated from days (no vision)"
                  : (g.progress?.status || "").replace("_", " "))
              : d.expected_stages.length ? "estimated from days" : "Add transplant date"}
          </span>
        </div>
        <div className="kpi" onClick={() => goto("disease")}>
          <small>Latest leaf scan</small>
          <strong className={scan ? (scan.disease_id === "healthy" ? "good" : "bad") : ""}>{scan ? diseaseName(scan.disease_id) : "No scans"}</strong>
          <span>{scan ? `${Math.round(scan.confidence * 100)}% · ${scan.filename}` : "Upload a leaf photo"}</span>
        </div>
        <div className="kpi" onClick={() => goto("planning")}>
          <small>Estimated cost</small>
          <strong>{rupees(d.cost.total_cost)}</strong>
          <span>profit at ₹{d.cost.price_per_kg}/kg: {rupees(d.cost.expected_profit)}</span>
        </div>
      </div>

      <Card title="Crop progress" icon="📈">
        <StageTrack current={g?.stage_id || d.expected_stages[0]} expected={d.expected_stages} />
        {g?.progress?.message && <p className="muted">{g.progress.message}</p>}
        {d.growth_timeline.length > 0 ? (
          <Timeline history={d.growth_timeline.map((t) => ({ ...t, days_after_transplant: t.days }))} />
        ) : (
          <p className="muted small">Upload whole-plant photos in Growth Monitor to build the timeline.</p>
        )}
      </Card>

      <div className="grid two">
        <Card title={d.current_tasks ? `This week: ${d.current_tasks.stage_name}` : "Current tasks"} icon="✅">
          {d.current_tasks ? (
            <>
              <ul>{d.current_tasks.tasks.map((t, i) => <li key={i}>{t}</li>)}</ul>
              {d.current_tasks.watch_for?.length > 0 && (
                <p className="small muted">Watch for: {d.current_tasks.watch_for.map(diseaseName).join(", ")}</p>
              )}
            </>
          ) : (
            <Empty>Set the transplanting date on the farm to see dated tasks.</Empty>
          )}
        </Card>
        <Card title="Why this suitability? (SHAP)" icon="🧪">
          {s?.top_reasons ? (
            <ul className="reasons">
              {s.top_reasons.map((r) => (
                <li key={r.feature} className={r.impact >= 0 ? "pos" : "neg"}>
                  <strong>{r.label || r.feature}</strong> = {r.value} → {r.direction}
                </li>
              ))}
            </ul>
          ) : (
            <Empty>No suitability check yet.</Empty>
          )}
        </Card>
      </div>

      <Card title="Recent disease scans" icon="🍃">
        {!d.recent_scans.length ? (
          <Empty>No scans yet.</Empty>
        ) : (
          <div className="thumbs">
            {d.recent_scans.map((r) => (
              <div className="thumb" key={r.id}>
                {r.gradcam_url && <img src={media(r.gradcam_url)} alt={r.filename} />}
                <div><strong>{diseaseName(r.disease_id)}</strong> {Math.round(r.confidence * 100)}%</div>
                <small>{r.filename}</small>
              </div>
            ))}
          </div>
        )}
      </Card>
    </>
  );
}
