import { useEffect, useState } from "react";
import { api, media, stageName, STAGES } from "../api";
import { Alert, Card, Empty, Field, ImagePicker, Markdown, PageHeader, Spinner, StageTrack } from "../components/ui.jsx";

const STATUS = {
  on_track: ["ok", "On track ✅"],
  behind: ["bad", "Behind schedule ⚠️"],
  ahead: ["warn", "Ahead of schedule"],
  estimated: ["warn", "Stage estimated from days — no photo analysis"],
  no_reference: ["", "No transplant date set"],
  unknown: ["", "Unknown"],
};

export default function GrowthMonitor({ farm, farmId, health }) {
  const visionOk = health?.vision?.available ?? true;  // default true until health loads
  const [file, setFile] = useState(null);
  const [days, setDays] = useState("");
  const [useLlm, setUseLlm] = useState(false);  // default off — day-based is always reliable
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState([]);

  const load = () => api.get("/api/growth/history" + (farmId ? `?farm_id=${farmId}` : "")).then(setHistory).catch(() => {});
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [farmId]);

  const run = async () => {
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      const r = await api.upload("/api/growth/analyze", { file, farm_id: farmId, days, use_llm: useLlm });
      setRes(r);
      load();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const st = res ? STATUS[res.progress?.status] || STATUS.unknown : null;

  return (
    <>
      <PageHeader
        title="Growth Monitoring"
        subtitle={
          visionOk
            ? `Steps 5-6: upload a whole-plant photo. The vision LLM (${health?.vision?.provider || "AI"}) identifies the growth stage, compares it with the expected stage for the plant's age, and RAG gives stage-wise care advice.`
            : "Steps 5-6: vision model not configured. Add GEMINI_API_KEY to backend/.env (free at aistudio.google.com) to enable plant photo analysis. Currently showing day-based stage estimate only."
        }
      />
      <div className="grid two">
        <Card title="Upload a whole-plant photo" icon="🌿">
          <ImagePicker file={file} onChange={(f) => { setFile(f); setRes(null); }} hint="Whole plant in frame, side view, daylight." />
          <Field label="Days after transplanting"
            hint={farm?.transplant_date ? `Blank = ${farm.days_after_transplant} (from the farm's transplanting date)` : "Set a transplanting date on the farm, or type the days here"}>
            <input type="number" value={days} onChange={(e) => setDays(e.target.value)} placeholder={farm?.days_after_transplant ?? ""} />
          </Field>
          <label className="check" style={{ opacity: visionOk ? 1 : 0.5 }}>
            <input type="checkbox" checked={useLlm && visionOk}
              disabled={!visionOk}
              onChange={(e) => setUseLlm(e.target.checked)} />
            Use vision LLM to identify stage from the photo
            {!visionOk && <span className="pill warn" style={{ marginLeft: 6 }}>No vision model — using day estimate</span>}
          </label>
          <button disabled={!file || busy} onClick={run}>Analyse growth</button>
          {busy && <Spinner label="Looking at the plant..." />}
          {err && <Alert kind="error">{err}</Alert>}
        </Card>

        <Card title="Growth stage" icon="📍">
          {!res ? (
            <Empty>Upload a plant photo to see its stage and progress.</Empty>
          ) : (
            <div className="result">
              <div className="muted small">Input file: <strong>{res.filename}</strong> · method: {res.method}</div>
              {res.warning && <Alert kind="warn">{res.warning}</Alert>}
              <div className="verdict neutral">
                {stageName(res.stage_id)}
                {res.confidence != null && <span>{Math.round(res.confidence * 100)}% confidence</span>}
              </div>
              <StageTrack current={res.stage_id} expected={res.expected_stages} />
              <p className="muted small">Dashed outline = stage expected for {res.days_after_transplant ?? "?"} days after transplanting.</p>
              {res.progress && (
                <div className={"progress-box " + st[0]}>
                  <strong>{st[1]}</strong>
                  {res.progress.stage_source === "days" ? (
                    <div className="small" style={{ marginTop: 4 }}>📅 {res.progress.message}</div>
                  ) : (
                    <div> - {res.progress.message}</div>
                  )}
                  {res.progress.since_last && res.progress.stage_source !== "days" && (
                    <div className="small">
                      Since last photo: {stageName(res.progress.since_last.previous_stage)} → {stageName(res.stage_id)}
                      {" "}({res.progress.since_last.stages_advanced >= 0 ? "+" : ""}{res.progress.since_last.stages_advanced} stage
                      {res.progress.since_last.days_between != null && `, ${res.progress.since_last.days_between} days`})
                    </div>
                  )}
                </div>
              )}

              {/* Mismatch detail box — shown when photo vs days comparison finds a problem */}
              {res.mismatch && res.mismatch.has_mismatch && (
                <div className="mismatch-box bad">
                  <strong>⚠️ Growth not as expected for {res.days_after_transplant} days</strong>
                  <p>{res.mismatch.health_note}</p>
                  {res.mismatch.recommendation && (
                    <div className="small mismatch-recs">
                      <strong>What to check:</strong>
                      {res.mismatch.recommendation.split(" | ").map((r, i) => (
                        <div key={i}>• {r}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {res.mismatch && !res.mismatch.has_mismatch && res.mismatch.health_note && (
                <div className="mismatch-box ok">
                  <p>{res.mismatch.health_note}</p>
                  {res.mismatch.recommendation && (
                    <div className="small">{res.mismatch.recommendation}</div>
                  )}
                </div>
              )}
              {res.observations?.length > 0 && (
                <>
                  <h4>What the model saw</h4>
                  <ul>{res.observations.map((o, i) => <li key={i}>{o}</li>)}</ul>
                </>
              )}
              {res.counts && Object.values(res.counts).some((v) => v != null) && (
                <div className="chips">
                  {res.counts.flower_clusters != null && <span className="pill">🌼 {res.counts.flower_clusters} flower clusters</span>}
                  {res.counts.green_fruits != null && <span className="pill">🟢 {res.counts.green_fruits} green fruits</span>}
                  {res.counts.ripe_fruits != null && <span className="pill">🍅 {res.counts.ripe_fruits} ripe fruits</span>}
                </div>
              )}
              {res.health_notes && <p>{res.health_notes}</p>}
            </div>
          )}
        </Card>
      </div>

      {res?.guidance && (
        <Card title={`Care for this stage: ${res.guidance.stage_name}`} icon="📘">
          <div className="grid two">
            <div>
              <h4>Key management</h4>
              <Markdown>{res.guidance.key_management}</Markdown>
              <h4>Water & nutrients</h4>
              <Markdown>{res.guidance.water_nutrients}</Markdown>
            </div>
            <div>
              <h4>Disease & pest risks now</h4>
              <Markdown>{res.guidance.disease_risks}</Markdown>
            </div>
          </div>
        </Card>
      )}

      <Card title="Growth timeline" icon="🕒">
        {!history.length ? (
          <Empty>No growth photos yet. Upload one every 7-10 days to track progress.</Empty>
        ) : (
          <>
            <Timeline history={history} />
            <div className="thumbs">
              {history.slice().reverse().map((h) => (
                <div key={h.id} className="thumb">
                  <img src={media(h.image_url)} alt={h.filename} />
                  <div><strong>{stageName(h.stage_id)}</strong></div>
                  <small>{h.days_after_transplant != null ? `Day ${h.days_after_transplant}` : ""} · {new Date(h.created_at + "Z").toLocaleDateString()}</small>
                  <small title={h.filename}>{h.filename}</small>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>
    </>
  );
}

/** Simple SVG chart: x = record (or day), y = stage index. */
export function Timeline({ history }) {
  const pts = history.filter((h) => h.stage_index != null);
  if (!pts.length) return null;
  const W = 640, H = 220, L = 150, R = 20, T = 10, B = 30;
  const xs = pts.map((p, i) => (p.days_after_transplant ?? p.days ?? i));
  const minX = Math.min(...xs), maxX = Math.max(...xs, minX + 1);
  const x = (v) => L + ((v - minX) / (maxX - minX)) * (W - L - R);
  const y = (i) => T + (1 - i / (STAGES.length - 1)) * (H - T - B);
  return (
    <svg className="timeline" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="growth stage over time">
      {STAGES.map(([id, name], i) => (
        <g key={id}>
          <line x1={L} x2={W - R} y1={y(i)} y2={y(i)} className="gridline" />
          <text x={L - 8} y={y(i) + 4} textAnchor="end" className="axis">{name}</text>
        </g>
      ))}
      <polyline points={pts.map((p, i) => `${x(xs[i])},${y(p.stage_index)}`).join(" ")} className="line" />
      {pts.map((p, i) => (
        <circle key={i} cx={x(xs[i])} cy={y(p.stage_index)} r="5" className="point">
          <title>{`${stageName(p.stage_id)} - day ${xs[i]}`}</title>
        </circle>
      ))}
      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" className="axis">days after transplanting</text>
    </svg>
  );
}
