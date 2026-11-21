import { useEffect, useState } from "react";
import { api, diseaseName, media, STAGES, stageName } from "../api";
import { Alert, Bar, Card, Empty, Field, ImagePicker, Markdown, PageHeader, Spinner } from "../components/ui.jsx";

export default function DiseaseScan({ farm, farmId }) {
  const [file, setFile] = useState(null);
  const [stage, setStage] = useState("");
  const [question, setQuestion] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState([]);
  const [view, setView] = useState("overlay");

  const loadHistory = () =>
    api.get("/api/disease/history" + (farmId ? `?farm_id=${farmId}` : "")).then(setHistory).catch(() => {});
  useEffect(() => {
    loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [farmId]);

  const run = async () => {
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      const r = await api.upload("/api/disease/predict", {
        file, farm_id: farmId, stage, question, use_llm: useLlm,
      });
      setRes(r);
      loadHistory();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Leaf Disease Detection"
        subtitle="Steps 7-8: EfficientNet-B0 (best_tomato_model.h5) classifies the leaf, Grad-CAM shows where it looked, and RAG maps the disease to its cure for your growth stage."
      />
      <div className="grid two">
        <Card title="Upload a leaf photo" icon="🍃">
          <ImagePicker file={file} onChange={(f) => { setFile(f); setRes(null); }}
            hint="One leaf, natural light, spots clearly visible." />
          <div className="row gap">
            <Field label="Growth stage seen on plant" hint={farm?.transplant_date ? `Blank = estimate from ${farm.days_after_transplant} days after transplanting` : "Optional"}>
              <select value={stage} onChange={(e) => setStage(e.target.value)}>
                <option value="">{farm?.transplant_date ? "Estimate from transplant date" : "Not specified"}</option>
                {STAGES.map(([id, n]) => <option key={id} value={id}>{n}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Question (English / मराठी / हिंदी)">
            <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="How do I treat this?" />
          </Field>
          <label className="check">
            <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
            Use LLM for the advice (off = answer straight from the knowledge base)
          </label>
          <button disabled={!file || busy} onClick={run}>Detect disease</button>
          {busy && <Spinner label="Running the model, Grad-CAM and RAG..." />}
          {err && <Alert kind="error">{err}</Alert>}
        </Card>

        <Card title="Result" icon="🔬">
          {!res ? (
            <Empty>Upload a leaf photo and press “Detect disease”.</Empty>
          ) : (
            <div className="result">
              <div className="muted small">Input file: <strong>{res.filename}</strong> · preprocessing: {res.preprocess}</div>
              <div className={"verdict " + (res.is_healthy ? "good" : "bad")}>
                {res.is_healthy ? "✅ " : "🦠 "}
                {res.disease_info.common_name || diseaseName(res.prediction.disease_id)}
                <span>{(res.prediction.confidence * 100).toFixed(1)}% confidence</span>
              </div>
              {!res.is_healthy && res.disease_info.pathogen && (
                <p className="muted">Cause: {res.disease_info.pathogen} · {res.disease_info.curable}</p>
              )}
              {res.prediction.confidence < 0.5 && (
                <Alert kind="warn">Low confidence - retake the photo (single leaf, natural light, both sides).</Alert>
              )}
              <div className="tabs small">
                <button className={view === "overlay" ? "active" : ""} onClick={() => setView("overlay")}>Grad-CAM overlay</button>
                <button className={view === "side" ? "active" : ""} onClick={() => setView("side")}>Side by side</button>
              </div>
              {res.gradcam.overlay_url ? (
                view === "overlay" ? (
                  <img className="result-img" src={media(res.gradcam.overlay_url)} alt="Grad-CAM" />
                ) : (
                  <div className="grid two tight">
                    <figure><img className="result-img" src={media(res.image_url)} alt="original" /><figcaption>Original</figcaption></figure>
                    <figure><img className="result-img" src={media(res.gradcam.heatmap_url)} alt="heatmap" /><figcaption>Heat-map</figcaption></figure>
                  </div>
                )
              ) : (
                <Alert kind="warn">{res.gradcam.error}</Alert>
              )}
              {res.gradcam.highlighted_pct !== undefined && (
                <p className="muted small">
                  Red/yellow = regions that most influenced the prediction ({res.gradcam.highlighted_pct}% of the leaf,
                  mostly {res.gradcam.region}).
                </p>
              )}
              <h4>Top predictions</h4>
              {res.top_predictions.map((p, i) => (
                <Bar key={p.label} label={diseaseName(p.disease_id)} value={p.confidence} highlight={i === 0} />
              ))}
            </div>
          )}
        </Card>
      </div>

      {res && (
        <Card title="Cure & management (RAG)" icon="💊"
          right={<span className="pill">{res.advice.mode === "llm" ? "LLM + retrieved sources" : "Knowledge base"}</span>}>
          <p className="muted small">
            Growth stage: {res.advice.stages.length ? res.advice.stages.map(stageName).join(", ") : "not given"} · {res.advice.days_info}
          </p>
          {res.advice.warning && <Alert kind="warn">{res.advice.warning}</Alert>}
          <Markdown>{res.advice.answer}</Markdown>
          {res.advice.sources?.length > 0 && (
            <details>
              <summary>Sources ({res.advice.sources.length} retrieved passages)</summary>
              <ul className="small">
                {res.advice.sources.map((s) => <li key={s.id}><strong>{s.id}</strong> {s.file} › {s.section}</li>)}
              </ul>
            </details>
          )}
        </Card>
      )}

      <Card title="Scan history" icon="🗂️">
        {!history.length ? (
          <Empty>No scans yet.</Empty>
        ) : (
          <div className="thumbs">
            {history.map((h) => (
              <div key={h.id} className="thumb">
                <img src={media(h.gradcam_url || h.image_url)} alt={h.filename} />
                <div><strong>{diseaseName(h.disease_id)}</strong> {(h.confidence * 100).toFixed(0)}%</div>
                <small title={h.filename}>{h.filename}</small>
                <small>{new Date(h.created_at + "Z").toLocaleString()}</small>
              </div>
            ))}
          </div>
        )}
      </Card>
    </>
  );
}
