import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { STAGES } from "../api";

export function Card({ title, icon, right, children, className = "" }) {
  return (
    <section className={"card " + className}>
      {(title || right) && (
        <header>
          <h3>
            {icon && <span className="icon">{icon}</span>}
            {title}
          </h3>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, subtitle }) {
  return (
    <div className="page-header">
      <h1>{title}</h1>
      {subtitle && <p>{subtitle}</p>}
    </div>
  );
}

export function Alert({ kind = "info", children }) {
  if (!children) return null;
  return <div className={"alert " + kind}>{children}</div>;
}

export function Spinner({ label = "Working..." }) {
  return (
    <div className="spinner">
      <div className="ring" /> {label}
    </div>
  );
}

export function Markdown({ children }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children || ""}</ReactMarkdown>
    </div>
  );
}

/** Horizontal bar, value in 0..1 (or signed for SHAP with `signed`). */
export function Bar({ label, value, max = 1, signed = false, suffix, highlight }) {
  const pct = Math.min(100, (Math.abs(value) / (max || 1)) * 100);
  return (
    <div className={"bar-row" + (highlight ? " highlight" : "")}>
      <div className="bar-label">{label}</div>
      <div className={"bar-track" + (signed ? " signed" : "")}>
        {signed ? (
          <div
            className={"bar-fill " + (value >= 0 ? "pos" : "neg")}
            style={{ width: pct / 2 + "%", [value >= 0 ? "left" : "right"]: "50%" }}
          />
        ) : (
          <div className="bar-fill" style={{ width: pct + "%" }} />
        )}
      </div>
      <div className="bar-value">{suffix ? suffix : (value * 100).toFixed(1) + "%"}</div>
    </div>
  );
}

/** Upload box with preview; keeps the original File object (and its name) untouched. */
export function ImagePicker({ file, onChange, hint }) {
  const [preview, setPreview] = useState(null);
  const [drag, setDrag] = useState(false);
  const input = useRef();
  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const pick = (f) => f && f.type.startsWith("image/") && onChange(f);
  return (
    <div
      className={"dropzone" + (drag ? " drag" : "")}
      onClick={() => input.current.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        pick(e.dataTransfer.files[0]);
      }}
    >
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files[0])} />
      {preview ? (
        <>
          <img src={preview} alt="selected" />
          <div className="filename">📄 {file.name}</div>
        </>
      ) : (
        <div className="placeholder">
          <div className="big">📷</div>
          <div>Click or drop an image here</div>
          {hint && <small>{hint}</small>}
        </div>
      )}
    </div>
  );
}

export function StageTrack({ current, expected = [] }) {
  const ci = STAGES.findIndex((s) => s[0] === current);
  return (
    <div className="stage-track">
      {STAGES.map(([id, name], i) => (
        <div
          key={id}
          className={
            "stage" +
            (i < ci ? " done" : "") +
            (i === ci ? " current" : "") +
            (expected.includes(id) ? " expected" : "")
          }
          title={expected.includes(id) ? "expected for the plant's age" : ""}
        >
          <div className="pip">{i + 1}</div>
          <div className="name">{name}</div>
        </div>
      ))}
    </div>
  );
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}
