import { useState } from "react";
import { api, auth } from "../api";
import { Alert, Field } from "../components/ui.jsx";

export default function Login({ onLogin }) {
  const [mode, setMode] = useState("login"); // "login" | "signup"
  const [form, setForm] = useState({
    username: "", password: "", confirm: "",
    full_name: "", mobile: "", village: "", district: "", state: "Maharashtra",
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setErr(null);
    if (mode === "signup" && form.password !== form.confirm) {
      return setErr("Passwords do not match.");
    }
    setBusy(true);
    try {
      let res;
      if (mode === "login") {
        res = await api.post("/api/auth/login/json", {
          username: form.username, password: form.password,
        });
      } else {
        res = await api.post("/api/auth/signup", {
          username: form.username, password: form.password,
          full_name: form.full_name || null, mobile: form.mobile || null,
          village: form.village || null, district: form.district || null,
          state: form.state || "Maharashtra",
        });
      }
      const token = res.token || res.access_token;
      auth.setToken(token);
      auth.setUser(res.user);
      onLogin(res.user);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-wrap">
      <div className="auth-card">
        <div className="auth-brand">
          <span className="auth-logo">🍅</span>
          <h1>Smart Cultivation</h1>
          <p>AI-powered tomato farming assistant</p>
        </div>

        <div className="tabs">
          <button className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setErr(null); }}>
            Sign in
          </button>
          <button className={mode === "signup" ? "active" : ""} onClick={() => { setMode("signup"); setErr(null); }}>
            Create account
          </button>
        </div>

        <form onSubmit={submit} className="auth-form">
          {err && <Alert kind="error">{err}</Alert>}

          <Field label="Username *">
            <input value={form.username} onChange={set("username")}
              placeholder="e.g. patil_ramesh" autoComplete="username" required />
          </Field>
          <Field label="Password *">
            <input type="password" value={form.password} onChange={set("password")}
              placeholder="Minimum 6 characters" autoComplete={mode === "login" ? "current-password" : "new-password"} required />
          </Field>

          {mode === "signup" && (
            <>
              <Field label="Confirm password *">
                <input type="password" value={form.confirm} onChange={set("confirm")}
                  placeholder="Re-enter password" autoComplete="new-password" required />
              </Field>
              <hr className="auth-sep" />
              <p className="muted small">Farmer profile (optional — you can fill this later)</p>
              <Field label="Full name">
                <input value={form.full_name} onChange={set("full_name")} placeholder="Ramesh Patil" />
              </Field>
              <div className="row gap">
                <Field label="Mobile">
                  <input value={form.mobile} onChange={set("mobile")} placeholder="9876543210" />
                </Field>
                <Field label="State">
                  <input value={form.state} onChange={set("state")} />
                </Field>
              </div>
              <div className="row gap">
                <Field label="District">
                  <input value={form.district} onChange={set("district")} placeholder="Sangli" />
                </Field>
                <Field label="Village / Taluka">
                  <input value={form.village} onChange={set("village")} placeholder="Miraj" />
                </Field>
              </div>
            </>
          )}

          <button type="submit" disabled={busy} style={{ width: "100%", marginTop: 8 }}>
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <p className="muted small" style={{ textAlign: "center", marginTop: 12 }}>
          {mode === "login"
            ? <>No account? <button className="link-btn" onClick={() => { setMode("signup"); setErr(null); }}>Sign up free</button></>
            : <>Already have an account? <button className="link-btn" onClick={() => { setMode("login"); setErr(null); }}>Sign in</button></>
          }
        </p>
      </div>
    </div>
  );
}
