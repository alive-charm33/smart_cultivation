import { useState } from "react";
import { api, auth } from "../api";
import { Alert, Card, Field, PageHeader } from "../components/ui.jsx";

export default function Profile({ user, onUpdate, onLogout }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({
    full_name: user?.full_name || "",
    mobile: user?.mobile || "",
    village: user?.village || "",
    district: user?.district || "",
    state: user?.state || "Maharashtra",
  });
  const [pwForm, setPwForm] = useState({ old_password: "", new_password: "", confirm: "" });
  const [msg, setMsg] = useState(null);
  const [pwMsg, setPwMsg] = useState(null);
  const [busy, setBusy] = useState(false);
  const [pwBusy, setPwBusy] = useState(false);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const setPw = (k) => (e) => setPwForm({ ...pwForm, [k]: e.target.value });

  const saveProfile = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const updated = await api.put("/api/auth/profile", form);
      auth.setUser(updated);
      onUpdate(updated);
      setEditing(false);
      setMsg({ kind: "success", text: "Profile updated." });
    } catch (e) {
      setMsg({ kind: "error", text: e.message });
    } finally {
      setBusy(false);
    }
  };

  const changePassword = async (e) => {
    e.preventDefault();
    if (pwForm.new_password !== pwForm.confirm)
      return setPwMsg({ kind: "error", text: "New passwords do not match." });
    setPwBusy(true);
    setPwMsg(null);
    try {
      await api.post("/api/auth/change-password", {
        old_password: pwForm.old_password,
        new_password: pwForm.new_password,
      });
      setPwForm({ old_password: "", new_password: "", confirm: "" });
      setPwMsg({ kind: "success", text: "Password changed successfully." });
    } catch (e) {
      setPwMsg({ kind: "error", text: e.message });
    } finally {
      setPwBusy(false);
    }
  };

  const handleLogout = () => {
    auth.clearToken();
    onLogout();
  };

  return (
    <>
      <PageHeader
        title="Farmer Profile"
        subtitle="Your account details and login settings."
      />

      {msg && <Alert kind={msg.kind}>{msg.text}</Alert>}

      <div className="grid two">
        <Card title="Profile details" icon="👤"
          right={
            !editing
              ? <button className="ghost" onClick={() => setEditing(true)}>Edit</button>
              : null
          }
        >
          {!editing ? (
            <dl className="facts">
              <dt>Username</dt><dd><strong>{user?.username}</strong></dd>
              <dt>Full name</dt><dd>{user?.full_name || <span className="muted">—</span>}</dd>
              <dt>Mobile</dt><dd>{user?.mobile || <span className="muted">—</span>}</dd>
              <dt>State</dt><dd>{user?.state || <span className="muted">—</span>}</dd>
              <dt>District</dt><dd>{user?.district || <span className="muted">—</span>}</dd>
              <dt>Village / Taluka</dt><dd>{user?.village || <span className="muted">—</span>}</dd>
              <dt>Member since</dt>
              <dd className="muted small">
                {user?.created_at ? new Date(user.created_at + "Z").toLocaleDateString("en-IN", { year: "numeric", month: "long", day: "numeric" }) : "—"}
              </dd>
            </dl>
          ) : (
            <div className="form">
              <Field label="Full name">
                <input value={form.full_name} onChange={set("full_name")} placeholder="Ramesh Patil" />
              </Field>
              <Field label="Mobile">
                <input value={form.mobile} onChange={set("mobile")} placeholder="9876543210" />
              </Field>
              <div className="row gap">
                <Field label="State">
                  <input value={form.state} onChange={set("state")} />
                </Field>
                <Field label="District">
                  <input value={form.district} onChange={set("district")} placeholder="Sangli" />
                </Field>
              </div>
              <Field label="Village / Taluka">
                <input value={form.village} onChange={set("village")} placeholder="Miraj" />
              </Field>
              <div className="row gap">
                <button disabled={busy} onClick={saveProfile}>Save</button>
                <button className="ghost" onClick={() => setEditing(false)}>Cancel</button>
              </div>
            </div>
          )}
        </Card>

        <div>
          <Card title="Change password" icon="🔑">
            {pwMsg && <Alert kind={pwMsg.kind}>{pwMsg.text}</Alert>}
            <form onSubmit={changePassword} className="form">
              <Field label="Current password">
                <input type="password" value={pwForm.old_password} onChange={setPw("old_password")}
                  autoComplete="current-password" required />
              </Field>
              <Field label="New password">
                <input type="password" value={pwForm.new_password} onChange={setPw("new_password")}
                  autoComplete="new-password" placeholder="Minimum 6 characters" required />
              </Field>
              <Field label="Confirm new password">
                <input type="password" value={pwForm.confirm} onChange={setPw("confirm")}
                  autoComplete="new-password" required />
              </Field>
              <button type="submit" disabled={pwBusy}>
                {pwBusy ? "Updating…" : "Change password"}
              </button>
            </form>
          </Card>

          <Card title="Session" icon="🔐">
            <p className="muted">Signed in as <strong>{user?.username}</strong>.</p>
            <button className="ghost" style={{ color: "var(--red)", borderColor: "var(--red)" }}
              onClick={handleLogout}>
              Sign out
            </button>
          </Card>
        </div>
      </div>
    </>
  );
}
