import { useEffect, useState, useCallback } from "react";
import { api, auth } from "./api";
import Dashboard from "./pages/Dashboard.jsx";
import FarmSoil from "./pages/FarmSoil.jsx";
import DiseaseScan from "./pages/DiseaseScan.jsx";
import GrowthMonitor from "./pages/GrowthMonitor.jsx";
import Planning from "./pages/Planning.jsx";
import Profile from "./pages/Profile.jsx";
import Login from "./pages/Login.jsx";

const TABS = [
  ["dashboard", "Dashboard", "📊"],
  ["farm", "Farm & Soil", "🌱"],
  ["disease", "Leaf Disease", "🍃"],
  ["growth", "Growth Monitor", "📈"],
  ["planning", "Planning & Cost", "🗓️"],
  ["profile", "My Profile", "👤"],
];

function readStored(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeStored(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

export default function App() {
  const [user, setUser] = useState(() => auth.getUser());
  const [tab, setTab] = useState(readStored("tab") || "dashboard");
  const [farms, setFarms] = useState([]);
  const [farmId, setFarmId] = useState(Number(readStored("farmId")) || null);
  const [health, setHealth] = useState(null);

  const loadFarms = useCallback(async () => {
    const list = await api.get("/api/farms");
    setFarms(list);
    if (list.length && !list.find((f) => f.id === farmId)) setFarmId(list[0].id);
    return list;
  }, [farmId]);

  useEffect(() => {
    if (!user) return;
    loadFarms().catch(() => {});
    const poll = () => api.get("/api/health").then(setHealth).catch(() => setHealth(null));
    poll();
    const t = setInterval(poll, 15000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => writeStored("tab", tab), [tab]);
  useEffect(() => { if (farmId) writeStored("farmId", String(farmId)); }, [farmId]);

  const handleLogin = (u) => { setUser(u); setFarms([]); setFarmId(null); };
  const handleLogout = () => { setUser(null); setFarms([]); setFarmId(null); setTab("dashboard"); };
  const handleProfileUpdate = (u) => setUser(u);

  if (!user) return <Login onLogin={handleLogin} />;

  const farm = farms.find((f) => f.id === farmId) || null;
  const props = { farm, farmId, farms, reloadFarms: loadFarms, setFarmId, goto: setTab, user, health };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">🍅</span>
          <div>
            <strong>Smart Cultivation</strong>
            <small>AI tomato farming assistant</small>
          </div>
        </div>
        <nav>
          {TABS.map(([id, label, icon]) => (
            <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
              <span>{icon}</span> {label}
            </button>
          ))}
        </nav>
        <div className="farm-picker">
          <label>Farm</label>
          <select value={farmId || ""} onChange={(e) => setFarmId(Number(e.target.value) || null)}>
            {!farms.length && <option value="">No farm yet</option>}
            {farms.map((f) => (
              <option key={f.id} value={f.id}>{f.name}</option>
            ))}
          </select>
          <button className="link" onClick={() => setTab("farm")}>+ Add / edit farm</button>
        </div>
        <div className="sidebar-user" onClick={() => setTab("profile")}>
          <span className="avatar">👤</span>
          <div>
            <strong>{user.full_name || user.username}</strong>
            <small>{user.village || user.district || "Farmer"}</small>
          </div>
        </div>
        <ModelStatus health={health} />
      </aside>
      <main className="content">
        {tab === "dashboard" && <Dashboard {...props} />}
        {tab === "farm" && <FarmSoil {...props} />}
        {tab === "disease" && <DiseaseScan {...props} />}
        {tab === "growth" && <GrowthMonitor {...props} />}
        {tab === "planning" && <Planning {...props} />}
        {tab === "profile" && (
          <Profile user={user} onUpdate={handleProfileUpdate} onLogout={handleLogout} />
        )}
      </main>
    </div>
  );
}

function ModelStatus({ health }) {
  if (!health)
    return (
      <div className="status">
        <div className="dot bad" /> Backend offline - start FastAPI on port 8000
      </div>
    );
  const items = [
    ["Leaf model (.h5)", health.disease_model.loaded, health.disease_model.exists],
    ["Soil model (XGBoost)", health.soil_model.loaded, health.soil_model.exists],
    ["RAG knowledge base", health.rag.loaded, health.rag.chroma_db],
    ["LLM key", health.rag.llm_key_set, health.rag.llm_key_set],
    ["Vision model", health.vision?.available,  health.vision?.available],
  ];
  return (
    <div className="status">
      {items.map(([label, loaded, exists]) => (
        <div key={label} title={loaded ? "loaded" : exists ? "found, loading..." : "missing"}>
          <span className={"dot " + (loaded ? "ok" : exists ? "warn" : "bad")} /> {label}
        </div>
      ))}
    </div>
  );
}
