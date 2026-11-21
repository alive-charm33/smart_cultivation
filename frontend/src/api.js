// Small fetch wrapper. In dev Vite proxies /api and /media to FastAPI (see vite.config.js).
const BASE = import.meta.env.VITE_API_BASE || "";

// ------------------------------------------------------------------ token store
export const auth = {
  getToken: () => { try { return localStorage.getItem("sc_token"); } catch { return null; } },
  setToken: (t) => { try { localStorage.setItem("sc_token", t); } catch {} },
  clearToken: () => { try { localStorage.removeItem("sc_token"); localStorage.removeItem("sc_user"); } catch {} },
  getUser: () => { try { const u = localStorage.getItem("sc_user"); return u ? JSON.parse(u) : null; } catch { return null; } },
  setUser: (u) => { try { localStorage.setItem("sc_user", JSON.stringify(u)); } catch {} },
};

async function handle(res) {
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

function authHeaders() {
  const t = auth.getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export const api = {
  get: (path) => fetch(BASE + path, { headers: authHeaders() }).then(handle),
  post: (path, body) =>
    fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(body),
    }).then(handle),
  put: (path, body) =>
    fetch(BASE + path, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(body),
    }).then(handle),
  upload: (path, fields) => {
    const fd = new FormData();
    Object.entries(fields).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") fd.append(k, v);
    });
    return fetch(BASE + path, { method: "POST", headers: authHeaders(), body: fd }).then(handle);
  },
};

export const media = (url) => (url ? BASE + url : null);

export const STAGES = [
  ["nursery", "Nursery / seedling"],
  ["transplanting", "Transplanting"],
  ["vegetative", "Vegetative"],
  ["flowering_fruit_set", "Flowering & fruit set"],
  ["fruit_development", "Fruit development"],
  ["ripening_harvest", "Ripening & harvest"],
  ["end_of_season", "End of season"],
];
export const stageName = (id) => (STAGES.find((s) => s[0] === id) || [id, id])[1];

export const DISEASE_NAMES = {
  bacterial_spot: "Bacterial spot",
  early_blight: "Early blight",
  late_blight: "Late blight",
  leaf_mold: "Leaf mold",
  septoria_leaf_spot: "Septoria leaf spot",
  spider_mites: "Spider mites",
  target_spot: "Target spot",
  tylcv: "Yellow leaf curl virus",
  tomato_mosaic_virus: "Mosaic virus",
  healthy: "Healthy",
};
export const diseaseName = (id) => DISEASE_NAMES[id] || id;

export const rupees = (n) =>
  n === null || n === undefined ? "-" : "₹" + Math.round(n).toLocaleString("en-IN");
