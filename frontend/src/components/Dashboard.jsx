// frontend/src/components/Dashboard.jsx
import { useState, useEffect } from "react";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

function StatCard({ icon, label, value, color = "#1F4E79" }) {
  return (
    <div style={{
      background: "#fff", borderRadius: 10, padding: "20px 24px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.07)", textAlign: "center",
      borderTop: `4px solid ${color}`
    }}>
      <div style={{ fontSize: 32, marginBottom: 6 }}>{icon}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{value ?? "—"}</div>
      <div style={{ fontSize: 13, color: "#777", marginTop: 4 }}>{label}</div>
    </div>
  );
}

export default function Dashboard() {
  const [stats,   setStats]   = useState(null);
  const [health,  setHealth]  = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  async function fetchData() {
    setLoading(true); setError("");
    try {
      const [sRes, hRes] = await Promise.all([
        fetch(`${API}/monitor/stats`),
        fetch(`${API}/health`),
      ]);
      setStats(await sRes.json());
      setHealth(await hRes.json());
    } catch (e) {
      setError("No se pudo conectar con el backend. ¿Está corriendo?");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchData(); }, []);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <h2 style={{ margin: 0, color: "#1F4E79" }}>📊 Panel de Monitoreo</h2>
        <button onClick={fetchData} style={{
          padding: "8px 18px", borderRadius: 8, border: "1px solid #2E75B6",
          background: "#EBF3FB", color: "#1F4E79", cursor: "pointer", fontSize: 13
        }}>
          🔄 Actualizar
        </button>
      </div>

      {/* Estado del servicio */}
      {health && (
        <div style={{
          background: health.status === "ok" ? "#E8F5E9" : "#FFEBEE",
          border: `1px solid ${health.status === "ok" ? "#66BB6A" : "#EF5350"}`,
          borderRadius: 8, padding: "10px 18px", marginBottom: 24, fontSize: 14
        }}>
          {health.status === "ok" ? "✅" : "⚠️"} API: <strong>{health.status.toUpperCase()}</strong>
          &nbsp;|&nbsp; Modelo cargado: <strong>{health.model_loaded ? "Sí" : "No"}</strong>
          &nbsp;|&nbsp; Versión: <strong>{health.version}</strong>
        </div>
      )}

      {error && (
        <div style={{ background: "#FFEBEE", borderRadius: 8, padding: 16, color: "#C62828", marginBottom: 24 }}>
          ⚠️ {error}
        </div>
      )}

      {loading && <p style={{ textAlign: "center", color: "#888" }}>Cargando estadísticas...</p>}

      {stats && !stats.error && (
        <>
          {/* Tarjetas de stats */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 32 }}>
            <StatCard icon="📨" label="Total predicciones" value={stats.total_predictions} color="#1F4E79" />
            <StatCard icon="🚨" label="Spam detectado"     value={stats.spam_count}         color="#EF5350" />
            <StatCard icon="✅" label="Ham (legítimo)"     value={stats.ham_count}           color="#43A047" />
            <StatCard icon="📈" label="Tasa de spam"       value={`${(stats.spam_rate * 100).toFixed(1)}%`} color="#FB8C00" />
            <StatCard icon="🎯" label="Prob. media spam"   value={stats.avg_spam_prob?.toFixed(3)} color="#7B1FA2" />
          </div>

          {/* Actividad diaria */}
          {stats.daily_last_7?.length > 0 && (
            <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.07)" }}>
              <h3 style={{ marginTop: 0, color: "#1F4E79" }}>Actividad — últimos 7 días</h3>
              {stats.daily_last_7.map((d, i) => {
                const maxCount = Math.max(...stats.daily_last_7.map(x => x.count), 1);
                const pct = (d.count / maxCount) * 100;
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                    <span style={{ width: 95, fontSize: 13, color: "#555" }}>{d.date}</span>
                    <div style={{ flex: 1, background: "#EEE", borderRadius: 4, height: 20, overflow: "hidden" }}>
                      <div style={{ width: `${pct}%`, height: "100%", background: "#2E75B6", borderRadius: 4, transition: "width 0.4s" }} />
                    </div>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#1F4E79", width: 30 }}>{d.count}</span>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
