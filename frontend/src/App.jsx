// frontend/src/App.jsx
import { useState } from "react";
import Predictor from "./components/Predictor";
import Dashboard from "./components/Dashboard";

export default function App() {
  const [tab, setTab] = useState("predictor");

  return (
    <div style={{ fontFamily: "Segoe UI, sans-serif", minHeight: "100vh", background: "#f0f4f8" }}>
      {/* Header */}
      <header style={{
        background: "linear-gradient(135deg, #1F4E79, #2E75B6)",
        color: "#fff", padding: "18px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        boxShadow: "0 2px 8px rgba(0,0,0,0.2)"
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🛡️ Spam Detector</h1>
          <p style={{ margin: 0, fontSize: 12, opacity: 0.8 }}>ACIF104 — Aprendizaje de Máquinas</p>
        </div>
        <nav style={{ display: "flex", gap: 12 }}>
          {["predictor", "dashboard"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "8px 20px", borderRadius: 6, border: "none", cursor: "pointer",
              background: tab === t ? "#fff" : "rgba(255,255,255,0.15)",
              color: tab === t ? "#1F4E79" : "#fff", fontWeight: 600, fontSize: 14,
              transition: "all 0.2s"
            }}>
              {t === "predictor" ? "🔍 Clasificar" : "📊 Monitoreo"}
            </button>
          ))}
        </nav>
      </header>

      {/* Contenido */}
      <main style={{ maxWidth: 900, margin: "40px auto", padding: "0 24px" }}>
        {tab === "predictor" ? <Predictor /> : <Dashboard />}
      </main>
    </div>
  );
}
