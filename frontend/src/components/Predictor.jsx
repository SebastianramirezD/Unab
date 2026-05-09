// frontend/src/components/Predictor.jsx
import { useState } from "react";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

const EJEMPLOS = [
  "Congratulations! You've won a FREE iPhone. Click now to claim your prize!",
  "Hey, are you coming to the meeting tomorrow at 10am?",
  "URGENT: Your account will be suspended. Verify now: http://bit.ly/fake",
  "Thanks for dinner last night, it was really fun!",
];

function Badge({ label, prob }) {
  const isSpam = label === "spam";
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 10,
      padding: "10px 22px", borderRadius: 30,
      background: isSpam ? "#FFEBEE" : "#E8F5E9",
      border: `2px solid ${isSpam ? "#EF5350" : "#66BB6A"}`,
    }}>
      <span style={{ fontSize: 28 }}>{isSpam ? "🚨" : "✅"}</span>
      <div>
        <div style={{ fontWeight: 700, fontSize: 18, color: isSpam ? "#C62828" : "#2E7D32" }}>
          {isSpam ? "SPAM" : "HAM"}
        </div>
        <div style={{ fontSize: 13, color: "#555" }}>
          Probabilidad spam: <strong>{(prob * 100).toFixed(1)}%</strong>
        </div>
      </div>
    </div>
  );
}

function ShapBar({ feature, value }) {
  const isSpam = value > 0;
  const pct    = Math.min(Math.abs(value) * 500, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
      <span style={{ width: 140, fontSize: 13, textAlign: "right", color: "#333", fontFamily: "monospace" }}>
        {feature}
      </span>
      <div style={{ flex: 1, background: "#eee", borderRadius: 4, height: 16, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 4,
          background: isSpam ? "#EF5350" : "#42A5F5",
          transition: "width 0.4s"
        }} />
      </div>
      <span style={{ fontSize: 12, color: isSpam ? "#C62828" : "#1565C0", width: 55 }}>
        {value > 0 ? "+" : ""}{value.toFixed(4)}
      </span>
    </div>
  );
}

export default function Predictor() {
  const [text,    setText]    = useState("");
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");
  const [explain, setExplain] = useState(false);

  async function classify() {
    if (!text.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const endpoint = explain ? "/explain" : "/predict";
      const res  = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(`No se pudo conectar con la API: ${e.message}. ¿Está corriendo el backend?`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div style={{ background: "#fff", borderRadius: 12, padding: 32, boxShadow: "0 2px 12px rgba(0,0,0,0.08)" }}>
        <h2 style={{ marginTop: 0, color: "#1F4E79" }}>🔍 Clasificación de Mensajes</h2>

        {/* Ejemplos rápidos */}
        <div style={{ marginBottom: 16 }}>
          <p style={{ fontSize: 13, color: "#666", marginBottom: 8 }}>Ejemplos rápidos:</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {EJEMPLOS.map((e, i) => (
              <button key={i} onClick={() => setText(e)} style={{
                padding: "5px 12px", borderRadius: 20, border: "1px solid #2E75B6",
                background: "#EBF3FB", color: "#1F4E79", fontSize: 12, cursor: "pointer"
              }}>
                Ejemplo {i + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Textarea */}
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Escribe o pega aquí el mensaje a clasificar..."
          rows={5}
          style={{
            width: "100%", padding: 14, borderRadius: 8, fontSize: 15,
            border: "1.5px solid #ccc", resize: "vertical", boxSizing: "border-box",
            fontFamily: "inherit", outline: "none",
          }}
        />

        {/* Opciones */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 14 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, cursor: "pointer" }}>
            <input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} />
            Incluir explicación SHAP
          </label>
          <button onClick={classify} disabled={loading || !text.trim()} style={{
            padding: "10px 28px", borderRadius: 8, border: "none",
            background: loading ? "#aaa" : "#1F4E79", color: "#fff",
            fontWeight: 700, fontSize: 15, cursor: loading ? "not-allowed" : "pointer",
          }}>
            {loading ? "Clasificando..." : "Clasificar"}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: "#FFEBEE", border: "1px solid #EF5350", borderRadius: 8, padding: 16, marginTop: 20, color: "#C62828" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Resultado */}
      {result && (
        <div style={{ background: "#fff", borderRadius: 12, padding: 32, marginTop: 24, boxShadow: "0 2px 12px rgba(0,0,0,0.08)" }}>
          <h3 style={{ marginTop: 0, color: "#1F4E79" }}>Resultado</h3>
          <Badge label={result.label} prob={result.probability} />
          <div style={{ display: "flex", gap: 24, marginTop: 16, fontSize: 14, color: "#555" }}>
            <span>🎯 Confianza: <strong>{result.confidence}</strong></span>
            <span>⚡ Latencia: <strong>{result.latency_ms} ms</strong></span>
          </div>

          {/* SHAP */}
          {result.top_features && result.top_features.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <h4 style={{ color: "#1F4E79", marginBottom: 12 }}>
                🧠 Explicación SHAP — Características más influyentes
              </h4>
              <p style={{ fontSize: 13, color: "#777", marginBottom: 14 }}>
                <span style={{ color: "#EF5350" }}>■</span> Rojo: contribuye a clasificar como spam  &nbsp;
                <span style={{ color: "#42A5F5" }}>■</span> Azul: contribuye a clasificar como ham
              </p>
              {result.top_features.map((f, i) => (
                <ShapBar key={i} feature={f.feature} value={f.shap_value} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
