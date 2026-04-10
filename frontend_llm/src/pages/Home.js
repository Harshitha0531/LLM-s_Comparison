import React, { useState } from "react";
import CountUp from "react-countup";

/* Metric Card Component */
function MetricCard({ title, value, progress }) {
  const displayValue =
    value && typeof value === "object"
      ? value.model ?? value.name ?? value.best_model ?? JSON.stringify(value)
      : value;

  return (
    <div className="p-6 rounded-xl border border-white/20 backdrop-blur-xl bg-gradient-to-br from-white/10 to-white/5 shadow-lg hover:shadow-indigo-500/30 transition relative overflow-hidden">
      {/* subtle glow */}
      <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 opacity-0 hover:opacity-100 transition"></div>

      <div className="relative z-10">
        <div className="text-sm text-gray-400 tracking-wide">{title}</div>

        <div className="text-3xl font-bold mt-2 text-white">
          {typeof value === "number" ? (
            <CountUp end={value} decimals={2} duration={1.5} />
          ) : (
            displayValue
          )}
        </div>

        {progress && typeof value === "number" && (
          <div className="w-full bg-white/10 rounded-full h-2 mt-4">
            <div
              className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all"
              style={{ width: `${value * 100}%` }}
            ></div>
          </div>
        )}
      </div>
    </div>
  );
}

/* Metric Bar Component */
function MetricBar({ label, value, max }) {
  const percentage = Math.min((value / max) * 100, 100);
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="w-full bg-white/10 rounded-full h-2">
        <div
          className="bg-gradient-to-r from-purple-500 to-indigo-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("compare_fast");
  const [hasChosenMode, setHasChosenMode] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const [showModeOptions, setShowModeOptions] = useState(false);
  const [message, setMessage] = useState("");
  const [showAbout, setShowAbout] = useState(false);

  const bestModelName =
    result && result.best_model && typeof result.best_model === "object"
      ? result.best_model.model ?? result.best_model.name
      : result?.best_model;

  const handleSubmit = async () => {
    console.log("Submit clicked", { query, file, mode });
    if (query && file) {
      setMessage("⚠ Please provide either a query OR a file, not both.");
      return;
    }

    if (!query && !file) {
      setMessage("⚠ Please enter a query or upload a file.");
      return;
    }

    setMessage("");
    setLoading(true);
    setShowModeOptions(false);
    setResult(null);
    setStreamedText("");

    try {
      let res;
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("mode", mode);
        if (query) formData.append("query", query);
        res = await fetch("http://127.0.0.1:8000/query-file", {
          method: "POST",
          body: formData,
        });
      } else {
        res = await fetch("http://127.0.0.1:8000/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, mode }),
        });
      }
      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }
      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setResult(data);
      setHasChosenMode(false); 

      const bestFromBackend = data.best_model;
      const bestModelForDisplay =
        bestFromBackend && typeof bestFromBackend === "object"
          ? bestFromBackend.model ?? bestFromBackend.name
          : bestFromBackend;

      const formatted = `Best Model: ${bestModelForDisplay}\nMode Selected: ${data.mode}\nSimilarity: ${data.model_agreement ?? "-"}`;
      setStreamedText(formatted);
    } catch (err) {
      console.error("Submit error:", err);
      setMessage(`⚠ ${err.message || "Could not reach backend. Is the server running?"}`);
      setHasChosenMode(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white px-6 py-10 font-[Inter] relative">

      {/* Glow */}
      <div className="absolute top-0 left-0 w-72 h-72 bg-purple-500 blur-3xl opacity-20 rounded-full pointer-events-none"></div>
      <div className="absolute bottom-0 right-0 w-72 h-72 bg-blue-500 blur-3xl opacity-20 rounded-full pointer-events-none"></div>

      {/* Hero */}
      <div className="text-center mb-12 relative z-10">
        <h1 className="text-6xl font-extrabold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          ModelLens
        </h1>
      </div>

  {/* About Button */}
  <div className="flex justify-center mb-6">
        <button
          onClick={() => setShowAbout(!showAbout)}
          className="px-8 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg text-xl font-semibold hover:scale-105 transition"
        >
          {showAbout ? "Hide About" : "About"}
        </button>
      </div>


{showAbout && (
  <div className="max-w-4xl mx-auto mt-6 mb-10 p-8 rounded-2xl bg-white/10 border border-white/20 backdrop-blur-xl shadow-xl animate-fade-in">

    <h2 className="text-2xl font-semibold text-indigo-300 mb-4 text-center">
      About ModelLens
    </h2>

    <p className="text-gray-300 text-center leading-relaxed max-w-2xl mx-auto">
      ModelLens is an AI-powered platform that helps you compare multiple
      Large Language Models (LLMs) using Retrieval-Augmented Generation (RAG).
      It evaluates responses based on quality, speed, and semantic relevance,
      helping you choose the most effective model for your needs.
    </p>

    <div className="grid md:grid-cols-2 gap-4 mt-8">
      {[
        {
          title: "Multi-Model Comparison",
          desc: "Compare outputs from multiple LLMs side-by-side."
        },
        {
          title: "RAG Integration",
          desc: "Use uploaded documents for context-aware answers."
        },
        {
          title: "Performance Metrics",
          desc: "Analyze latency, similarity, and response quality."
        },
        {
          title: "Smart Evaluation",
          desc: "Automatically identify the best-performing model."
        }
      ].map((item, i) => (
        <div
          key={i}
          className="p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition"
        >
          <h3 className="text-indigo-200 font-semibold">
            {item.title}
          </h3>
          <p className="text-gray-400 text-sm mt-1">
            {item.desc}
          </p>
        </div>
      ))}
    </div>
  </div>
)}

      {/* Input Card */}
      <div className="max-w-2xl mx-auto bg-white/10 p-6 rounded-2xl border border-white/20 backdrop-blur-xl shadow-xl hover:shadow-indigo-500/20 transition-all duration-300 relative z-10">

        {/* Query */}
        <input
  className="w-full p-2 rounded-lg bg-white/10 mb-3 outline-none focus:ring-2 focus:ring-indigo-400 text-center text-lg"
  placeholder="Ask something..."
  value={query}
  disabled={file !== null}   
  onChange={(e) => {
    setQuery(e.target.value);
    setMessage(""); 

    if (file) {
      setFile(null);
      setMessage("📄 File removed. Using query mode.");
    }
  }}
/>

        {/* File Upload */}
        <label className="block mb-3 p-2 border-2 border-dashed border-indigo-400 text-center rounded-lg cursor-pointer hover:bg-indigo-500/10 text-lg">
          📂 {file ? file.name : "Upload file (PDF/TXT)"}
          <input
  type="file"
  hidden
  disabled={query !== ""}   
  onChange={(e) => {
    setFile(e.target.files[0]);
    setMessage(""); 

    if (query) {
      setQuery("");
      setMessage("✏ Query cleared. Using document mode.");
    }
  }}
/>
        </label>

   {/* Mode */}
<button
  onClick={() => setShowModeOptions(!showModeOptions)}   
  className="w-full py-2 text-xl bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg font-semibold"
>
  {hasChosenMode ? mode.replace("_", " ").toUpperCase() : "Choose Mode"}
</button>

{showModeOptions && (
  <div className="mt-2 grid grid-cols-2 gap-2">
    {["compare_fast", "compare_full", "cascade_fast", "cascade_full"].map((m) => (
      <button
        key={m}
        onClick={() => {
          setMode(m);
          setHasChosenMode(true);
          setShowModeOptions(false);
          setMessage(`✅ ${m.replace("_", " ").toUpperCase()} selected`);

          
          setTimeout(() => setMessage(""), 2000);
        }}
        className={`py-2 text-sm rounded-lg ${
          mode === m ? "bg-indigo-500" : "bg-white/10"
        }`}
      >
        {m.replace("_", " ").toUpperCase()}
      </button>
    ))}
  </div>
)}

        {/* Message */}
        {message && (
          <div className="mt-3 p-2 text-center text-sm rounded bg-indigo-500/20 border border-indigo-400">
            {message}
          </div>
        )}

        {/* Submit */}
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!query && !file}
          className={`w-full py-2 mt-3 text-xl rounded-lg font-semibold ${
            !query && !file
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-gradient-to-r from-indigo-500 to-purple-500"
          }`}
        >
          {loading ? (
            <span className="inline-flex items-center justify-center gap-2">
              <span className="animate-spin h-5 w-5 rounded-full border-2 border-white/40 border-t-transparent" />
              Loading...
            </span>
          ) : (
            "Submit"
          )}
        </button>
      </div>

      {/* Streamed Text */}
  
{streamedText && (
  <div className="mt-6 max-w-3xl mx-auto p-6 bg-black/40 rounded-xl text-lg space-y-2">
    {streamedText.split("\n").map((line, idx) => (
      <div key={idx} className="font-semibold text-indigo-300">
        {line}
      </div>
    ))}
  </div>
)}
{/* Dashboard */}
{result && result.results && (
  <div className="mt-12 max-w-6xl mx-auto relative z-10">

    {/* Metrics Dashboard */}
    <div className="grid md:grid-cols-3 gap-6 mb-10">
      <MetricCard
        title="Best Model"
        value={bestModelName}
      />
      <MetricCard title="Model Agreement" value={result.model_agreement ?? 0} progress />
      <MetricCard title="Mode" value={result.mode} />
    </div>

    {/* Model Comparison */}
    <div className="grid md:grid-cols-2 gap-6">
      {result.results.map((r, i) => (
        <div
          key={i}
          className={`p-6 rounded-2xl border backdrop-blur-xl transition relative ${
            r.model === bestModelName
              ? "border-green-400 shadow-lg shadow-green-400/30"
              : "border-white/20"
          }`}
        >
          {r.model === bestModelName && (
            <span className="absolute top-3 right-3 bg-gradient-to-r from-green-400 to-emerald-500 text-black px-3 py-1 rounded-full text-xs font-semibold">
              Best Model
            </span>
          )}

          {/* Model Name */}
          <h2 className="text-xl font-bold mb-2">{r.model}</h2>

          {/* Response */}
          <div className="mt-2 bg-black/30 p-3 rounded-lg max-h-56 overflow-y-auto">
            <p className="text-gray-200 leading-relaxed whitespace-pre-wrap text-sm">
              {r.response}
            </p>
          </div>

          {/* Metrics */}
          <div className="mt-4 text-sm text-gray-300 space-y-3">

            {/* Latency */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Latency</span>
                <span>{r.latency}s</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full"
                  style={{ width: `${Math.min((r.latency / 5) * 100, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Relevance */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Relevance</span>
                <span>{typeof r.relevance === "number" ? r.relevance.toFixed(2) : r.relevance}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-green-400"
                  style={{ width: `${Math.min((Number(r.relevance) || 0) * 100, 100)}%` }}
                ></div>
              </div>
            </div>

            <div className="text-xs text-gray-400">
              Tokens: {r.token_usage_estimate ?? "-"} | Truncated: {r.truncation_detected ? "Yes" : "No"}
            </div>

          </div>
        </div>
      ))}
    </div>

  </div>
)}

    </div>
  );
}
