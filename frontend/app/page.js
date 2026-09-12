"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"];

function formatTick(dt) {
  const [datePart, timePart] = dt.split(" ");
  const [, month, day] = datePart.split("-");
  const [hh, mm] = timePart.split(":");
  return `${month}/${day} ${hh}:${mm}`;
}

function ThemeToggle() {
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    const stored = localStorage.getItem("theme") || "light";
    setTheme(stored);
    document.documentElement.dataset.theme = stored;
  }, []);

  const toggle = () => {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
  };

  return (
    <button
      onClick={toggle}
      aria-label="Toggle dark mode"
      className="fixed top-5 right-6 p-2 rounded-full border border-hairline bg-surface hover:opacity-80 transition-opacity"
    >
      {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
    </button>
  );
}

function SignalCard({ data }) {
  const isBuy = data.signal === "BUY";
  const color = isBuy ? "var(--accent-buy)" : "var(--accent-sell)";

  return (
    <div className="bg-surface rounded-2xl border border-hairline px-8 py-6 text-center">
      <p className="font-mono text-xs uppercase tracking-wide opacity-60">
        {data.ticker}
      </p>
      <p className="font-display text-3xl font-semibold mt-1" style={{ color }}>
        {data.signal}
      </p>
      <p className="font-mono text-sm mt-2 opacity-80">
        Confidence: {(data.confidence * 100).toFixed(1)}%
      </p>
      <p className="font-mono text-sm mt-1 opacity-80">
        Price: ${data.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </p>

      {!data.reliable && (
        <p className="font-mono text-xs mt-4 inline-block bg-accent-neutral/15 text-accent-neutral px-3 py-1 rounded-full">
          Model trained on BTC-USD only — less reliable for this asset
        </p>
      )}
    </div>
  );
}

function PriceChart({ history }) {
  return (
    <div className="bg-surface rounded-2xl border border-hairline px-6 py-6">
      <p className="font-display text-sm font-semibold mb-4">Price</p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={history}>
          <CartesianGrid stroke="var(--hairline)" vertical={false} />
          <XAxis
            dataKey="Datetime"
            tickFormatter={formatTick}
            tick={{ fontSize: 11, fill: "var(--accent-neutral)" }}
            minTickGap={40}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "var(--accent-neutral)" }}
            domain={["auto", "auto"]}
            width={60}
          />
          <Tooltip
            labelFormatter={formatTick}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--hairline)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
          />
          <Line type="monotone" dataKey="Close" stroke="var(--foreground)" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="SMA_20" stroke="var(--accent-buy)" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
          <Line type="monotone" dataKey="BBU" stroke="var(--accent-neutral)" dot={false} strokeWidth={1} strokeDasharray="2 3" />
          <Line type="monotone" dataKey="BBL" stroke="var(--accent-neutral)" dot={false} strokeWidth={1} strokeDasharray="2 3" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function RsiChart({ history }) {
  return (
    <div className="bg-surface rounded-2xl border border-hairline px-6 py-6">
      <p className="font-display text-sm font-semibold mb-4">RSI (14)</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={history}>
          <CartesianGrid stroke="var(--hairline)" vertical={false} />
          <XAxis
            dataKey="Datetime"
            tickFormatter={formatTick}
            tick={{ fontSize: 11, fill: "var(--accent-neutral)" }}
            minTickGap={40}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: "var(--accent-neutral)" }}
            width={40}
          />
          <ReferenceLine y={70} stroke="var(--accent-sell)" strokeDasharray="3 3" />
          <ReferenceLine y={30} stroke="var(--accent-buy)" strokeDasharray="3 3" />
          <Tooltip
            labelFormatter={formatTick}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--hairline)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
          />
          <Line type="monotone" dataKey="RSI_14" stroke="var(--foreground)" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Home() {
  const [ticker, setTicker] = useState("BTC-USD");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSignal = async (symbol) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict?ticker=${symbol}`);
      if (!res.ok) throw new Error(`API responded with ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignal(ticker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="flex-1 flex flex-col items-center px-6 py-16 md:py-20">
      <ThemeToggle />

      <div className="max-w-xl w-full text-center">
        <h1 className="font-display text-3xl md:text-4xl font-semibold">
          Market Brief
        </h1>
        <p className="font-display text-sm mt-3 opacity-70">
          Live technical-indicator signals from a 24-feature XGBoost model.
        </p>
      </div>

      <div className="flex items-center gap-3 mt-8">
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="font-mono text-sm bg-surface border border-hairline rounded-full px-4 py-2"
        >
          {TICKERS.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <button
          onClick={() => fetchSignal(ticker)}
          disabled={loading}
          className="font-mono text-sm bg-foreground text-background rounded-full px-5 py-2 disabled:opacity-50"
        >
          {loading ? "Loading..." : "Get Signal"}
        </button>
      </div>

      {error && (
        <p className="font-mono text-sm text-accent-sell mt-6">{error}</p>
      )}

      {data && (
        <div className="w-full max-w-2xl flex flex-col gap-6 mt-10">
          <SignalCard data={data} />
          <PriceChart history={data.history} />
          <RsiChart history={data.history} />
        </div>
      )}
    </main>
  );
}