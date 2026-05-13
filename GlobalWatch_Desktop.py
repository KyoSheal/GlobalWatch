#!/usr/bin/env python3
"""
GlobalWatch Desktop v5
Pure tkinter — sidebar nav + frame stack.
No customtkinter, no ttk.Notebook (both unreliable on macOS system Tk).
"""
from __future__ import annotations
import json, os, queue, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
CONFIG_PATH   = ROOT / "paper_config.json"
SNAPSHOT_PATH = ROOT / "outputs" / "snapshot_live.json"
STATE_PATH    = ROOT / "outputs" / "state" / "risk_profile_state.json"
ATTR_PATH     = ROOT / "outputs" / "factor_attribution.jsonl"
SCORE_PATH    = ROOT / "outputs" / "scoreboard.jsonl"

# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(
    bg="#111827", side="#0d1424", card="#1e2a3a", entry="#0a1120",
    green="#00c87a", red="#ff4d4d", yellow="#f5a623", blue="#4a9eff",
    gray="#6b7280", fg="#e0e8f0", fg2="#9ca3af", sel="#1d4ed8",
)

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_cfg() -> dict:
    try:    return json.loads(CONFIG_PATH.read_text())
    except: return {}

def save_cfg(d: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(d, indent=2))

def load_snap() -> dict:
    try:    return json.loads(SNAPSHOT_PATH.read_text())
    except: return {}

def load_jsonl(p: Path, n: int = 200) -> list[dict]:
    rows: list[dict] = []
    try:
        for ln in p.read_text().strip().splitlines()[-n:]:
            try: rows.append(json.loads(ln))
            except: pass
    except: pass
    return rows

def trade_file() -> Path | None:
    hits = sorted((ROOT/"outputs").glob("????-??/*/trade_history.jsonl"),
                  key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return hits[-1] if hits else None

def usd(v, d=2) -> str:
    try:    return f"${float(v):,.{d}f}"
    except: return "—"

def pct(v) -> str:
    try:
        x = float(v)
        return f"{'+' if x>=0 else ''}{x:.2f}%"
    except: return "—"

def col(v) -> str:
    try:    return C["green"] if float(v) >= 0 else C["red"]
    except: return C["fg"]

def tb_set(t: tk.Text, s: str) -> None:
    t.configure(state="normal")
    t.delete("1.0","end")
    t.insert("1.0", s)
    t.configure(state="disabled")

# ── Widget helpers ─────────────────────────────────────────────────────────────
MONO = ("Menlo",11) if sys.platform=="darwin" else ("Courier",11)
SAN  = ("Helvetica",11)
SAN_B= ("Helvetica",11,"bold")
SAN_L= ("Helvetica",14,"bold")
SAN_S= ("Helvetica",10)

def Text(parent, **kw) -> tk.Text:
    defaults = dict(bg=C["entry"],fg=C["fg"],insertbackground=C["fg"],
                    font=MONO,bd=0,relief="flat",state="disabled",
                    selectbackground=C["sel"])
    defaults.update(kw)
    return tk.Text(parent, **defaults)

def Label(parent, text="", big=False, dim=False, **kw) -> tk.Label:
    fg = C["fg2"] if dim else C["fg"]
    fn = SAN_L if big else SAN
    return tk.Label(parent, text=text, bg=C["card"], fg=fg, font=fn, **kw)

def Btn(parent, text, cmd, green=False, red=False, width=None, **kw) -> tk.Button:
    bg = C["green"] if green else (C["red"] if red else "#1e3a5f")
    fg = "black" if green else C["fg"]
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=bg, activeforeground=fg,
                  font=SAN_B, bd=0, padx=10, pady=6, cursor="hand2", **kw)
    if width:
        b.configure(width=width)
    return b

def Sep(parent) -> tk.Frame:
    return tk.Frame(parent, bg="#2a3a50", height=1)

def scrolled_text(parent, **kw) -> tuple[tk.Frame, tk.Text]:
    """Returns (frame, text_widget) with scrollbars."""
    f   = tk.Frame(parent, bg=C["entry"])
    vsb = tk.Scrollbar(f, bg=C["card"], troughcolor=C["entry"],
                       relief="flat", width=10)
    t   = Text(f, **kw)
    vsb.configure(command=t.yview)
    t.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    t.pack(side="left", fill="both", expand=True)
    return f, t

# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):

    NAV = ["Dashboard","Analytics","Trading","Trades","News & AI","Macro","Settings"]

    def __init__(self):
        super().__init__()
        self.title("GlobalWatch")
        self.geometry("1360x820")
        self.minsize(1000,600)
        self.configure(bg=C["bg"])
        self.protocol("WM_DELETE_WINDOW", self._close)

        # state
        self._tproc: subprocess.Popen|None = None
        self._nproc: subprocess.Popen|None = None
        self._tq:    queue.Queue = queue.Queue()
        self._nq:    queue.Queue = queue.Queue()
        self._alive  = True
        self._eq_fig = self._eq_cv = None

        self._build()
        self._show("Dashboard")
        self._drain()
        threading.Thread(target=self._poll, daemon=True).start()

    # ── Shell ──────────────────────────────────────────────────────────────────
    def _build(self):
        # sidebar
        side = tk.Frame(self, bg=C["side"], width=140)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="GlobalWatch", bg=C["side"], fg=C["fg"],
                 font=("Helvetica",13,"bold"),wraplength=130).pack(pady=(18,16))

        self._nav_btns: dict[str,tk.Button] = {}
        for name in self.NAV:
            b = tk.Button(
                side, text=name, bg=C["side"], fg=C["fg2"],
                activebackground=C["card"], activeforeground=C["fg"],
                font=SAN, bd=0, padx=0, pady=8, cursor="hand2",
                anchor="center", width=16,
                command=lambda n=name: self._show(n))
            b.pack(fill="x", padx=4, pady=1)
            self._nav_btns[name] = b

        Sep(side).pack(fill="x", padx=8, pady=8)
        self._status_lbl = tk.Label(side, text="Idle", bg=C["side"], fg=C["gray"],
                                     font=SAN_S, wraplength=130)
        self._status_lbl.pack(padx=6)

        # content area
        self._area = tk.Frame(self, bg=C["bg"])
        self._area.pack(side="left", fill="both", expand=True)

        # build all pages
        self._pages: dict[str,tk.Frame] = {}
        builders = [
            ("Dashboard",  self._page_dashboard),
            ("Analytics",  self._page_analytics),
            ("Trading",    self._page_trading),
            ("Trades",     self._page_trades),
            ("News & AI",  self._page_news),
            ("Macro",      self._page_macro),
            ("Settings",   self._page_settings),
        ]
        for name, builder in builders:
            f = tk.Frame(self._area, bg=C["bg"])
            self._pages[name] = f
            builder(f)

    def _show(self, name: str):
        for n, f in self._pages.items():
            f.place_forget()
        self._pages[name].place(relx=0, rely=0, relwidth=1, relheight=1)
        for n, b in self._nav_btns.items():
            b.configure(bg=C["card"] if n==name else C["side"],
                        fg=C["fg"]  if n==name else C["fg2"])

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    def _page_dashboard(self, p: tk.Frame):
        p.configure(bg=C["bg"])

        # ── Stat row ──
        row = tk.Frame(p, bg=C["side"])
        row.pack(fill="x", padx=8, pady=(8,4))
        self._dc: dict[str,tk.Label] = {}
        for key,title in [("equity","Total Equity"),("ret","Return"),
                           ("cash","Cash"),("dd","Drawdown"),
                           ("sortino","Sortino"),("calmar","Calmar"),
                           ("winrate","Win Rate"),("cycle","Cycle")]:
            c = tk.Frame(row, bg=C["card"], bd=0)
            c.pack(side="left", expand=True, fill="x", padx=3, pady=6)
            tk.Label(c, text=title, bg=C["card"], fg=C["fg2"], font=SAN_S).pack(pady=(6,0), padx=8)
            vl = tk.Label(c, text="—", bg=C["card"], fg=C["fg"], font=SAN_L)
            vl.pack(pady=(1,6), padx=8)
            self._dc[key] = vl

        # ── Status strip ──
        self._mstrip = tk.Label(p, text="Market: —", bg=C["side"], fg=C["fg2"],
                                 font=SAN_S, anchor="w")
        self._mstrip.pack(fill="x", padx=8, pady=(0,4))

        # ── Holdings + chart ──
        mid = tk.Frame(p, bg=C["bg"])
        mid.pack(fill="both", expand=True, padx=8, pady=(0,4))

        lf = tk.Frame(mid, bg=C["card"])
        lf.pack(side="left", fill="both", expand=True, padx=(0,4))
        tk.Label(lf, text="Holdings", bg=C["card"], fg=C["fg"], font=SAN_B).pack(
            anchor="w", padx=10, pady=(8,2))
        hf, self._hold_t = scrolled_text(lf)
        hf.pack(fill="both", expand=True, padx=8, pady=(0,8))

        rf = tk.Frame(mid, bg=C["card"])
        rf.pack(side="left", fill="both", expand=True)
        tk.Label(rf, text="Equity Curve", bg=C["card"], fg=C["fg"], font=SAN_B).pack(
            anchor="w", padx=10, pady=(8,2))
        self._chart_f = tk.Frame(rf, bg=C["entry"])
        self._chart_f.pack(fill="both", expand=True, padx=8, pady=(0,8))

        self._upd = tk.Label(p, text="", bg=C["bg"], fg=C["gray"], font=SAN_S, anchor="e")
        self._upd.pack(fill="x", padx=10, pady=(0,4))

    def _upd_dashboard(self, s: dict):
        eq   = float(s.get("total_equity",0) or 0)
        cash = float(s.get("cash",0) or 0)
        init = float(s.get("initial_cash", s.get("initial_cash_usd",80000)) or 80000)
        rraw = s.get("return")
        rpct = float(rraw)*100 if rraw is not None else ((eq-init)/init*100 if init else 0)
        ddraw= float(s.get("drawdown", s.get("current_drawdown",0)) or 0)
        ddpct= ddraw*100 if abs(ddraw)<=1 else ddraw
        perf = s.get("enhanced_performance", s.get("performance_metrics",{})) or {}
        so   = perf.get("sortino_ratio", perf.get("sortino"))
        ca   = perf.get("calmar_ratio",  perf.get("calmar"))
        wr   = perf.get("win_rate")

        self._dc["equity"].configure(text=usd(eq))
        self._dc["ret"].configure(text=pct(rpct), fg=col(rpct))
        self._dc["cash"].configure(text=usd(cash))
        self._dc["dd"].configure(text=pct(ddpct), fg=C["red"] if ddpct>5 else C["fg"])
        self._dc["sortino"].configure(
            text=f"{float(so):.2f}" if so is not None else "—",
            fg=C["green"] if so is not None and float(so)>0 else C["gray"])
        self._dc["calmar"].configure(text=f"{float(ca):.2f}" if ca is not None else "—")
        self._dc["winrate"].configure(text=f"{float(wr)*100:.0f}%" if wr is not None else "—")
        self._dc["cycle"].configure(text=str(s.get("cycle","—")))

        ms   = s.get("market_session",{}) or {}
        st   = str(ms.get("state","—")).upper()
        et   = str(ms.get("now_et",""))[:19].replace("T"," ")
        mc   = C["green"] if st=="OPEN" else (C["yellow"] if "PRE" in st or "AFTER" in st else C["gray"])
        prof = str(s.get("active_risk_profile","—")).upper()
        self._mstrip.configure(
            text=f"  Market: {st}   ET: {et}   Risk: {prof}   Status: {s.get('status','—')}",
            fg=mc)

        pd  = s.get("positions_detail",{}) or {}
        pos = s.get("positions",{}) or {}
        lines = [f"{'Ticker':<8} {'Qty':>6}  {'Price':>10}  {'Value':>10}  {'Wt':>6}", "─"*46]
        for tk2, info in pd.items():
            wt = pos.get(tk2)
            lines.append(
                f"{tk2:<8} {float(info.get('quantity',0)):>6.0f}  "
                f"{usd(info.get('price',0)):>10}  {usd(info.get('value',0)):>10}  "
                f"{float(wt)*100:.1f}%" if wt is not None else "  —")
        if not pd:
            lines.append("  No positions.")
        tb_set(self._hold_t, "\n".join(lines))

        hist = s.get("equity_history",[]) or []
        if len(hist) >= 2:
            self._draw_curve(hist, init)
        self._upd.configure(text=f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    def _draw_curve(self, hist, initial):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib.ticker as mticker
        except ImportError:
            return
        vals = [float(h.get("equity", h.get("total_equity", initial))) for h in hist]
        if len(vals) < 2: return
        if self._eq_cv is None:
            self._eq_fig = Figure(figsize=(4,2.5), facecolor=C["entry"], tight_layout=True)
            self._eq_cv  = FigureCanvasTkAgg(self._eq_fig, master=self._chart_f)
            self._eq_cv.get_tk_widget().pack(fill="both", expand=True)
        self._eq_fig.clear()
        ax = self._eq_fig.add_subplot(111, facecolor="#0f1e32")
        ax.tick_params(colors=C["fg2"], labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3a50")
        clr = C["green"] if vals[-1]>=initial else C["red"]
        ax.plot(vals, color=clr, linewidth=1.8)
        ax.fill_between(range(len(vals)), vals, initial, alpha=0.15, color=clr)
        ax.axhline(initial, color="#334466", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.set_ylabel("$", color=C["fg2"], fontsize=8)
        self._eq_cv.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    def _page_analytics(self, p: tk.Frame):
        p.configure(bg=C["bg"])
        top = tk.Frame(p, bg=C["bg"])
        top.pack(fill="x", padx=8, pady=(8,4))

        pf = tk.Frame(top, bg=C["card"]); pf.pack(side="left", fill="both", expand=True, padx=(0,4))
        tk.Label(pf, text="Performance Metrics", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=10, pady=(8,2))
        _, self._perf_t = scrolled_text(pf, height=10)
        _.pack(fill="x", padx=8, pady=(0,8))

        rf = tk.Frame(top, bg=C["card"]); rf.pack(side="left", fill="both", expand=True)
        tk.Label(rf, text="Risk Diagnostics", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=10, pady=(8,2))
        _, self._risk_t = scrolled_text(rf, height=10)
        _.pack(fill="x", padx=8, pady=(0,8))

        bot = tk.Frame(p, bg=C["bg"])
        bot.pack(fill="both", expand=True, padx=8, pady=(0,8))

        af = tk.Frame(bot, bg=C["card"]); af.pack(side="left", fill="both", expand=True, padx=(0,4))
        ah = tk.Frame(af, bg=C["card"]); ah.pack(fill="x", padx=10, pady=(8,2))
        tk.Label(ah, text="Factor Attribution", bg=C["card"], fg=C["fg"], font=SAN_B).pack(side="left")
        Btn(ah, "Refresh", self._do_attr).pack(side="right")
        atf, self._attr_t = scrolled_text(af)
        atf.pack(fill="both", expand=True, padx=8, pady=(0,8))

        cbf = tk.Frame(bot, bg=C["card"]); cbf.pack(side="left", fill="both", expand=True)
        tk.Label(cbf, text="Circuit Breaker", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=10, pady=(8,2))
        cbff, self._cb_t = scrolled_text(cbf)
        cbff.pack(fill="both", expand=True, padx=8, pady=(0,8))

    def _upd_analytics(self, s: dict):
        perf = s.get("enhanced_performance", s.get("performance_metrics",{})) or {}
        lines = []
        for k,lbl in [("sortino_ratio","Sortino Ratio        "),("calmar_ratio","Calmar Ratio         "),
                      ("win_rate","Win Rate             "),("avg_win","Avg Win              "),
                      ("avg_loss","Avg Loss             "),("win_loss_ratio","Win/Loss             "),
                      ("max_consecutive_losses","Max Consec Losses    ")]:
            v = perf.get(k, s.get(k))
            if v is None: continue
            try:
                fv=float(v)
                if k=="win_rate": lines.append(f"  {lbl} {fv*100:.1f}%")
                elif k in("avg_win","avg_loss"): lines.append(f"  {lbl} {fv*100:+.2f}%")
                else: lines.append(f"  {lbl} {fv:.3f}")
            except: lines.append(f"  {lbl} {v}")
        tb_set(self._perf_t, "\n".join(lines) or "  No data.")

        cov = s.get("cov_risk_diag",{}) or {}
        rlines=[]
        for k,lbl in [("realized_vol","Realised Vol   "),("target_vol","Target Vol     "),
                      ("vol_scale_factor","Vol Scale      "),("crisis_mode","Crisis Mode    "),
                      ("max_corr","Max Corr       "),("avg_corr","Avg Corr       ")]:
            v = cov.get(k, s.get(k))
            if v is None: continue
            try: rlines.append(f"  {lbl} {float(v):.4f}")
            except: rlines.append(f"  {lbl} {v}")
        tb_set(self._risk_t, "\n".join(rlines) or "  No data.")

        cb = s.get("circuit_breaker", s.get("cb_state",{})) or {}
        clines=[]
        for k,lbl in [("triggered","Triggered      "),("rolling_triggered","Rolling CB     "),
                      ("cumulative_dd","Cumulative DD  "),("rolling_dd","Rolling DD     "),
                      ("recovery_mode","Recovery Mode  ")]:
            v=cb.get(k)
            if v is None: continue
            try: clines.append(f"  {lbl} {float(v):.4f}" if isinstance(v,float) else f"  {lbl} {v}")
            except: clines.append(f"  {lbl} {v}")
        tb_set(self._cb_t, "\n".join(clines) or "  No data.")

    def _do_attr(self):
        rows = load_jsonl(ATTR_PATH, 50)
        if not rows: tb_set(self._attr_t,"  No attribution data."); return
        hdr = f"  {'Cycle':>6}  {'Momentum':>9}  {'Sharpe':>9}  {'News':>9}  {'Vol':>9}"
        sep = "  " + "─"*52
        def f(v):
            try: return f"{float(v):+.3f}"
            except: return str(v) if v else "—"
        lines=[hdr,sep]+[
            f"  {str(r.get('cycle','?')):>6}  {f(r.get('avg_momentum_z')):>9}  "
            f"{f(r.get('avg_sharpe_z')):>9}  {f(r.get('avg_news_contrib')):>9}  "
            f"{f(r.get('avg_vol_score')):>9}" for r in rows[-20:]]
        tb_set(self._attr_t, "\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: TRADING
    # ══════════════════════════════════════════════════════════════════════════
    def _page_trading(self, p: tk.Frame):
        p.configure(bg=C["bg"])
        lf = tk.Frame(p, bg=C["card"], width=300)
        lf.pack(side="left", fill="y", padx=(8,4), pady=8)
        lf.pack_propagate(False)

        self._tbtn = Btn(lf, "  Start Trading  ", self._tog_trade, green=True)
        self._tbtn.pack(fill="x", padx=14, pady=(14,4))
        self._tstat = tk.Label(lf, text="Status: Stopped", bg=C["card"], fg=C["gray"], font=SAN)
        self._tstat.pack(pady=(0,8))

        Sep(lf).pack(fill="x", padx=14, pady=4)
        tk.Label(lf, text="Quick Settings", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=14, pady=(8,4))

        tk.Label(lf, text="Risk Profile", bg=C["card"], fg=C["fg2"], font=SAN).pack(anchor="w", padx=14)
        self._rp = tk.StringVar(value=load_cfg().get("active_risk_profile","mid"))
        om = tk.OptionMenu(lf, self._rp, "low","mid","high")
        om.configure(bg=C["entry"],fg=C["fg"],activebackground=C["sel"],bd=0,
                     font=SAN, highlightthickness=0)
        om["menu"].configure(bg=C["entry"],fg=C["fg"])
        om.pack(anchor="w", padx=14, pady=(2,8))

        tk.Label(lf, text="Rebalance (min)", bg=C["card"], fg=C["fg2"], font=SAN).pack(anchor="w", padx=14)
        self._rint_v = tk.StringVar(value=str(load_cfg().get("rebalance_minutes",20)))
        self._rint_lbl = tk.Label(lf, textvariable=self._rint_v, bg=C["card"], fg=C["fg"], font=SAN)
        self._rint_lbl.pack(anchor="w", padx=14)
        self._rint_sl = tk.Scale(lf, from_=5, to=120, orient="horizontal",
                                  bg=C["card"], fg=C["fg"], troughcolor=C["entry"],
                                  highlightthickness=0, bd=0,
                                  command=lambda v: self._rint_v.set(f"{int(float(v))} min"))
        self._rint_sl.set(load_cfg().get("rebalance_minutes",20))
        self._rint_sl.pack(fill="x", padx=14, pady=(0,8))

        tk.Label(lf, text="Duration (hours)", bg=C["card"], fg=C["fg2"], font=SAN).pack(anchor="w", padx=14)
        self._dur_e = tk.Entry(lf, bg=C["entry"], fg=C["fg"], insertbackground=C["fg"],
                                bd=1, relief="flat", font=SAN, width=20)
        self._dur_e.insert(0, str(load_cfg().get("duration_hours",48)))
        self._dur_e.pack(anchor="w", padx=14, pady=(2,8))

        Btn(lf, "Apply Settings", self._apply_tsettings).pack(fill="x", padx=14, pady=4)
        Sep(lf).pack(fill="x", padx=14, pady=8)
        Btn(lf, "Clear Snapshot & State", self._clear_state).pack(fill="x", padx=14, pady=(0,14))

        rf = tk.Frame(p, bg=C["card"])
        rf.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)
        th = tk.Frame(rf, bg=C["card"]); th.pack(fill="x", padx=10, pady=(8,2))
        tk.Label(th, text="Live Output", bg=C["card"], fg=C["fg"], font=SAN_B).pack(side="left")
        Btn(th, "Clear", lambda: self._clr_log(self._tlog)).pack(side="right")
        tlf, self._tlog = scrolled_text(rf, wrap="none")
        tlf.pack(fill="both", expand=True, padx=8, pady=(0,8))

    def _tog_trade(self):
        if self._tproc and self._tproc.poll() is None: self._stop_trade()
        else: self._start_trade()

    def _start_trade(self):
        self._tproc = subprocess.Popen(
            [sys.executable, str(ROOT/"paper_trading.py")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT))
        threading.Thread(target=self._read, args=(self._tproc,self._tq), daemon=True).start()
        self._tbtn.configure(text="  Stop Trading  ", bg=C["red"], activebackground=C["red"], fg=C["fg"])
        self._tstat.configure(text="Status: Running", fg=C["green"])

    def _stop_trade(self):
        if self._tproc:
            self._tproc.terminate()
            try: self._tproc.wait(5)
            except: self._tproc.kill()
        self._tproc = None
        self._tbtn.configure(text="  Start Trading  ", bg=C["green"], activebackground=C["green"], fg="black")
        self._tstat.configure(text="Status: Stopped", fg=C["gray"])

    def _apply_tsettings(self):
        cfg = load_cfg()
        try: cfg["rebalance_minutes"] = int(float(self._rint_sl.get()))
        except: pass
        try: cfg["duration_hours"] = float(self._dur_e.get())
        except: pass
        save_cfg(cfg)
        rp = self._rp.get()
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            try: st = json.loads(STATE_PATH.read_text())
            except: st = {}
            st["requested"] = rp
            STATE_PATH.write_text(json.dumps(st, indent=2))
        except: pass
        self._tstat.configure(text=f"Applied (Risk: {rp})", fg=C["yellow"])

    def _clear_state(self):
        for path in [SNAPSHOT_PATH, STATE_PATH]:
            try: path.unlink(missing_ok=True)
            except: pass

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: TRADES
    # ══════════════════════════════════════════════════════════════════════════
    def _page_trades(self, p: tk.Frame):
        p.configure(bg=C["bg"])
        top = tk.Frame(p, bg=C["side"]); top.pack(fill="x", padx=8, pady=(8,4))
        sh  = tk.Frame(top, bg=C["side"]); sh.pack(fill="x", padx=6, pady=6)
        self._tsum = tk.Label(sh, text="—", bg=C["side"], fg=C["fg2"], font=SAN)
        self._tsum.pack(side="left", padx=6)
        Btn(sh, "Refresh", self._load_trades).pack(side="right", padx=6)

        fr = tk.Frame(top, bg=C["side"]); fr.pack(fill="x", padx=6, pady=(0,6))
        tk.Label(fr, text="Filter:", bg=C["side"], fg=C["fg2"], font=SAN).pack(side="left", padx=(6,4))
        self._tfilt = tk.StringVar(value="ALL")
        for val in ["ALL","BUY","SELL"]:
            tk.Radiobutton(fr, text=val, variable=self._tfilt, value=val,
                           bg=C["side"], fg=C["fg"], selectcolor=C["side"],
                           activebackground=C["side"], font=SAN,
                           command=self._load_trades).pack(side="left", padx=8)

        mf = tk.Frame(p, bg=C["card"]); mf.pack(fill="both", expand=True, padx=8, pady=(0,8))
        xsb = tk.Scrollbar(mf, orient="horizontal", bg=C["card"], troughcolor=C["entry"], relief="flat")
        ysb = tk.Scrollbar(mf, bg=C["card"], troughcolor=C["entry"], relief="flat")
        self._trade_t = Text(mf, wrap="none",
                              xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        xsb.configure(command=self._trade_t.xview)
        ysb.configure(command=self._trade_t.yview)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self._trade_t.pack(fill="both", expand=True, padx=8, pady=8)
        self._load_trades()

    def _load_trades(self):
        f = trade_file()
        if not f: tb_set(self._trade_t,"  No trade_history.jsonl found."); return
        rows = load_jsonl(f, 500)
        flt  = self._tfilt.get()
        if flt != "ALL":
            rows = [r for r in rows if str(r.get("side",r.get("action",""))).upper()==flt]
        total = len(rows)
        buys  = sum(1 for r in rows if str(r.get("side",r.get("action",""))).upper()=="BUY")
        fees  = sum(float(r.get("cost",r.get("commission",0)) or 0) for r in rows)
        self._tsum.configure(
            text=f"Total: {total}   BUY: {buys}   SELL: {total-buys}   Fees: {usd(fees)}", fg=C["fg"])
        hdr  = f"  {'Date':>16}  {'Ticker':<7}  {'Side':<5}  {'Qty':>7}  {'Price':>10}  {'Value':>11}"
        sep  = "  "+"─"*64
        lines= [hdr,sep]
        for r in reversed(rows[-200:]):
            ts   = str(r.get("timestamp",r.get("time","—")))[:16].replace("T"," ")
            tkr  = str(r.get("ticker",r.get("symbol","—")))
            side = str(r.get("side",r.get("action","—"))).upper()
            qty  = r.get("quantity",r.get("qty",0)) or 0
            price= r.get("price",r.get("fill_price","—"))
            val  = r.get("value",r.get("total_value","—"))
            lines.append(
                f"  {ts:>16}  {tkr:<7}  {side:<5}  {float(qty):>7.0f}"
                f"  {usd(price) if price!='—' else '—':>10}"
                f"  {usd(val)   if val!='—'   else '—':>11}")
        tb_set(self._trade_t, "\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: NEWS & AI
    # ══════════════════════════════════════════════════════════════════════════
    def _page_news(self, p: tk.Frame):
        p.configure(bg=C["bg"])
        lf = tk.Frame(p, bg=C["card"], width=300)
        lf.pack(side="left", fill="y", padx=(8,4), pady=8)
        lf.pack_propagate(False)

        Btn(lf, "  Run Once  ", self._news_once, green=True).pack(fill="x", padx=12, pady=(12,4))
        self._nbtn = Btn(lf, "  Start Loop  ", self._tog_news)
        self._nbtn.pack(fill="x", padx=12, pady=4)
        self._nstat = tk.Label(lf, text="Status: Idle", bg=C["card"], fg=C["gray"], font=SAN)
        self._nstat.pack(pady=(0,8))

        Sep(lf).pack(fill="x", padx=12, pady=4)
        tk.Label(lf, text="Loop interval (min)", bg=C["card"], fg=C["fg2"], font=SAN).pack(anchor="w", padx=12)
        self._nint_v = tk.StringVar(value="30 min")
        tk.Label(lf, textvariable=self._nint_v, bg=C["card"], fg=C["fg"], font=SAN).pack(anchor="w", padx=12)
        self._nint_sl = tk.Scale(lf, from_=10, to=120, orient="horizontal",
                                  bg=C["card"], fg=C["fg"], troughcolor=C["entry"],
                                  highlightthickness=0, bd=0,
                                  command=lambda v: self._nint_v.set(f"{int(float(v))} min"))
        self._nint_sl.set(30)
        self._nint_sl.pack(fill="x", padx=12, pady=(0,8))

        tk.Label(lf, text="Ollama Model", bg=C["card"], fg=C["fg2"], font=SAN).pack(anchor="w", padx=12)
        self._model_e = tk.Entry(lf, bg=C["entry"], fg=C["fg"], insertbackground=C["fg"],
                                  bd=1, relief="flat", font=SAN, width=22)
        self._model_e.insert(0, load_cfg().get("macro_integration",{}).get("llm_topic_model","qwen2.5:32b"))
        self._model_e.pack(anchor="w", padx=12, pady=(2,8))

        Sep(lf).pack(fill="x", padx=12, pady=4)
        sh = tk.Frame(lf, bg=C["card"]); sh.pack(fill="x", padx=12, pady=(4,2))
        tk.Label(sh, text="Scoreboard", bg=C["card"], fg=C["fg"], font=SAN_B).pack(side="left")
        Btn(sh, "R", self._do_score).pack(side="right")
        sf, self._score_t = scrolled_text(lf, height=4)
        sf.pack(fill="x", padx=12, pady=(2,4))

        sh2 = tk.Frame(lf, bg=C["card"]); sh2.pack(fill="x", padx=12, pady=(4,2))
        tk.Label(sh2, text="Recent Signals", bg=C["card"], fg=C["fg"], font=SAN_B).pack(side="left")
        Btn(sh2, "R", self._do_signals).pack(side="right")
        sgf, self._sig_t = scrolled_text(lf)
        sgf.pack(fill="both", expand=True, padx=12, pady=(2,12))

        rf = tk.Frame(p, bg=C["card"])
        rf.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)
        nh = tk.Frame(rf, bg=C["card"]); nh.pack(fill="x", padx=10, pady=(8,2))
        tk.Label(nh, text="Pipeline Output", bg=C["card"], fg=C["fg"], font=SAN_B).pack(side="left")
        Btn(nh, "Clear", lambda: self._clr_log(self._nlog)).pack(side="right")
        nlf, self._nlog = scrolled_text(rf, wrap="none")
        nlf.pack(fill="both", expand=True, padx=8, pady=(0,8))

        self._do_score(); self._do_signals()

    def _news_once(self):
        env = {**os.environ, "GW_LOCAL_MODEL": self._model_e.get().strip() or "qwen2.5:32b"}
        proc = subprocess.Popen(
            [sys.executable, str(ROOT/"run_news_pipeline.py"), "--once"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT), env=env)
        threading.Thread(target=self._read, args=(proc,self._nq), daemon=True).start()
        self._nstat.configure(text="Running…", fg=C["yellow"])

    def _tog_news(self):
        if self._nproc and self._nproc.poll() is None: self._stop_news()
        else: self._start_news()

    def _start_news(self):
        interval = int(float(self._nint_sl.get()))
        env = {**os.environ, "GW_LOCAL_MODEL": self._model_e.get().strip() or "qwen2.5:32b"}
        self._nproc = subprocess.Popen(
            [sys.executable, str(ROOT/"run_news_pipeline.py"), "--interval", str(interval)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT), env=env)
        threading.Thread(target=self._read, args=(self._nproc,self._nq), daemon=True).start()
        self._nbtn.configure(text="  Stop Loop  ", bg=C["red"], activebackground=C["red"], fg=C["fg"])
        self._nstat.configure(text=f"Loop every {interval}m", fg=C["green"])

    def _stop_news(self):
        if self._nproc:
            self._nproc.terminate()
            try: self._nproc.wait(5)
            except: self._nproc.kill()
        self._nproc = None
        self._nbtn.configure(text="  Start Loop  ", bg="#1e3a5f", activebackground="#1e3a5f", fg=C["fg"])
        self._nstat.configure(text="Status: Idle", fg=C["gray"])

    def _do_score(self):
        rows = load_jsonl(SCORE_PATH, 100)
        if not rows: tb_set(self._score_t,"  No data yet."); return
        total = len(rows)
        correct = sum(1 for r in rows if r.get("correct") or r.get("outcome")=="correct")
        acc = correct/total*100 if total else 0
        rets = [float(r["actual_return"]) for r in rows if r.get("actual_return") is not None]
        ar = sum(rets)/len(rets)*100 if rets else 0
        tb_set(self._score_t, f"  Signals: {total}   Accuracy: {acc:.1f}%   Avg Ret: {ar:+.2f}%")

    def _do_signals(self):
        try:
            import chromadb
            cfg = load_cfg()
            cp  = cfg.get("macro_integration",{}).get("chroma_path","./memory_db")
            cli = chromadb.PersistentClient(path=str(ROOT/cp.lstrip("./")))
            col = cli.get_or_create_collection("trading_signals")
            res = col.get(limit=14, include=["metadatas"])
            metas = res.get("metadatas",[]) or []
            lines = []
            for m in reversed(metas):
                d  = str(m.get("direction","?")).upper()
                t  = str(m.get("topic","?"))[:24]
                c  = m.get("confidence",0)
                ts = str(m.get("timestamp",""))[:16].replace("T"," ")
                ar = "^" if d=="BULLISH" else ("v" if d=="BEARISH" else "-")
                lines.append(f"{ar} {t:<24} {float(c):.0%}  {ts}")
            tb_set(self._sig_t, "\n".join(lines) or "No signals yet.")
        except ImportError:
            tb_set(self._sig_t, "chromadb not installed.\npip3 install chromadb")
        except Exception as e:
            tb_set(self._sig_t, f"ChromaDB: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: MACRO
    # ══════════════════════════════════════════════════════════════════════════
    def _page_macro(self, p: tk.Frame):
        p.configure(bg=C["bg"])
        lf = tk.Frame(p, bg=C["card"]); lf.pack(side="left", fill="both", expand=True, padx=(8,4), pady=8)
        tk.Label(lf, text="Macro Themes", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=10, pady=(8,2))
        mf, self._macro_t = scrolled_text(lf)
        mf.pack(fill="both", expand=True, padx=8, pady=(0,4))
        tk.Label(lf, text="Regime & Trend", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=10, pady=(4,2))
        rgf, self._regime_t = scrolled_text(lf, height=8)
        rgf.pack(fill="x", padx=8, pady=(0,8))

        rf = tk.Frame(p, bg=C["card"]); rf.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)
        ewh = tk.Frame(rf, bg=C["card"]); ewh.pack(fill="x", padx=10, pady=(8,2))
        tk.Label(ewh, text="Early Warning Scores", bg=C["card"], fg=C["fg"], font=SAN_B).pack(side="left")
        Btn(ewh, "Refresh", self._do_ew).pack(side="right")
        ewf, self._ew_t = scrolled_text(rf)
        ewf.pack(fill="both", expand=True, padx=8, pady=(0,4))
        tk.Label(rf, text="FX / Config", bg=C["card"], fg=C["fg"], font=SAN_B).pack(anchor="w", padx=10, pady=(4,2))
        fxf, self._fx_t = scrolled_text(rf, height=8)
        fxf.pack(fill="x", padx=8, pady=(0,8))
        self._do_ew()

    def _upd_macro(self, s: dict):
        lm = s.get("last_macro", s.get("macro_state",{})) or {}
        lines = [f"  {k:<30} {v}" for k,v in lm.items() if isinstance(v,(str,float,int,bool))]
        tb_set(self._macro_t, "\n".join(lines) or "  No macro data yet.")
        reg = s.get("regime", s.get("market_regime",{})) or {}
        rlines = [f"  {k:<30} {v}" for k,v in reg.items() if isinstance(v,(str,float,int,bool))]
        tb_set(self._regime_t, "\n".join(rlines) or "  No regime data.")
        cfg = load_cfg()
        fx  = cfg.get("macro_integration",{})
        fxl = [f"  {k:<30} {v}" for k,v in fx.items()
               if any(x in k.lower() for x in ["fx","rate","currency"])]
        tb_set(self._fx_t, "\n".join(fxl) or "  (No FX entries in config)")

    def _do_ew(self):
        try:
            import chromadb
            cfg = load_cfg()
            cp  = cfg.get("macro_integration",{}).get("chroma_path","./memory_db")
            cli = chromadb.PersistentClient(path=str(ROOT/cp.lstrip("./")))
            col = cli.get_or_create_collection("trading_signals")
            res = col.get(limit=20, include=["metadatas"])
            metas = res.get("metadatas",[]) or []
            td: dict[str,list] = {}
            for m in metas:
                td.setdefault(str(m.get("topic","?")), []).append(m)
            lines = [f"  {'Topic':<28} {'Score':>5}  {'Signal':<6}", "  "+"─"*44]
            for topic, sigs in sorted(td.items(), key=lambda x: -max(float(s.get("confidence",0)) for s in x[1])):
                dirs = [s.get("direction","") for s in sigs]
                bear = sum(1 for d in dirs if "bear" in str(d).lower())
                bull = sum(1 for d in dirs if "bull" in str(d).lower())
                sc   = int(max(float(s.get("confidence",0)) for s in sigs)*10)
                lbl  = "RISK" if bear>bull else ("OK" if bull>bear else "WATCH")
                lines.append(f"  {topic[:28]:<28} {sc:>5}/10  {lbl:<6}")
            tb_set(self._ew_t, "\n".join(lines))
        except ImportError:
            tb_set(self._ew_t,"  chromadb not installed.\n  pip3 install chromadb")
        except Exception as e:
            tb_set(self._ew_t, f"  ChromaDB: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    def _page_settings(self, p: tk.Frame):
        p.configure(bg=C["bg"])

        canvas = tk.Canvas(p, bg=C["bg"], highlightthickness=0)
        vsb    = tk.Scrollbar(p, orient="vertical", command=canvas.yview,
                               bg=C["card"], troughcolor=C["entry"], relief="flat", width=12)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", pady=6, padx=(0,6))
        canvas.pack(side="left", fill="both", expand=True, padx=(8,0), pady=6)

        inner  = tk.Frame(canvas, bg=C["bg"])
        wid    = canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(wid, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        tk.Label(inner, text="Trading Parameters", bg=C["bg"], fg=C["fg"],
                 font=("Helvetica",16,"bold")).pack(anchor="w", padx=14, pady=(12,2))
        tk.Label(inner, text="Saved to paper_config.json  ·  Restart trading for changes to take effect",
                 bg=C["bg"], fg=C["gray"], font=SAN_S).pack(anchor="w", padx=14, pady=(0,10))

        cfg      = load_cfg()
        ex       = cfg.get("execution",{})
        mac      = cfg.get("macro_integration",{})
        mid_obj  = cfg.get("risk_profiles",{}).get("mid",{}).get("objectives",{})
        rm       = cfg.get("risk_model",{})
        self._sw: dict[str,tuple] = {}

        sections = [
            ("Portfolio", [
                ("initial_cash_usd",  "Initial Cash ($)",          cfg.get("initial_cash_usd",80000),  "entry"),
                ("duration_hours",    "Duration (hours)",          cfg.get("duration_hours",48),        "entry"),
                ("rebalance_minutes", "Rebalance Interval (min)",  cfg.get("rebalance_minutes",20),     "scale",5,120),
            ]),
            ("Risk Management", [
                ("risk.min_cash_fraction",    "Min Cash %",       mid_obj.get("min_cash_fraction",0.12),    "scale",0.02,0.50),
                ("risk.max_weight_per_asset", "Max Position Size",mid_obj.get("max_weight_per_asset",0.15), "scale",0.02,0.40),
                ("risk.exposure_cap",         "Exposure Cap",     mid_obj.get("exposure_cap",0.95),         "scale",0.50,1.00),
                ("risk.correlation_threshold","Corr. Threshold",  rm.get("correlation_threshold",0.80),     "scale",0.40,1.00),
            ]),
            ("Circuit Breaker", [
                ("execution.circuit_breaker_rolling_enabled",      "Rolling CB Enabled",    ex.get("circuit_breaker_rolling_enabled",True),      "check"),
                ("execution.circuit_breaker_rolling_window",       "CB Window (cycles)",    ex.get("circuit_breaker_rolling_window",10),          "scale",3,30),
                ("execution.circuit_breaker_rolling_drawdown_pct", "CB Drawdown Threshold", ex.get("circuit_breaker_rolling_drawdown_pct",0.12),  "scale",0.03,0.30),
                ("execution.circuit_breaker_rolling_recovery_pct", "CB Recovery %",         ex.get("circuit_breaker_rolling_recovery_pct",0.03),  "scale",0.01,0.15),
            ]),
            ("Macro / AI", [
                ("macro_integration.enabled",                  "Macro Enabled",       mac.get("enabled",True),                        "check"),
                ("macro_integration.enable_llm_topic_signals", "LLM Topic Signals",   mac.get("enable_llm_topic_signals",True),       "check"),
                ("macro_integration.llm_topic_model",          "Ollama Model",        mac.get("llm_topic_model","qwen2.5:32b"),       "entry"),
                ("macro_integration.signal_max_age_hours",     "Max Signal Age (hrs)",mac.get("signal_max_age_hours",72),             "scale",12,168),
                ("macro_integration.news_score_weight",        "News Score Weight",   mac.get("news_score_weight",0.15),              "scale",0.0,0.5),
            ]),
            ("Execution", [
                ("execution.min_trade_notional_usd",         "Min Trade Notional ($)",ex.get("min_trade_notional_usd",400),          "scale",50,2000),
                ("execution.weight_threshold",               "Weight Threshold",      ex.get("weight_threshold",0.025),              "scale",0.005,0.10),
                ("execution.max_turnover_pct_per_rebalance", "Max Turnover %",        ex.get("max_turnover_pct_per_rebalance",0.4),  "scale",0.05,1.00),
                ("execution.stale_price_skip_minutes",       "Stale Price Skip (min)",ex.get("stale_price_skip_minutes",60),         "scale",10,240),
            ]),
        ]

        for sec_name, params in sections:
            sec = tk.LabelFrame(inner, text=f"  {sec_name}  ",
                                bg=C["card"], fg=C["fg"], font=SAN_B,
                                bd=1, relief="groove", labelanchor="nw")
            sec.pack(fill="x", padx=14, pady=6, ipady=4)
            for ri, param in enumerate(params):
                key, lbl, val, ptype = param[0],param[1],param[2],param[3]
                tk.Label(sec, text=lbl, bg=C["card"], fg=C["fg2"], font=SAN,
                         width=32, anchor="w").grid(row=ri, column=0, padx=(14,8), pady=5, sticky="w")
                if ptype == "entry":
                    w = tk.Entry(sec, bg=C["entry"], fg=C["fg"], insertbackground=C["fg"],
                                  bd=1, relief="flat", font=SAN, width=22)
                    w.insert(0, str(val)); w.grid(row=ri, column=1, padx=4, pady=5, sticky="w")
                elif ptype == "check":
                    bv = tk.BooleanVar(value=bool(val))
                    tk.Checkbutton(sec, variable=bv, bg=C["card"], activebackground=C["card"],
                                   selectcolor=C["entry"], cursor="hand2",
                                   fg=C["fg"]).grid(row=ri, column=1, padx=4, pady=5, sticky="w")
                    w = bv
                elif ptype == "scale":
                    fv, tv = float(param[4]), float(param[5])
                    is_f   = tv <= 1.5
                    vl = tk.Label(sec, bg=C["card"], fg=C["fg"], font=SAN,
                                  text=f"{float(val):.3f}" if is_f else f"{int(float(val))}", width=8, anchor="w")
                    vl.grid(row=ri, column=2, padx=4, pady=5)
                    def _cmd(label, float_mode):
                        return lambda v: label.configure(
                            text=f"{float(v):.3f}" if float_mode else f"{int(float(v))}")
                    w = tk.Scale(sec, from_=fv, to=tv, orient="horizontal", length=200,
                                  bg=C["card"], fg=C["fg"], troughcolor=C["entry"],
                                  highlightthickness=0, bd=0, command=_cmd(vl, is_f))
                    w.set(float(val)); w.grid(row=ri, column=1, padx=4, pady=5)
                else:
                    continue
                self._sw[key] = (w, ptype)

        br = tk.Frame(inner, bg=C["bg"]); br.pack(fill="x", padx=14, pady=(4,24))
        Btn(br, "  Save All Settings  ", self._save_settings, green=True).pack(side="left", padx=4)
        self._save_lbl = tk.Label(br, text="", bg=C["bg"], fg=C["green"], font=SAN)
        self._save_lbl.pack(side="left", padx=10)

    def _save_settings(self):
        cfg = load_cfg()
        for key,(w,ptype) in self._sw.items():
            if ptype=="entry":
                raw=w.get().strip()
                try: val: object = float(raw) if "." in raw else int(raw)
                except: val=raw
            elif ptype=="check": val=bool(w.get())
            elif ptype=="scale": val=float(w.get())
            else: continue
            parts = key.split(".")
            if len(parts)==1: cfg[parts[0]]=val
            elif len(parts)==2:
                sec2,k=parts
                if sec2=="risk":
                    for prof in cfg.get("risk_profiles",{}).values():
                        prof.setdefault("objectives",{})[k]=val
                else: cfg.setdefault(sec2,{})[k]=val
            elif len(parts)==3:
                cfg.setdefault(parts[0],{}).setdefault(parts[1],{})[parts[2]]=val
        save_cfg(cfg)
        self._save_lbl.configure(text=f"Saved {datetime.now().strftime('%H:%M:%S')}")
        self.after(3000, lambda: self._save_lbl.configure(text=""))

    # ══════════════════════════════════════════════════════════════════════════
    # BACKGROUND
    # ══════════════════════════════════════════════════════════════════════════
    def _poll(self):
        while self._alive:
            s = load_snap()
            if s:
                self.after(0, self._upd_dashboard, s)
                self.after(0, self._upd_analytics, s)
                self.after(0, self._upd_macro, s)
            if self._tproc and self._tproc.poll() is not None:
                self._tproc = None
                def _died():
                    self._tbtn.configure(text="  Start Trading  ", bg=C["green"],
                                         activebackground=C["green"], fg="black")
                    self._tstat.configure(text="Status: Exited", fg=C["yellow"])
                self.after(0, _died)
            self._status_lbl.configure(text=datetime.now().strftime("%H:%M:%S"))
            time.sleep(5)

    def _drain(self):
        for _ in range(80):
            try: self._append_log(self._tlog, self._tq.get_nowait())
            except queue.Empty: break
        for _ in range(80):
            try: self._append_log(self._nlog, self._nq.get_nowait())
            except queue.Empty: break
        self.after(150, self._drain)

    def _read(self, proc: subprocess.Popen, q: queue.Queue):
        try:
            for ln in proc.stdout: q.put(ln)
        except: pass

    def _append_log(self, t: tk.Text, text: str):
        t.configure(state="normal")
        t.insert("end", text)
        t.see("end")
        n = int(t.index("end-1c").split(".")[0])
        if n > 2000: t.delete("1.0", f"{n-2000}.0")
        t.configure(state="disabled")

    def _clr_log(self, t: tk.Text):
        t.configure(state="normal"); t.delete("1.0","end"); t.configure(state="disabled")

    def _close(self):
        self._alive = False
        for proc in (self._tproc, self._nproc):
            if proc and proc.poll() is None:
                proc.terminate()
                try: proc.wait(4)
                except: proc.kill()
        self.destroy()


def main():
    os.environ.setdefault("TK_SILENCE_DEPRECATION","1")
    App().mainloop()

if __name__ == "__main__":
    main()
