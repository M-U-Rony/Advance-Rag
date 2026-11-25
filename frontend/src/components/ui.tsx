import { Info } from "lucide-react";
import type { ReactNode } from "react";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Small "i" icon that reveals an explanation on hover or keyboard focus. */
export function Hint({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        aria-label={text}
        className="rounded-full text-slate-500 outline-none transition hover:text-indigo-300 focus-visible:text-indigo-300 focus-visible:ring-2 focus-visible:ring-indigo-400/60"
      >
        <Info size={13} />
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 hidden w-56 -translate-x-1/2 rounded-lg border border-white/10 bg-ink-700 px-3 py-2 text-xs font-normal leading-relaxed text-slate-200 shadow-xl group-focus-within:block group-hover:block"
      >
        {text}
      </span>
    </span>
  );
}

export function Field({
  label,
  hint,
  right,
  children,
}: {
  label: string;
  hint: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-400">
          {label}
          <Hint text={hint} />
        </span>
        {right}
      </div>
      {children}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid auto-cols-fr grid-flow-col gap-1 rounded-lg bg-black/30 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md px-2 py-1.5 text-xs font-medium outline-none transition focus-visible:ring-2 focus-visible:ring-indigo-400/70",
            o.value === value
              ? "bg-gradient-to-r from-indigo-500 to-violet-500 text-white shadow"
              : "text-slate-400 hover:bg-white/5 hover:text-slate-200",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cx(
        "relative h-6 w-11 shrink-0 rounded-full outline-none transition focus-visible:ring-2 focus-visible:ring-indigo-400/70",
        checked ? "bg-gradient-to-r from-indigo-500 to-violet-500" : "bg-white/10",
      )}
    >
      <span
        className={cx(
          "absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all",
          checked ? "left-[22px]" : "left-0.5",
        )}
      />
    </button>
  );
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  label: string;
}) {
  return (
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-indigo-400"
    />
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "accent";
}) {
  const tones = {
    neutral: "border-white/10 bg-white/5 text-slate-300",
    good: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
    warn: "border-amber-400/30 bg-amber-400/10 text-amber-300",
    accent: "border-indigo-400/30 bg-indigo-400/10 text-indigo-200",
  };
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium", tones[tone])}>
      {children}
    </span>
  );
}
