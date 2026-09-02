import { num } from '../../lib/ui'

export type Counts = { tp: number; fp: number; fn: number; tn: number }

type Cell = {
  key: keyof Counts
  title: string
  meaning: string
  tone: 'good' | 'warning' | 'critical' | 'neutral'
}

const CELLS: Cell[] = [
  { key: 'tp', title: 'Called, subscribed', meaning: 'Revenue won', tone: 'good' },
  { key: 'fn', title: 'Skipped, would have subscribed', meaning: 'Revenue missed', tone: 'critical' },
  { key: 'fp', title: 'Called, declined', meaning: 'Budget wasted', tone: 'warning' },
  { key: 'tn', title: 'Skipped, would have declined', meaning: 'Budget saved', tone: 'neutral' },
]

const TONE = {
  good: { ink: 'var(--color-good)', wash: 'rgba(12,163,12,0.14)', glyph: '✓' },
  warning: { ink: 'var(--color-warning)', wash: 'rgba(250,178,25,0.12)', glyph: '!' },
  critical: { ink: 'var(--color-critical)', wash: 'rgba(230,103,103,0.13)', glyph: '×' },
  neutral: { ink: 'var(--color-ink-3)', wash: 'rgba(255,255,255,0.04)', glyph: '·' },
} as const

/**
 * Outcome grid rather than a raw confusion matrix. Each cell is a business event,
 * so the colour is a status token (with a glyph, never colour alone) instead of a
 * magnitude ramp — which would have made the huge true-negative cell drown the rest.
 */
export function ConfusionGrid({ counts }: { counts: Counts }) {
  const total = counts.tp + counts.fp + counts.fn + counts.tn

  return (
    <ul className="m-0 grid list-none grid-cols-2 gap-2 p-0">
      {CELLS.map((cell) => {
        const tone = TONE[cell.tone]
        const value = counts[cell.key]
        return (
          <li
            key={cell.key}
            className="rounded-lg border border-white/8 p-3"
            style={{ background: tone.wash }}
          >
            <div className="flex items-baseline gap-1.5">
              <span aria-hidden className="text-[0.8125rem] leading-none" style={{ color: tone.ink }}>
                {tone.glyph}
              </span>
              <span
                className="text-[0.6875rem] font-medium tracking-wide uppercase"
                style={{ color: tone.ink }}
              >
                {cell.meaning}
              </span>
            </div>
            <div className="metric num mt-2 text-[1.5rem] tabular-nums">{num(value)}</div>
            <div className="num mt-0.5 text-[0.6875rem] text-ink-3 tabular-nums">
              {((value / total) * 100).toFixed(1)}% of test set
            </div>
            <div className="mt-1.5 text-[0.75rem] leading-snug text-ink-2">{cell.title}</div>
          </li>
        )
      })}
    </ul>
  )
}
