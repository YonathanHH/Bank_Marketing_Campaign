type Props = {
  /** Model probability, 0–1. */
  value: number
  /** Decision cut-off, drawn as a tick on the arc. */
  threshold: number
  max: number
  label: string
}

const R = 78
const CX = 100
const CY = 96
const SPAN = Math.PI // semicircle, left to right

function point(t: number) {
  const angle = Math.PI - t * SPAN
  return [CX + R * Math.cos(angle), CY - R * Math.sin(angle)] as const
}

function arc(from: number, to: number) {
  const [x0, y0] = point(from)
  const [x1, y1] = point(to)
  return `M${x0.toFixed(2)},${y0.toFixed(2)} A${R},${R} 0 ${to - from > 0.5 ? 1 : 0} 1 ${x1.toFixed(2)},${y1.toFixed(2)}`
}

/**
 * The hero figure of the predictor. The arc is scaled to the model's own output
 * range rather than 0–1, because a class-balanced booster on an 11% base rate
 * almost never emits a probability above ~0.95 and a flat-looking needle lies.
 */
export function Gauge({ value, threshold, max, label }: Props) {
  const t = Math.min(1, Math.max(0, value / max))
  const tickAt = Math.min(1, threshold / max)
  const [tx, ty] = point(tickAt)
  const [ix, iy] = [CX + (R - 13) * Math.cos(Math.PI - tickAt * SPAN), CY - (R - 13) * Math.sin(Math.PI - tickAt * SPAN)]
  const above = value >= threshold

  return (
    <svg
      viewBox="0 0 200 116"
      className="w-full max-w-[16rem]"
      role="img"
      aria-label={`${label}: ${(value * 100).toFixed(1)} percent, decision threshold ${(threshold * 100).toFixed(0)} percent`}
    >
      <path d={arc(0, 1)} fill="none" stroke="var(--color-grid)" strokeWidth={10} strokeLinecap="round" />
      {t > 0.004 && (
        <path
          d={arc(0, t)}
          fill="none"
          stroke={above ? 'var(--color-series-3)' : 'var(--color-series-1)'}
          strokeWidth={10}
          strokeLinecap="round"
          style={{ transition: 'd 240ms ease' }}
        />
      )}
      <line x1={tx} y1={ty} x2={ix} y2={iy} stroke="var(--color-ink)" strokeWidth={2} />

      <text
        x={CX}
        y={CY - 18}
        textAnchor="middle"
        className="fill-white text-[30px] font-semibold tabular-nums"
        style={{ letterSpacing: '-0.03em' }}
      >
        {(value * 100).toFixed(1)}%
      </text>
      <text x={CX} y={CY + 2} textAnchor="middle" className="fill-ink-3 text-[9px]">
        subscription probability
      </text>
      <text x={CX - R} y={CY + 14} textAnchor="middle" className="fill-ink-3 text-[9px] tabular-nums">
        0
      </text>
      <text x={CX + R} y={CY + 14} textAnchor="middle" className="fill-ink-3 text-[9px] tabular-nums">
        {(max * 100).toFixed(0)}%
      </text>
    </svg>
  )
}
