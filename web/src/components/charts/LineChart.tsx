import { useState } from 'react'
import { useWidth } from '../../lib/ui'

export type Series = {
  label: string
  color: string
  points: [number, number][]
  dashed?: boolean
  /** Draw a filled marker at the last point (used for the elbow's chosen k). */
  markAt?: number
}

type Props = {
  series: Series[]
  xDomain: [number, number]
  yDomain: [number, number]
  xLabel: string
  yLabel: string
  xTicks: number[]
  yTicks: number[]
  format?: (v: number, axis: 'x' | 'y') => string
  height?: number
  ariaLabel: string
}

const PAD = { top: 12, right: 14, bottom: 34, left: 46 }

/** Multi-series line chart on a single y-axis. Never a second scale. */
export function LineChart({
  series,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  xTicks,
  yTicks,
  format = (v) => String(v),
  height = 300,
  ariaLabel,
}: Props) {
  const [hover, setHover] = useState<{ x: number; y: number; sx: number; sy: number } | null>(null)
  // one SVG unit = one CSS pixel, so tick labels stay ~11px at every container width
  const { ref, width } = useWidth<HTMLDivElement>(560)
  const W = Math.max(280, width)
  const H = height
  const iw = W - PAD.left - PAD.right
  const ih = H - PAD.top - PAD.bottom

  const sx = (v: number) => PAD.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * iw
  const sy = (v: number) => PAD.top + ih - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * ih

  const path = (pts: [number, number][]) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`).join(' ')

  const track = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    const px = ((event.clientX - box.left) / box.width) * W
    const primary = series[0].points
    let best = primary[0]
    for (const p of primary) {
      if (Math.abs(sx(p[0]) - px) < Math.abs(sx(best[0]) - px)) best = p
    }
    setHover({ x: best[0], y: best[1], sx: sx(best[0]), sy: sy(best[1]) })
  }

  return (
    <div className="relative" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={ariaLabel}
        onMouseMove={track}
        onMouseLeave={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={sy(t)}
              y2={sy(t)}
              stroke="var(--color-grid)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 8}
              y={sy(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-ink-3 text-[11px] tabular-nums"
            >
              {format(t, 'y')}
            </text>
          </g>
        ))}

        {xTicks.map((t) => (
          <text
            key={`x${t}`}
            x={sx(t)}
            y={H - PAD.bottom + 16}
            textAnchor="middle"
            className="fill-ink-3 text-[11px] tabular-nums"
          >
            {format(t, 'x')}
          </text>
        ))}

        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={PAD.top + ih}
          y2={PAD.top + ih}
          stroke="var(--color-axis)"
          strokeWidth={1}
        />

        {series.map((s) => (
          <path
            key={s.label}
            d={path(s.points)}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray={s.dashed ? '5 5' : undefined}
            opacity={s.dashed ? 0.6 : 1}
          />
        ))}

        {series.map((s) =>
          s.markAt === undefined ? null : (
            <circle
              key={`${s.label}-mark`}
              cx={sx(s.markAt)}
              cy={sy(s.points.find((p) => p[0] === s.markAt)?.[1] ?? 0)}
              r={5}
              fill={s.color}
              stroke="var(--color-panel)"
              strokeWidth={2}
            />
          ),
        )}

        {hover && (
          <g aria-hidden>
            <line
              x1={hover.sx}
              x2={hover.sx}
              y1={PAD.top}
              y2={PAD.top + ih}
              stroke="var(--color-axis)"
              strokeWidth={1}
            />
            <circle
              cx={hover.sx}
              cy={hover.sy}
              r={4.5}
              fill={series[0].color}
              stroke="var(--color-panel)"
              strokeWidth={2}
            />
          </g>
        )}

        <text
          x={PAD.left + iw / 2}
          y={H - 4}
          textAnchor="middle"
          className="fill-ink-3 text-[11px]"
        >
          {xLabel}
        </text>
        <text
          x={12}
          y={PAD.top + ih / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${PAD.top + ih / 2})`}
          className="fill-ink-3 text-[11px]"
        >
          {yLabel}
        </text>
      </svg>

      {hover && (
        <div
          className="num pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border border-white/12 bg-panel-2 px-2 py-1 text-[11px] whitespace-nowrap text-ink shadow-lg tabular-nums"
          style={{ left: `${(hover.sx / W) * 100}%`, top: `${(hover.sy / H) * 100 - 3}%` }}
        >
          {format(hover.x, 'x')} · {format(hover.y, 'y')}
        </div>
      )}
    </div>
  )
}
