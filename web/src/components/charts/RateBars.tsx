import { prettyLabel } from '../../lib/ui'

export type RateRow = { label: string; n: number; yes: number; rate: number }

type Props = {
  rows: RateRow[]
  /** Portfolio-average conversion rate, drawn as a reference line. */
  baseline: number
  max?: number
  /** Pass labels through untouched (already-formatted band labels). */
  raw?: boolean
}

/** Row template shared by the bars and the baseline overlay so they stay aligned. */
const COLS = 'grid grid-cols-[7.5rem_1fr_3.25rem] sm:grid-cols-[9.5rem_1fr_3.25rem_4rem] gap-x-3'

/**
 * One series, one colour — bar length already encodes magnitude, so hue stays free.
 * The baseline rule is what turns each bar from a number into a judgement.
 */
export function RateBars({ rows, baseline, max, raw = false }: Props) {
  const top = max ?? Math.max(baseline, ...rows.map((r) => r.rate)) * 1.08
  const x = (v: number) => `${(v / top) * 100}%`

  return (
    <div>
      <div className="relative">
        <div aria-hidden className={`pointer-events-none absolute inset-0 ${COLS}`}>
          <span />
          <span className="relative">
            <span
              className="absolute inset-y-0 block border-l border-dashed border-ink-3/80"
              style={{ left: x(baseline) }}
            />
          </span>
        </div>

        <ul className="relative m-0 flex list-none flex-col gap-2 p-0">
          {rows.map((row) => (
            <li key={row.label} className={`${COLS} items-center`}>
              <span className="truncate text-right text-[0.8125rem] text-ink-2">
                {raw ? row.label : prettyLabel(row.label)}
              </span>
              <span className="flex h-4 items-center">
                <span
                  className="block h-full rounded-r-[4px] bg-series-1 transition-[width] duration-700 ease-out"
                  style={{ width: x(row.rate) }}
                />
              </span>
              <span className="num text-[0.8125rem] font-medium text-ink tabular-nums">
                {row.rate.toFixed(1)}%
              </span>
              <span className="num hidden text-right text-[0.75rem] text-ink-3 tabular-nums sm:block">
                {row.n.toLocaleString('en-US')}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className={`mt-3 ${COLS} text-[0.75rem] text-ink-3`}>
        <span />
        <span className="flex items-center gap-2">
          <span aria-hidden className="inline-block h-3 border-l border-dashed border-ink-3" />
          Baseline {baseline.toFixed(1)}%
        </span>
        <span />
        <span className="hidden text-right sm:block">calls</span>
      </div>
    </div>
  )
}
