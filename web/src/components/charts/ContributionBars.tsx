import type { Contribution } from '../../lib/lgbm'
import { prettyLabel } from '../../lib/ui'

const FIELD_NAMES: Record<string, string> = {
  age: 'Age',
  campaign: 'Calls so far',
  previous: 'Prior contacts',
  'cons.conf.idx': 'Consumer confidence',
  euribor3m: 'Euribor rate',
  'nr.employed': 'Employment level',
  job: 'Occupation',
  marital: 'Marital status',
  education: 'Education',
  housing: 'Housing loan',
  loan: 'Personal loan',
  contact: 'Channel',
  month: 'Month',
  day_of_week: 'Day',
  poutcome: 'Previous outcome',
  was_contacted_before: 'Contacted before',
}

/**
 * Diverging bars around a neutral zero line: blue lifts the score, red drags it.
 * The zero line moves to an edge when every contribution shares a sign, so the
 * chart never spends half its width on an empty arm.
 */
export function ContributionBars({ items }: { items: Contribution[] }) {
  const top = items.slice(0, 7)
  if (top.length === 0) {
    return (
      <p role="status" className="py-6 text-center text-[0.8125rem] leading-relaxed text-ink-3">
        This profile already matches the population baseline on every field, so there is nothing to
        attribute. Change an input to see what moves the score.
      </p>
    )
  }

  const scale = Math.max(...top.map((c) => Math.abs(c.delta)))
  const hasUp = top.some((c) => c.delta >= 0)
  const hasDown = top.some((c) => c.delta < 0)
  const both = hasUp && hasDown
  // leave room for the value label at the end of the longest bar
  const arm = both ? 44 : 78
  const zero = both ? 50 : hasDown ? 96 : 4

  return (
    <ul className="m-0 flex list-none flex-col gap-3 p-0">
      {top.map((item) => {
        const up = item.delta >= 0
        const length = (Math.abs(item.delta) / scale) * arm
        return (
          <li key={item.field} className="grid grid-cols-[9.75rem_1fr] items-center gap-3">
            <span className="min-w-0">
              <span className="block truncate text-[0.8125rem] leading-tight text-ink-2">
                {FIELD_NAMES[item.field] ?? item.field}
              </span>
              <span className="num block truncate text-[0.75rem] leading-tight text-ink-3 tabular-nums">
                {prettyLabel(item.value)}
              </span>
            </span>

            <span className="relative flex h-4 items-center">
              <span
                aria-hidden
                className="absolute inset-y-0 w-px bg-axis"
                style={{ left: `${zero}%` }}
              />
              <span
                className={`absolute h-full ${up ? 'rounded-r-[4px]' : 'rounded-l-[4px]'}`}
                style={{
                  left: up ? `${zero}%` : `${zero - length}%`,
                  width: `${length}%`,
                  background: up ? 'var(--color-series-1)' : 'var(--color-critical)',
                }}
              />
              <span
                className={`num absolute text-[0.75rem] tabular-nums ${up ? 'text-series-1' : 'text-critical'}`}
                style={
                  up
                    ? { left: `calc(${zero + length}% + 6px)` }
                    : { right: `calc(${100 - zero + length}% + 6px)` }
                }
              >
                {up ? '+' : '−'}
                {(Math.abs(item.delta) * 100).toFixed(1)}
              </span>
            </span>
          </li>
        )
      })}
    </ul>
  )
}
